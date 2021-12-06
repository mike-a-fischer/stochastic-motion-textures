import sys
import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage import color
import numba

# We provide a default window size of 9 × 9 pixels
# However, for speed reasons, I am increasing the size of the window to 15
WINDOW = 15
HALF_WINDOW = WINDOW//2

def identify_fill_front(mask):
    # https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
    fill_front = np.where(cv2.Laplacian(mask, cv2.CV_64F) < 0)
    return fill_front

def compute_priorities(fill_image, fill_front, confidence, target_mask):
    # Given a patch Ψp centred at the point p for some p ∈ δΩ (see fig. 3),
    # its priority P (p) is defined as the product of two terms:
    #           P(p) = C(p)D(p). (1)
    conf_term = calculate_confidence_term(confidence, fill_front)
    data_term = calculate_data_term(fill_image,fill_front, target_mask)
    priorities = conf_term * data_term
    return priorities[fill_front[0], fill_front[1]]

def calculate_confidence_term(confidence, fill_front):
    # The confidence term C(p) may be thought of as a measure of the amount of
    # reliable information surrounding the pixel p. The intention is to fill
    # first those patches which have more of their pixels already filled, with
    # additional preference given to pixels that were filled early on (or that
    # were never part of the target region).

    conf_term = np.zeros(confidence.shape[:2])
    for i,j in zip(fill_front[0], fill_front[1]):
        patch, area = generate_patch((i, j), confidence)
        h, w = patch.shape[:2]
        conf_term[i, j] = np.sum(patch) / area
    return conf_term

def calculate_data_term(fill_image, fill_front, target_mask):
    # The data term D(p) is a function of the strength of isophotes hitting the
    # front δΩ at each iteration. This term boosts the priority of a patch that
    # an isophote “flows” into. This factor is of fundamental importance in our
    # algorithm because it encourages linear structures to be synthesized first,
    # and, therefore propagated securely into the target region. Broken lines
    # tend to connect, thus realizing the “Connectivity Principle” of vision
    # psychology [7, 17] (cf., fig. 4, fig. 7d, fig. 8b and fig. 13d).
    #           D(p) = |∇Ip⊥ · np| / α

    # Generate Gradient Magnitudes
    gradients = np.gradient(cv2.cvtColor(fill_image, cv2.COLOR_BGR2GRAY)/255.)
    gradient_magnitudes = np.linalg.norm(gradients, axis=0)
    max_gradient_magnitudes = np.zeros(fill_image.shape[:2])
    for fill_y, fill_x in zip(fill_front[0], fill_front[1]):
        patch, _ = generate_patch((fill_y, fill_x), gradient_magnitudes)
        max_gradient_magnitudes[fill_y, fill_x] = np.max(patch)

    # np is a unit vector orthogonal to the front δΩ in the point p
    #   To obtain the normal unit vectors at the front δΩ, using the taking the
    #   derivatives in x and y will yield the direction of change.
    # https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html

    normals = np.zeros(target_mask.shape)
    kernel_dh = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    kernel_dw = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    # https://stackoverflow.com/questions/26857829/does-filter2d-in-opencv-really-do-its-job
    #dh = cv2.filter2D(target_mask,-1,kernel_dh)
    #dw = cv2.filter2D(target_mask,-1,kernel_dw)
    dh = convolve2d(target_mask, kernel_dh, mode='same')
    dw = convolve2d(target_mask, kernel_dw, mode='same')

    norm = np.sqrt(dh**2+dw**2)
    norm[norm == 0] = 0.0001 # ensure I dont get a divide by zero error
    dh /= norm
    dw /= norm
    # (e.g., α = 255 for a typical grey-level image)
    alpha = 255.
    data_term = np.sqrt((dh * max_gradient_magnitudes)**2 \
                      + (dw * max_gradient_magnitudes)**2) / alpha
    return data_term

def generate_patch(point, image):
    # need to handle the both the 2D and 3D input images
    if (len(image.shape) < 3):
        image = image[:,:,np.newaxis]
    h, w, c = image.shape
    image_points = np.array([[max(point[0] - HALF_WINDOW, 0), min(point[0] + HALF_WINDOW + 1, h-1)],
                    [max(point[1] - HALF_WINDOW, 0), min(point[1] + HALF_WINDOW + 1, w-1)]]).astype(np.int)
    patch_points = np.array([[image_points[0,0] - point[0] + HALF_WINDOW, image_points[0,1] - point[0] + HALF_WINDOW],
                    [image_points[1,0] - point[1] + HALF_WINDOW, image_points[1,1] - point[1] + HALF_WINDOW]]).astype(np.int)
    patch = np.zeros((WINDOW, WINDOW, c))
    patch[patch_points[0,0]:patch_points[0,1],
          patch_points[1,0]:patch_points[1,1]] = \
                        image[image_points[0,0]:image_points[0,1],
                              image_points[1,0]:image_points[1,1]]
    area = (patch_points[0,1] - patch_points[0,0]) * (patch_points[1,1] - patch_points[1,0])
    return patch, area

@numba.jit()
def find_exemplar_patch(point, fill_image, target_mask):
    # We use the CIE Lab colour space because of its property of perceptual uniformity [18]
    # https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    # https://en.wikipedia.org/wiki/CIELAB_color_space
    #cielab_image = cv2.cvtColor(fill_image.astype(np.float32)/255., cv2.COLOR_BGR2Lab)
    cielab_image = color.rgb2lab(fill_image)
    h,w,d = cielab_image.shape
    target_patch, _  = generate_patch(point, cielab_image)
    # Only use the part of the image that is going to be used to compare to
    target_patch_mask, _  = generate_patch(point, target_mask)
    # Invert to select the painted points and not the blank points
    target_patch_mask = 1 - target_patch_mask
    # Make sure the target patch is adjusted per the inverted mask
    target_patch = target_patch_mask*target_patch
    min_ssd = float('Inf')
    exemplar_point = 0, 0
    # Iterate over the image and look for similar patches
    for i in range(HALF_WINDOW + 1, h - HALF_WINDOW):
        for j in range(HALF_WINDOW + 1, w - HALF_WINDOW):
            # Dont take any patch that includes any part of the target fill area
            new_mask, _ = generate_patch((i, j), target_mask)
            if not np.isin(1, new_mask):
                exemplar_patch, _ = generate_patch((i, j), cielab_image)
                exemplar_patch = target_patch_mask * exemplar_patch
                ssd = np.sum((target_patch - exemplar_patch)**2)
                if min_ssd > ssd:
                    min_ssd = ssd
                    exemplar_point = (i, j)
    return exemplar_point

def copy_image_data(maximum_point, exemplar_point, fill_image, target_mask, confidence):
    h, w   = fill_image.shape[:2]
    mi, mj = maximum_point
    ei, ej = exemplar_point

    for i in range(-HALF_WINDOW, HALF_WINDOW+1):
        for j in range(-HALF_WINDOW, HALF_WINDOW+1):
            image_h, image_w = min(max(mi - i,0),h-1), min(max(mj - j,0),w-1)
            patch_h, patch_w = min(max(ei - i,0),h-1), min(max(ej - j,0),w-1)
            # replace only targeted locations
            if target_mask[image_h, image_w] == 1:
                # Update image with patch
                fill_image[image_h, image_w] = fill_image[patch_h, patch_w]
                # Update mask to indicate these pixels have been filled
                target_mask[image_h, image_w] = 0
                # After the patch Ψpˆ has been filled with new pixel values,
                # the confidence C(p) is updated in the area delimited by Ψpˆ
                # as follows:
                #               C(q)=C(pˆ) ∀q∈Ψpˆ ∩Ω.
                confidence[image_h, image_w] = confidence[mi, mj]
    return fill_image, target_mask, confidence

@numba.jit()
def inpaint(mask, fill_image):
    target = np.where(mask > 0)
    target_mask = np.zeros(fill_image.shape[:2])
    target_mask[target[0], target[1]] = 1
    # During initialization, the function C(p) is set to
    # C(p) = 0 ∀p∈Ω, and C(p)=1 ∀p∈I−Ω.
    confidence = np.ones(fill_image.shape[:2])
    confidence[target[0], target[1]] = 0

    # Extract the manually selected initial front δΩ_0.
    # Repeat until done:
    done = False
    while not done:
        #   1a. Identify the fill front δΩt. If Ωt = ∅, exit.
        fill_front = identify_fill_front(target_mask)
        #   1b. Compute priorities P (p) ∀p ∈ δΩt.
        priorities = compute_priorities(fill_image,
                                        fill_front,
                                        confidence,
                                        target_mask)
        #   2a. Find the patch Ψpˆ with the maximum priority,
        #       i.e., Ψpˆ |pˆ=argmaxp∈δΩt P(p)
        max_priority = priorities.argmax()
        maximum_point = (fill_front[0][max_priority],fill_front[1][max_priority])
        #   2b. Find the exemplar Ψqˆ ∈Φ that minimizes d(Ψpˆ,Ψqˆ).
        exemplar_point = find_exemplar_patch(maximum_point, fill_image, target_mask)
        #   2c. Copy image data from Ψqˆ to Ψpˆ.
        #   3. Update C(p) ∀p |p∈Ψpˆ ∩Ω
        fill_image, target_mask, confidence = copy_image_data(maximum_point,
                                                              exemplar_point,
                                                              fill_image,
                                                              target_mask,
                                                              confidence)
        print(np.sum(target_mask))
        if np.sum(target_mask) == 0:
            done = True
    return fill_image

if __name__ == '__main__':

    # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_cvpr2003.pdf

    background_filename = sys.argv[1]
    mask_filename = sys.argv[2]
    background = cv2.imread(background_filename, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)/255.
    mask_ = np.abs(1-np.repeat((mask)[:,:,np.newaxis], 3, axis=2))
    filled_background = inpaint(mask_, background)
    cv2.imwrite("background_inpaint__new.jpg", filled_background)
