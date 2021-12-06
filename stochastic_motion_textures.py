import sys
import cv2
import numpy as np
from layer_inpaint import inpaint
from layer_motion import *

def get_masked_layer(image,mask):
    return image*np.repeat(mask[:,:,np.newaxis], 3, axis=2)

# Input format:
#   python stochastic_motion_textures.py image.jpg layer_1_mask.jpg layer_1_type layer_2_mask.jpg layer_2_type number_of_frames

n_argv = len(sys.argv)
original_image_filename = sys.argv[1]
frames = int(sys.argv[-1])
foreground_mask_filename = []
layer_type = []
for i in range(2,n_argv-1,2):
    foreground_mask_filename.append(sys.argv[i])
    layer_type.append(sys.argv[i + 1])

print(original_image_filename)
print(frames)
print(foreground_mask_filename)
print(layer_type)

image = cv2.imread(original_image_filename, cv2.IMREAD_COLOR)

X = input('Press anykey to begin masking (s to skip). ')
if 's' in X:
    print('Skipping masking.')
else:
    for filename in foreground_mask_filename:
        mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)/255.
        foreground_mask = mask
        background_mask = 1-mask
        cv2.imwrite(filename[:-4]+'_foreground_mask.png',255*foreground_mask)
        cv2.imwrite(filename[:-4]+'_background_mask.png',255*background_mask)
        foreground = get_masked_layer(image,foreground_mask).astype(np.uint8)
        background = get_masked_layer(image,background_mask).astype(np.uint8)
        cv2.imwrite(filename[:-4]+'_foreground_layer.png',foreground)
        cv2.imwrite(filename[:-4]+'_background_layer.png',background)

X = input('End of masking. Press anykey to begin infill (s to skip). ')
if 's' in X:
    print('Skipping infill.')
else:
    background = cv2.imread(foreground_mask_filename[0][:-4]+'_background_layer.png', cv2.IMREAD_COLOR)
    for filename in foreground_mask_filename:
        mask = cv2.imread(filename[:-4]+'_background_mask.png', cv2.IMREAD_GRAYSCALE)/255.
        mask_ = np.abs(1-np.repeat((mask)[:,:,np.newaxis], 3, axis=2))
        background = inpaint(mask_, background)
        cv2.imwrite(filename[:-4]+'_background_inpaint.jpg', background)

X = input('End of filling. Press anykey to begin motion (s to skip). ')
if 's' in X:
    print('Skipping motion.')
else:
    layers = []
    for filename, motion_type in zip(foreground_mask_filename, layer_type):
        background = cv2.imread(filename[:-4]+'_background_inpaint.jpg', cv2.IMREAD_COLOR)
        h,w,c = background.shape
        foreground = cv2.imread(filename[:-4]+'_foreground_layer.png', cv2.IMREAD_COLOR)
        alpha      = cv2.imread(filename[:-4]+'_foreground_mask.png', cv2.IMREAD_GRAYSCALE)/255.

        if motion_type == "boat":
            # Use the centroid of the boat shape as the anchor point
            # calculate moments of binary image
            M = cv2.moments(alpha)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            layer = Boat(foreground, alpha, 3.0, 3.0, frames, (cX,cY))
        elif motion_type == "plant":
            # Use the centroid of the plant shape as the anchor point
            # calculate moments of binary image
            M = cv2.moments(alpha)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            layer = Plant(foreground, alpha, 5.0, frames, (cX,cY))
        elif motion_type == "water":
            layer = Water(foreground, alpha, 0.5, frames)
        elif motion_type == "cloud":
            # For looped motion, we would want the cloud to end up back at the same
            # location it started. Therefore, the amplitude will need to equal the
            # width of the picture divided by the number of frames
            layer = Cloud(foreground, alpha, w/frames)
        else:
            print('Uknown Motion Type')
            sys.exit()
        layers.append(layer)
    # A motion texture is essentially a time-varying displacement map defined by a
    # motion type, a set of motion parameters, and in some cases a motion armature.
    # This displacement map d(p; t) is a function of pixel coordinates p and time t.
    for t in range(frames):
        frame = background
        for layer in layers:
            foreground_ = np.zeros((h, w, c))
            alpha_      = np.zeros((h,w))
            for i in range(h):
                for j in range(w):
                    point = np.array([i,j])
                    disp = layer.displacement(point, t)
                    # Instead, since our motion fields are all very smooth,
                    # we simply dilate them by the extent of the largest possible
                    # motion and reverse their sign.
                    point_ = point + disp*-1
                    #print(disp)
                    if motion_type == "cloud":
                        # Check for cloud wrap around
                        point_[0] = point_[0] % h
                        point_[1] = point_[1] % w
                    else:
                        # Check for pixels displaced outside the image
                        point_[0] = max(0,min(point_[0],h-1))
                        point_[1] = max(0,min(point_[1],w-1))
                    foreground_[i,j] = layer.image[point_[0],point_[1]]
                    alpha_[i,j]      = layer.alpha[point_[0],point_[1]]
            alpha_ = np.repeat(alpha_[:,:,np.newaxis],3,axis=2)
            frame  = foreground_*alpha_ + frame*(1-alpha_)
        cv2.imwrite(original_image_filename[:-4]+"_{0:04d}.png".format(t),frame)
