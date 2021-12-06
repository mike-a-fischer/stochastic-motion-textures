import sys
import cv2
import numpy as np

# The spectral method for synthesizing a stochastic field has three steps:
# (1) generate a complex Gaussian random field in the frequency domain,
# (2) apply a domain-specific spectrum filter, and
# (3) compute the inverse Fourier transform to synthesize a stochastic field
#     in the time or frequency domain.

class Boat():
    # A ship moving on the surface of open water is almost always in
    # oscillatory motion. Hence, the simplest model is to assign a sinusoidal
    # translation and a sinusoidal rotation.
    def __init__(self, image, alpha, amp, rolling, frames, anchor):
        self.image = image
        self.alpha = alpha
        self.amp = amp
        self.rolling = rolling
        self.anchor = np.array(anchor)
        self.omega_h = np.concatenate((
                        np.linspace(-1*np.pi, np.pi, num=frames//2, endpoint=False),
                        np.linspace(np.pi, -1*np.pi, num=frames//2, endpoint=False))) #radians/frame
        self.omega_t = np.concatenate((
            np.linspace(-rolling/180.*np.pi, rolling/180.*np.pi, num=frames//2, endpoint=False),
            np.linspace(rolling/180.*np.pi, -rolling/180.*np.pi, num=frames//2, endpoint=False))) #radians/frame

    def displacement(self,p,t):
        # Vertical heaving oscillation
        dh = self.amp * (np.sin(self.omega_h[t])+np.sin(2*self.omega_h[t]))
        # Harmonic rolling around anchor point
        theta = self.omega_t[t]
        R = np.array([[ np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        v = np.array(p - self.anchor).astype(np.float64)
        v_ = np.matmul(R, v.T)
        d = v - v_
        dh = dh + d[0]
        dw = d[1]
        return np.array([dh, dw]).astype(np.int)

class Plant():
    def __init__(self, image, alpha, rolling, frames, anchor):
        self.image = image
        self.alpha = alpha
        self.rolling = rolling
        self.anchor = np.array(anchor)
        self.omega_t = np.concatenate((
            np.linspace(-rolling/180.*np.pi, rolling/180.*np.pi, num=frames//2, endpoint=False),
            np.linspace(rolling/180.*np.pi, -rolling/180.*np.pi, num=frames//2, endpoint=False))) #radians/frame

    def displacement(self,p,t):
        # Harmonic rolling around anchor point
        theta = self.omega_t[t]
        R = np.array([[ np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        v = np.array(p - self.anchor).astype(np.float64)
        v_ = np.matmul(R, v.T)
        d = v - v_
        dh = d[0]
        dw = d[1]
        return np.array([dh, dw]).astype(np.int)

class Water():
    def __init__(self, image, alpha, amp, frames):
        self.image = image
        self.alpha = alpha
        self.amp = amp
        self.omega_h = np.concatenate((
                        np.linspace(-1*np.pi, np.pi, num=frames//2, endpoint=False),
                        np.linspace(np.pi, -1*np.pi, num=frames//2, endpoint=False))) #radians/frame

    def displacement(self,p,t):
        # They target a similar set of natural phenomena to those we study:
        # plants, waves, and boats, which can all be explained as harmonic
        # oscillations.

        # In order to model a resonable water effect without having to account
        # for projections and doing a wave height model, the choice is to use
        # multiple harmonic oscillations to minimic ripples at different rates.
        # However, I need to apply a phase shift that is pixel h dependent,
        # otherwise this just replicates the boat up and down motion.
        # Also note that I chose 3 different rates of oscillation, none of which
        # are harmonics of the others.
        h, w = p
        dh = np.sin(2*self.omega_h[t] + h%20) \
           + np.sin(3*self.omega_h[t] + h%30) \
           + np.sin(5*self.omega_h[t] + h%50)
        # Water looks unnatural side to side so setting dw to zero
        dw = 0
        return np.array([dh,dw]).astype(np.int)

class Cloud():
    # Since clouds often move very slowly and their motion does not attract too
    # much attention, we simply assign a translational motion field to them.
    # We extend the clouds outside the image frame to create a cyclic
    # texture using our inpainting algorithm, since their motion in one
    # direction will create holes that we have to fill.
    def __init__(self, image, alpha, amp):
        self.image = image
        self.alpha = alpha
        self.amp = amp

    def displacement(self,p,t):
        dh = 0 # assume no vertical displacement of cloud, could add slight bobbing later
        dw = int(t*self.amp)
        return np.array([dh,dw]).astype(np.int)

if __name__ == "__main__":

    # currently only supports a single foreground layer

    background_filename = sys.argv[1]
    foreground_filename = sys.argv[2]
    alpha_filename      = sys.argv[3]
    motion_type         = sys.argv[4]
    frames              = int(sys.argv[5])

    background = cv2.imread(background_filename, cv2.IMREAD_COLOR)
    foreground = cv2.imread(foreground_filename, cv2.IMREAD_COLOR)
    alpha      = cv2.imread(alpha_filename, cv2.IMREAD_GRAYSCALE)/255.
    h,w,c = foreground.shape

    layers = []

    if motion_type == "boat":
        # Use the centroid of the boat shape as the anchor point
        # calculate moments of binary image
        M = cv2.moments(alpha)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        layer = Boat(foreground, alpha, 5.0, 5.0, frames, (cX,cY))
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
            frame  = foreground_*alpha_ + background*(1-alpha_)
        cv2.imwrite(str(t)+".png",frame)
