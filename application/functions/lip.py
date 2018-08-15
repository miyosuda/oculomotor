import cv2
import numpy as np

import brica


GAUSSIAN_KERNEL_SIZE = (5,5)


class LIP(object):
    """ Retina module.

    This LIP module calculates saliency map from retina image.
    """
    
    def __init__(self):
        self.timing = brica.Timing(2, 1, 0)

    def __call__(self, inputs):
        if 'from_retina' not in inputs:
            raise Exception('LIP did not recieve from Retina')

        retina_image = inputs['from_retina'] # (128, 128, 3)
        
        saliency_map = self._get_saliency_map(retina_image) # (128, 128)
        
        return dict(to_fef=saliency_map)

    def _get_saliency_magnitude(self, image):
        # Calculate FFT
        dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude, angle = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])

        log_magnitude = np.log10(magnitude.clip(min=1e-10))

        # Apply box filter
        log_magnitude_filtered = cv2.blur(log_magnitude, ksize=(3, 3))

        # Calculate residual
        magnitude_residual = np.exp(log_magnitude - log_magnitude_filtered)

        # Apply residual magnitude back to frequency domain
        dft[:, :, 0], dft[:, :, 1] = cv2.polarToCart(magnitude_residual, angle)
    
        # Calculate Inverse FFT
        image_processed = cv2.idft(dft)
        magnitude, _ = cv2.cartToPolar(image_processed[:, :, 0],
                                   image_processed[:, :, 1])
        return magnitude

    def _get_saliency_map(self, image):
        resize_shape = (64, 64) # (h,w)

        # Size argument of resize() is (w,h) while image shape is (h,w,c)
        image_resized = cv2.resize(image, resize_shape[1::-1])
        # (64,64,3)

        saliency = np.zeros_like(image_resized, dtype=np.float32)
        # (64,64,3)
    
        channel_size = image_resized.shape[2]
    
        for ch in range(channel_size):
            ch_image = image_resized[:, :, ch]
            saliency[:, :, ch] = self._get_saliency_magnitude(ch_image)

        # Calclate mean over channels
        saliency = np.mean(saliency, 2)
        # (64,64)

        saliency = cv2.GaussianBlur(saliency, GAUSSIAN_KERNEL_SIZE, sigmaX=8, sigmaY=0)
        saliency = (saliency ** 2)
        saliency = saliency / np.max(saliency) # Normalize
    
        # Resize to original size
        saliency = cv2.resize(saliency, image.shape[1::-1])
        return saliency
