
import math
import cv2
import numpy as np

import brica
from .utils import load_template, save_image

BALL_MAP_MASK_WINDOW = 10 # TODO

class OpticalFlow(object):
    def __init__(self):
        """ Calculating optical flow.
        Input image can be retina image or saliency map. 
        """
        self.last_gray_image = None
        self.hist_32 = np.zeros((128, 128), np.float32)
        
        self.inst = cv2.optflow.createOptFlow_DIS(
            cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.inst.setUseSpatialPropagation(False)
        self.flow = None
        
    def _warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res
        
    def process(self, image, is_saliency_map=False):
        if image is None:
            return

        if not is_saliency_map:
            # Input is retina image
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            # Input is saliency map
            gray_image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
            
        if self.last_gray_image is not None:
            if self.flow is not None:
                self.flow = self.inst.calc(self.last_gray_image,
                                           gray_image,
                                           self._warp_flow(self.flow, self.flow))
            else:
                self.flow = self.inst.calc(self.last_gray_image,
                                           gray_image,
                                           None)
            # (128, 128, 2)
        self.last_gray_image = gray_image
        return self.flow

class LIP(object):
    """ Retina module.

    This LIP module calculates optical flow from retina image.
    """
    
    def __init__(self):
        self.timing = brica.Timing(2, 1, 0)

        self.optical_flow = OpticalFlow()

        self.large_e_template = load_template('data/general_e0.png', (17, 17), [0, 0, 0]) # black # TODO: size
        #save_image(self.large_e_template, 'large_e.png')
        self.small_e_template = load_template('data/general_e0.png', (9, 9), [0, 0, 0]) # black # TODO: size
        #save_image(self.small_e_template, 'small_e.png')
        self.magenta_t_template = load_template('data/general_t0.png', (9, 9), [191, 0, 255]) # [191, 0, 255] == magenta # TODO: size
        #save_image(self.magenta_t_template, 'magenta_t.png')
        self.green_ball_template = load_template('data/general_round0.png', (9, 9), [0, 199, 0]) # [0, 199, 0] == green # TODO: size
        #save_image(self.green_ball_template, 'green_ball.png')
        self.blue_ball_template = load_template('data/general_round0.png', (9, 9), [199, 0, 0]) # [199, 0, 0] == blue # TODO: size
        #save_image(self.blue_ball_template, 'blue_ball.png')
        self.black_ball_template = load_template('data/general_round0.png', (9, 9), [0, 0, 0]) # [0, 0, 0] == black # TODO: size
        #save_image(self.black_ball_template, 'black_ball.png')

        self.last_optical_flow = None
        self.last_large_e_match_map = None
        self.last_small_e_match_map = None
        self.last_green_ball_match_map = None
        self.last_blue_ball_match_map = None
        self.last_black_ball_match_map = None

    def __call__(self, inputs):
        if 'from_retina' not in inputs:
            raise Exception('LIP did not recieve from Retina')

        retina_image = inputs['from_retina'] # (128, 128, 3)

        large_e_match_map = self._get_match_map(retina_image, self.large_e_template, 0.95, 0.99) # (128, 128) # TODO: adjust threshold and max_value
        small_e_match_map = self._get_match_map(retina_image, self.small_e_template, 0.7, 1.0) # (128, 128) # TODO: adjust threshold and max_value
        magenta_t_match_map = self._get_match_map(retina_image, self.magenta_t_template, 0.9, 1.0) # (128, 128) # TODO: adjust threshold and max_value
        green_ball_match_map = self._get_match_map(retina_image, self.green_ball_template, 0.85, 1.0) # (128, 128) # TODO: adjust threshold and max_value
        blue_ball_match_map = self._get_match_map(self.mask_center(retina_image), self.blue_ball_template, 0.85, 1.0) # (128, 128) # TODO: adjust threshold and max_value
        black_ball_match_map = self._get_match_map(self.mask_center(retina_image), self.black_ball_template, 0.85, 1.0) # (128, 128) # TODO: adjust threshold and max_value

        # Calculate optical flow with retina image
        optical_flow = self.optical_flow.process(retina_image, is_saliency_map=False)

        # Store maps for debug visualizer
        self.last_optical_flow = optical_flow
        self.last_large_e_match_map = large_e_match_map
        self.last_small_e_match_map = small_e_match_map
        self.last_green_ball_match_map = green_ball_match_map
        self.last_blue_ball_match_map = blue_ball_match_map
        self.last_black_ball_match_map = black_ball_match_map
        
        return dict(to_fef=(optical_flow, large_e_match_map, small_e_match_map, magenta_t_match_map, green_ball_match_map, blue_ball_match_map, black_ball_match_map))

    def _get_match_map(self, image, template, threshold, max_value): # TODO
        match = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        # https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#matchtemplate
        # if input is (W, H) and template is (w, h), output is (W-w+1, H-h+1)
        # so it needs to pad to adjust input image shape.
        # (W, H) - (W-w+1, H-h+1) = (w-1, h-1) = (pad_x, pad_y)
        pad_x = image.shape[0] - match.shape[0]
        pad_y = image.shape[1] - match.shape[1]
        match_map = np.pad(match, [(pad_x // 2, pad_x - pad_x // 2), (pad_y // 2, pad_y - pad_y // 2)], 'constant')
        match_map[match_map < threshold] = 0.0
        match_map[match_map >= threshold] = max_value
        return match_map

    def mask_center(self, image):
        h, w = image.shape[:2]
        win = BALL_MAP_MASK_WINDOW
        mask = np.zeros_like(image)
        x = w // 2
        y = h // 2
        mask[y-win:y+win,  x-win:x+win] = [1.0, 1.0, 1.0]
        return mask * image
