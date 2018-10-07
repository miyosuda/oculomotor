import brica
import numpy as np
import cv2
import math
from oculoenv.geom import Matrix4

class HP(object):
    """ Hippocampal formation module.

    Create allocentric panel image.
    """
    
    def __init__(self):
        self.timing = brica.Timing(2, 1, 0)

        # Allocentric panel map image
        self.map_image = np.zeros((128, 128, 3), dtype=np.uint8)

        # Allocentric panel map image
        self.not_overlaid_map_image = np.zeros((128, 128, 3), dtype=np.uint8)

        # blured Allocentric panel map image
        self.blured_map_image = np.zeros((128, 128, 3), dtype=np.uint8)

        # copied from retina.py
        width = 128
        self.blur_rates, self.inv_blur_rates = self._create_rate_datas(width, sigma=0.48, clipping_gain=1.8, gain=1.0) # original param are sigma=0.32, clipping_gain=1.2, gain=1.0
        self.inv_blur_rates += 0.2
        self.inv_blur_rates = np.clip(self.inv_blur_rates, 0.0, 1.0)
        self.blur_rates = 1.0 - self.inv_blur_rates

        self.gray_rates, self.inv_gray_rates = self._create_rate_datas(width, gain=0.5)
        self.inv_gray_rates += 0.6
        self.inv_gray_rates = np.clip(self.inv_gray_rates, 0.0, 1.0)
        self.gray_rates = 1.0 - self.inv_gray_rates

    def __call__(self, inputs):
        if 'from_retina' not in inputs:
            raise Exception('HP did not recieve from Environment')

        # This image input from environment is a kind of cheat and not biologically
        # acculate.
        if inputs['from_retina'] is not None:
            image, angle = inputs['from_retina'] # (128, 128, 3), (2)

            # Transform input image into allocentric panel image
            transforemed_image = self._extract_transformed_image(image, angle)


            # Overlay into existing map image
            self._overlay_extracted_image(self.map_image, transforemed_image)

            self.not_overlaid_map_image = transforemed_image
            self.blured_map_image = self._create_inv_retina_image(transforemed_image)
        
        return dict(to_pfc=(angle, self.map_image, self.not_overlaid_map_image, self.blured_map_image))

    def _get_perspective_mat(self, fovy, aspect_ratio, znear, zfar):
        ymax = znear * math.tan(fovy * math.pi / 360.0)
        xmax = ymax * aspect_ratio

        t  = 2.0 * znear
        t2 = 2.0 * xmax
        t3 = 2.0 * ymax
        t4 = zfar - znear
    
        m = [[t/t2,  0.0,              0.0, 0.0],
             [0.0,  t/t3,              0.0, 0.0],
             [0.0,   0.0, (-zfar-znear)/t4, -1.0],
             [0.0,   0.0,     (-t*zfar)/t4, 0.0]]
        m = np.transpose(np.array(m, dtype=np.float32))
        mat = Matrix4(m)
        return mat

    def _extract_transformed_image(self, image, angle):
        # In order to use black color as a blank mask, set lower clip value for
        # input image
        mask_threshold = 3
    
        image = np.clip(image, mask_threshold, 255)
    
        angle_h = angle[0]
        angle_v = angle[1]
    
        m0 = Matrix4()
        m1 = Matrix4()
        m0.set_rot_x(angle_v)
        m1.set_rot_y(angle_h)
        camera_mat = m1.mul(m0)
        camera_mat_inv = camera_mat.invert()

        camera_fovy = 50
        pers_mat = self._get_perspective_mat(camera_fovy, 1.0, 0.04, 100.0)

        mat = pers_mat.mul(camera_mat_inv)
    
        plane_distance = 3.0
        
        point_srcs = [[ 1.0, 1.0, -plane_distance, 1.0],
                      [-1.0, 1.0, -plane_distance, 1.0],
                      [-1.0,-1.0, -plane_distance, 1.0],
                      [ 1.0,-1.0, -plane_distance, 1.0]]

        point_src_2ds = []
        point_dst_2ds = []
    
        for point_src in point_srcs:
            ps_x = (point_src[0] * 0.5 + 0.5) * 127.0
            ps_y = (-point_src[1] * 0.5 + 0.5) * 127.0
            point_src_2ds.append([ps_x, ps_y])
        
            p = mat.transform(np.array(point_src, dtype=np.float32))
            w = p[3]
            x = p[0]/w
            y = p[1]/w
            pd_x = (x * 0.5 + 0.5) * 127.0
            pd_y = (-y * 0.5 + 0.5) * 127.0
            point_dst_2ds.append([pd_x, pd_y])

        point_src_2ds = np.float32(point_src_2ds)
        point_dst_2ds = np.float32(point_dst_2ds)

        h,w,c = image.shape

        M = cv2.getPerspectiveTransform(point_dst_2ds, point_src_2ds)
        transformed_image = cv2.warpPerspective(image, M, (h,w))
        return transformed_image

    def _overlay_extracted_image(self, base_image, ext_image):
        GRID_DIVISION = 8
        GRID_WIDTH = 128 // GRID_DIVISION
    
        for ix in range(GRID_DIVISION):
            pixel_x = GRID_WIDTH * ix
            for iy in range(GRID_DIVISION):
                pixel_y = GRID_WIDTH * iy
                base_region_image = base_image[pixel_y:pixel_y+GRID_WIDTH,
                                               pixel_x:pixel_x+GRID_WIDTH, :]
                ext_region_image = ext_image[pixel_y:pixel_y+GRID_WIDTH,
                                             pixel_x:pixel_x+GRID_WIDTH, :]
                ext_region_image_sum = np.sum(ext_region_image, axis=2)
                has_zero = np.any(ext_region_image_sum==0)
                if not has_zero:
                    base_image[pixel_y:pixel_y+GRID_WIDTH,
                               pixel_x:pixel_x+GRID_WIDTH, :] = ext_region_image // 2 + base_region_image // 2

    # copied from retina.py
    def _gauss(self, x, sigma):
        sigma_sq = sigma * sigma
        return 1.0 / np.sqrt(2.0 * np.pi * sigma_sq) * np.exp(-x*x/(2 * sigma_sq))

    # copied from retina.py
    def _create_rate_datas(self, width, sigma=0.32, clipping_gain=1.2, gain=1.0):
        """ Create mixing rate.
        Arguments:
            width: (int) width of the target image.
            sigma: (float) standard deviation of the gaussian.
            clipping_gain: (float) To make the top of the curve flat, apply gain > 1.0
            gain: (float) Final gain for the mixing rate. 
                          e.g.) if gain=0.8, mixing rates => 0.2~1.0
        Returns:
            Float ndarray (128, 128, 1): Mixing rates and inverted mixing rates. 
        """
        rates = [0.0] * (width * width)
        hw = width // 2
        for i in range(width):
            x = (i - hw) / float(hw)
            for j in range(width):
                y = (j - hw) / float(hw)
                r = np.sqrt(x*x + y*y)
                rates[j*width + i] = self._gauss(r, sigma=sigma)
        rates = np.array(rates)
        # Normalize
        rates = rates / np.max(rates)
        
        # Make top flat by multipying and clipping 
        rates = rates * clipping_gain
        rates = np.clip(rates, 0.0, 1.0)

        # Apply final gain
        if gain != 1.0:
            rates = rates * gain + (1-gain)
        rates = rates.reshape([width, width, 1])
        inv_rates = 1.0 - rates
        return rates, inv_rates

    # copied from retina.py
    def _create_blur_image(self, image):
        h = image.shape[0]
        w = image.shape[1]

        # Resizeing to 1/2 size
        resized_image0 = cv2.resize(image,
                                  dsize=(h//2, w//2),
                                  interpolation=cv2.INTER_LINEAR)
        # Resizeing to 1/4 size
        resized_image1 = cv2.resize(resized_image0,
                                  dsize=(h//4, w//4),
                                  interpolation=cv2.INTER_LINEAR)
        # Resizeing to 1/8 size
        resized_image2 = cv2.resize(resized_image1,
                                  dsize=(h//8, w//8),
                                  interpolation=cv2.INTER_LINEAR)
        
        # Resizing to original size
        blur_image = cv2.resize(resized_image2,
                                dsize=(h, w),
                                interpolation=cv2.INTER_LINEAR)

        # Conver to Grayscale
        gray_blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
        gray_blur_image = np.reshape(gray_blur_image,
                                     [gray_blur_image.shape[0],
                                      gray_blur_image.shape[0], 1])
        gray_blur_image = np.tile(gray_blur_image, 3)
        return blur_image, gray_blur_image

    # copied from retina.py
    def _create_inv_retina_image(self, image):
        blur_image, gray_blur_image = self._create_blur_image(image)
        # Mix original and blur image
        #blur_mix_image = image * self.blur_rates + blur_image * self.inv_blur_rates
        blur_mix_image = image * self.inv_blur_rates + blur_image * self.blur_rates
        # Mix blur mixed image and gray blur image.
        #gray_mix_image = blur_mix_image * self.gray_rates + gray_blur_image * self.inv_gray_rates
        gray_mix_image = blur_mix_image * self.inv_gray_rates + gray_blur_image * self.gray_rates
        return gray_mix_image.astype(np.uint8)
