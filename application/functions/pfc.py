import os
from collections import deque
import cv2
import numpy as np

import brica
from .utils import load_image
from .pfc_task_detection import PFCTaskDetection

from oculoenv.environment import CAMERA_VERTICAL_ANGLE_MAX
from oculoenv.environment import CAMERA_HORIZONTAL_ANGLE_MAX

"""
This is a sample implemention of PFC (Prefrontal cortex) module.
You can change this as you like.
"""

MAX_AVOID_STEPS = 50 # TODO
MAX_WANDER_STEPS = 150 # TODO
MAX_EXPLORE_STEPS = 30 # TODO

ODD_ONE_OUT_MAX_WANDER_STEPS = 500 # TODO

CENTER_THRESHOLD = 0.05

# to adjust BriCA's timing delay
BRICA_TIMING_DELAY_STEPS = 2

POINT_TO_TARGET_COOLDOWN_STEPS = 5 # TODO
CHANGE_DETECTION_COOLDOWN_STEPS = 5 # TODO
ODD_ONE_OUT_COOLDOWN_STEPS = 20 # TODO
VISUAL_SEARCH_COOLDOWN_STEPS = 50 # TODO
RANDOM_DOT_MOTION_DISCRIMINATION_COOLDOWN_STEPS = 30 # TODO
MULTIPLE_OBJECT_TRACKING_COOLDOWN_STEPS = 20 # TODO

ANGLE_TO_PANEL_RATE = 0.55 # TODO

SEARCH_MAP_BACKGROUND = 0.1
SEARCH_MAP_MARGIN = 18
SEARCH_MAP_MASK_WINDOW = 8 # TODO

GAUSSIAN_KERNEL_SIZE = (5,5)

class Task(object):
    POINT_TO_TARGET = 1
    CHANGE_DETECTION = 2
    ODD_ONE_OUT = 3
    VISUAL_SEARCH = 4
    MULTIPLE_OBJECT_TRACKING = 5
    RANDOM_DOT_MOTION_DISCRIMINATION = 6

class Phase(object):
    # Subsumption Architecture
    AVOID = 1 # Avoiding obstable phase
    WANDER = 2 # Wandering randomly phase
    EXPLORE = 3 # Exploring target phase

class InternalPhase(object):
    # several tasks have internal phase
    START = 1
    LEARNING = 2
    INTERVAL = 3
    EVALUATION = 4

TARGETS = np.array([
    [0, 0], [-0.08, -0.26], [0.08, -0.26], [0.26, 0], [-0.26, 0],
    [ 0, -0.24],
    [-0.16970562748, -0.16970562748],
    [-0.24,  0],
    [-0.16970562748,  0.16970562748],
    [ 0,  0.24],
    [ 0.16970562748,  0.16970562748],
    [ 0.24,  0],
    [ 0.16970562748, -0.16970562748]
])

# DIRECTION_TARGETS = 0.24 * np.array([
#     [ 0, -1],
#     [-0.7071067811865476, -0.7071067811865476],
#     [-1,  0],
#     [-0.7071067811865476,  0.7071067811865476],
#     [ 0,  1],
#     [ 0.7071067811865476,  0.7071067811865476],
#     [ 1,  0],
#     [ 0.7071067811865476, -0.7071067811865476]
# ])

class PFC(object):
    def __init__(self):
        self.timing = brica.Timing(3, 1, 0)

        self.pfc_td = PFCTaskDetection()

        self.task = Task.POINT_TO_TARGET
        self.phase = Phase.AVOID
        self.internal_phase = None
        self.target = 0
        self.steps = 0

        self.working_memory = deque(maxlen=3)

        self.search_mask = self._init_search_mask()
        self.y_n_button_mask = self._init_y_n_button_mask()

        self.last_memory_image = None
        self.last_saliency_map = None
        self.last_search_map = None
        self.last_change_map = None

    def __call__(self, inputs):
        if 'from_vc' not in inputs:
            raise Exception('PFC did not recieve from VC')
        if 'from_fef' not in inputs:
            raise Exception('PFC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('PFC did not recieve from BG')
        if 'from_hp' not in inputs:
            raise Exception('PFC did not recieve from HP')

        # Image from Visual cortex module.
        retina_image = inputs['from_vc']
        # Allocentric map image from hippocampal formatin module.
        angle, allocentric_image, not_overlaid_allocentric_image, blured_allocentric_image = inputs['from_hp']

        reward, done, next_phase, next_target = inputs['from_bg'] if inputs['from_bg'] else (0, False, None, None)

        # TODO
        if reward != 0 or done:
            # self.task = Task.POINT_TO_TARGET
            self.phase = Phase.AVOID
            self.internal_phase = None
            self.target = 0
            self.steps = 0
            self.working_memory.clear()
            self.search_mask = self._init_search_mask()
        else:
            self.steps += 1
            if self.phase == Phase.AVOID:
                self.target = 0
                self.working_memory.clear()
                if self.steps > MAX_AVOID_STEPS or self._is_center(angle):
                    self.task = self._detect_task(retina_image)
                    self.search_mask = self._init_search_mask()
                    self.phase = Phase.WANDER
                    self.target = next_target
                    self.steps = 0
            elif self.phase == Phase.WANDER:
                if self.task == Task.ODD_ONE_OUT: # no EXPLORE phase in ODD_ONE_OUT
                    if self.steps > ODD_ONE_OUT_MAX_WANDER_STEPS: # must be stuck
                        self.phase = Phase.AVOID
                        self.target = 0
                        self.steps = 0
                elif self.steps > MAX_WANDER_STEPS or next_phase == Phase.EXPLORE:
                    self.phase = Phase.EXPLORE
                    self.target = next_target
                    self.steps = 0
            elif self.phase == Phase.EXPLORE:
                self.target = next_target
                if self.steps > MAX_EXPLORE_STEPS or next_phase == Phase.AVOID: # must be stuck
                    self.phase = Phase.AVOID
                    self.target = 0
                    self.steps = 0
            else:
                assert self.phase in (Phase.AVOID, Phase.WANDER, Phase.EXPLORE)

        memory_image = self.working_memory[-1] if len(self.working_memory) > 0 else None
        self.last_memory_image = memory_image

        diff_map = self._get_diff_map(self.working_memory[-3], self.working_memory[-1]) if len(self.working_memory) > 2 else None
        if self.task == Task.ODD_ONE_OUT and diff_map is not None and (np.max(diff_map) > 5).any():
            search_map = self._get_search_map(angle[0], angle[1], diff_map, self.search_mask)
        else:
            search_map = self._get_search_map(angle[0], angle[1], memory_image, self.search_mask)
        self.last_search_map = search_map

        self.internal_phase = self._detect_internal_phase(retina_image, self.phase, self.steps)
        if self.task == Task.CHANGE_DETECTION:
            change_map = self.y_n_button_mask * self._get_diff_map(memory_image, retina_image)
            if self.internal_phase == InternalPhase.LEARNING:
                self.working_memory.append(np.copy(retina_image))
        elif self.task == Task.POINT_TO_TARGET:
            change_map = self._get_diff_map(memory_image, not_overlaid_allocentric_image)
            if self.internal_phase == InternalPhase.INTERVAL:
                self.working_memory.append(np.copy(not_overlaid_allocentric_image))
            #self.working_memory.append(np.copy(allocentric_image))
        elif self.task == Task.ODD_ONE_OUT:
            change_map = self._get_diff_map(memory_image, blured_allocentric_image)
            if self.internal_phase == InternalPhase.INTERVAL:
                self.working_memory.append(np.copy(blured_allocentric_image))
            #self.working_memory.append(np.copy(blured_allocentric_image))
        else:
            change_map = self._get_diff_map(memory_image, retina_image)
            self.working_memory.append(np.copy(retina_image))

        self.last_change_map = change_map

        return dict(to_fef=(self.task, self.phase, self.internal_phase, self.target, angle, search_map, change_map),
                    to_bg=(self.task, self.phase, self.internal_phase, self.target))

    def _detect_internal_phase(self, retina_image, phase, steps):
        # TODO: supervised learning
        if phase == Phase.AVOID:
            return InternalPhase.START # could be wrong
        elif phase == Phase.EXPLORE:
            return InternalPhase.EVALUATION # could be wrong
        elif phase == Phase.WANDER:
            if self.task == Task.POINT_TO_TARGET:
                if steps < POINT_TO_TARGET_COOLDOWN_STEPS - BRICA_TIMING_DELAY_STEPS: # must wait 15 steps to take a cool-down period for allocentric_image
                    return InternalPhase.INTERVAL
            elif self.task == Task.CHANGE_DETECTION:
                # from oculoenv/oculoenv/contents/change_detection_content.py
                # self.max_learning_count = 20
                # self.max_interval_count = 10
                if steps < 20 - BRICA_TIMING_DELAY_STEPS:
                    return InternalPhase.LEARNING
                if steps < 20 + 10 + CHANGE_DETECTION_COOLDOWN_STEPS - BRICA_TIMING_DELAY_STEPS: # must wait 15 steps to take a cool-down period for change_likelihood
                    return InternalPhase.INTERVAL
            elif self.task == Task.ODD_ONE_OUT:
                if steps < ODD_ONE_OUT_COOLDOWN_STEPS - BRICA_TIMING_DELAY_STEPS: # must wait 15 steps to take a cool-down period for allocentric_image
                    return InternalPhase.INTERVAL
            elif self.task == Task.VISUAL_SEARCH:
                if steps < VISUAL_SEARCH_COOLDOWN_STEPS - BRICA_TIMING_DELAY_STEPS: # must wait 15 steps to take a cool-down period for template_likelihood
                    return InternalPhase.INTERVAL
            elif self.task == Task.MULTIPLE_OBJECT_TRACKING:
                # from oculoenv/oculoenv/contents/multiple_object_tracking_content.py
                # MEMORY_STEP_COUNT = 30
                # MOVE_STEP_COUNT = 60
                if steps < 30 - BRICA_TIMING_DELAY_STEPS:
                    return InternalPhase.LEARNING
                if steps < 30 + 60 + MULTIPLE_OBJECT_TRACKING_COOLDOWN_STEPS - BRICA_TIMING_DELAY_STEPS: # must wait 10 steps to take a cool-down period for template_likelihood
                    return InternalPhase.INTERVAL
            elif self.task == Task.RANDOM_DOT_MOTION_DISCRIMINATION:
                if steps < RANDOM_DOT_MOTION_DISCRIMINATION_COOLDOWN_STEPS - BRICA_TIMING_DELAY_STEPS: # must wait 30 steps to take a cool-down period for direction_likelihood
                    return InternalPhase.INTERVAL
            else:
                assert task in (Task.POINT_TO_TARGET, Task.CHANGE_DETECTION, Task.ODD_ONE_OUT, Task.VISUAL_SEARCH, Task.MULTIPLE_OBJECT_TRACKING, Task.RANDOM_DOT_MOTION_DISCRIMINATION)
            return InternalPhase.EVALUATION
        else:
            assert self.phase in (Phase.AVOID, Phase.WANDER, Phase.EXPLORE)

    def _is_center(self, angle):
        return np.linalg.norm(angle) < CENTER_THRESHOLD

    def _detect_task(self, retina_image):
        task_id, task_name = self.pfc_td.test_image(retina_image)
        #print(task_id, task_name)
        #return Task.ODD_ONE_OUT
        return task_id

    def _init_search_mask(self):
        w, h = 128, 128
        search_mask = np.zeros((w*3, h*3), np.float32)
        m = SEARCH_MAP_MARGIN
        search_mask[h+m:h*2-m, w+m:w*2-m] = 1.0
        return search_mask

    def _get_search_map(self, angle_h, angle_v, image, search_mask):
        w, h = 128, 128
        search_map = np.zeros((w*3, h*3), np.float32)
        x, y = self._angle_to_panel_coordinate(angle_h, angle_v, w, h)
        #print((angle_h, angle_v), self._panel_coordinate_to_angle(x, y, w, h))
        x += w
        y += h
        m = SEARCH_MAP_MARGIN
        win = SEARCH_MAP_MASK_WINDOW
        search_map[h+m:h*2-m, w+m:w*2-m] = SEARCH_MAP_BACKGROUND
        if image is not None:
            saliency_map = self._get_saliency_map(image)
            self.last_saliency_map = saliency_map
            search_map[h+m:h*2-m, w+m:w*2-m] += cv2.resize(saliency_map, (w-m*2, h-m*2))
        self.search_mask[y-win:y+win, x-win:x+win] = 0.0
        return (search_map * search_mask)[y-h//2:y+h//2, x-w//2:x+w//2]

    def _angle_to_panel_coordinate(self, angle_h, angle_v, w, h):
        # x = (1.0 + angle_h / unit) * 0.5 * w
        angle_h_unit = CAMERA_HORIZONTAL_ANGLE_MAX * ANGLE_TO_PANEL_RATE
        angle_v_unit = CAMERA_VERTICAL_ANGLE_MAX * ANGLE_TO_PANEL_RATE
        x = (1.0 - angle_h / angle_h_unit) * 0.5 * w
        y = (1.0 - angle_v / angle_v_unit) * 0.5 * h
        x = np.clip(x, 0, w - 1).astype(np.uint8)
        y = np.clip(y, 0, h - 1).astype(np.uint8)
        return x, y

    def load_model(self, file_path):
        module_dir, _ = os.path.split(os.path.realpath(__file__))
        absolute_path = os.path.join(module_dir, file_path)
        self.pfc_td.load_model(absolute_path)

    def _init_y_n_button_mask(self):
        # mask yes/no button on the left/right.
        button_mask = np.ones((128, 128, 3), np.float32)
        button_mask[64-14:64+14,      0: 36] = [0, 0, 0] # TODO
        button_mask[64-14:64+14, 128-36:128] = [0, 0, 0] # TODO
        return button_mask

    def _get_diff_map(self, prev_image, current_image):
        if current_image is None or prev_image is None:
            return np.zeros((128, 128, 3), np.float32)
        return cv2.absdiff(prev_image, current_image)

    # copied from lip.py
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

    # copied from lip.py
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

        # Calclate max over channels
        saliency = np.max(saliency, axis=2)
        # (64,64)

        saliency = cv2.GaussianBlur(saliency, GAUSSIAN_KERNEL_SIZE, sigmaX=8, sigmaY=0)

        #SALIENCY_ENHANCE_COEFF = 2.0 # Strong saliency contrst
        SALIENCY_ENHANCE_COEFF = 1.0
        #SALIENCY_ENHANCE_COEFF = 0.5 # Low saliency contrast, but sensible for weak saliency

        # Emphasize saliency
        saliency = (saliency ** SALIENCY_ENHANCE_COEFF)

        # Normalize to 0.0~1.0
        saliency = saliency / np.max(saliency)
    
        # Resize to original size
        saliency = cv2.resize(saliency, image.shape[1::-1])
        return saliency

