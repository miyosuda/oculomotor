# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np

import brica
from .utils import load_image
from .pfc import Task, InternalPhase, TARGETS

from oculoenv.environment import CAMERA_VERTICAL_ANGLE_MAX
from oculoenv.environment import CAMERA_HORIZONTAL_ANGLE_MAX

"""
This is an example implemention of FEF (Frontal Eye Field) module.
You can change this as you like.
"""

GRID_DIVISION = 8
GRID_WIDTH = 128 // GRID_DIVISION

TEMPLATE_COEFF = 0.3
DIRECTION_COEFF = 0.1
CHANGE_COEFF = 0.3
SEARCH_COEFF = 0.3

ANGLE_TO_PANEL_RATE = 0.55 # TODO
DIRECTION_TO_ANGLE_RATE = 0.4 # TODO

class ActionAccumulator(object):
    """
    Sample implementation of an accumulator.
    """
    def __init__(self, ex, ey, decay_rate=0.9):
        """
        Arguments:
          ex: Float eye move dir x
          ey: Float eye move dir Y
        """
        # Accumulated likehilood
        self.likelihood = 0.0
        # Eye movment
        self.ex = ex
        self.ey = ey
        # Decay rate of likehilood
        self.decay_rate = decay_rate

        # Connected accumulators
        self.accumulators = []
        
    def accumulate(self, value):
        self.likelihood += value

    def expose(self):
        # Sample implementation of accumulator connection.
        # Send accmulated likelihood to another accumulator.
        for accumulator in self.accumulators:
            weight = 0.01
            accumulator.accumulate(self.likelihood * weight)

    def post_process(self):
        # Clip likelihood
        self.likelihood = np.clip(self.likelihood, 0.0, 1.0)
        
        # Decay likelihood
        self.likelihood *= self.decay_rate

    def reset(self):
        self.likelihood = 0.0

    def connect_to(self, accumulator):
        self.accumulators.append(accumulator)

    @property
    def output(self):
        return [self.likelihood, self.ex, self.ey]


class TemplateAccumulator(ActionAccumulator):
    def __init__(self, pixel_x, pixel_y, ex, ey):
        super(TemplateAccumulator, self).__init__(ex, ey)
        # Pixel x,y pos at left top corner of the region.
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        
    def process(self, match_map, threshold):
        region_match = match_map[self.pixel_y:self.pixel_y+GRID_WIDTH,
                                 self.pixel_x:self.pixel_x+GRID_WIDTH]
        max_match = np.max(region_match)
        if max_match < threshold:
            max_match = 0.0
        self.accumulate(max_match * TEMPLATE_COEFF)
        self.expose()

class DirectionAccumulator(ActionAccumulator):
    def __init__(self, pixel_x, pixel_y, ex, ey):
        super(DirectionAccumulator, self).__init__(ex, ey)
        # Pixel x,y pos at left top corner of the region.
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def process(self, direction_map):
        region_direction = direction_map[self.pixel_y:self.pixel_y+GRID_WIDTH,
                                         self.pixel_x:self.pixel_x+GRID_WIDTH]
        max_direction = np.max(region_direction)
        self.accumulate(max_direction * DIRECTION_COEFF)
        self.expose()

class ChangeAccumulator(ActionAccumulator):
    def __init__(self, pixel_x, pixel_y, ex, ey):
        super(ChangeAccumulator, self).__init__(ex, ey, decay_rate=0.75)
        # Pixel x,y pos at left top corner of the region.
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def process(self, change_map):
        region_change = change_map[self.pixel_y:self.pixel_y+GRID_WIDTH,
                                   self.pixel_x:self.pixel_x+GRID_WIDTH]
        average_change = np.mean(region_change)
        self.accumulate(average_change * CHANGE_COEFF)
        self.expose()

class SearchAccumulator(ActionAccumulator):
    def __init__(self, pixel_x, pixel_y, ex, ey):
        super(SearchAccumulator, self).__init__(ex, ey, decay_rate=0.75)
        # Pixel x,y pos at left top corner of the region.
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def process(self, search_map, max_value):
        region_search = search_map[self.pixel_y:self.pixel_y+GRID_WIDTH,
                                   self.pixel_x:self.pixel_x+GRID_WIDTH]
        max_search = np.max(region_search)
        if max_search < max_value:
            max_search = 0
        self.accumulate(max_search * SEARCH_COEFF)
        self.expose()

class FEF(object):
    def __init__(self):
        self.timing = brica.Timing(4, 1, 0)
        
        self.template_accumulators = []
        self.direction_accumulators = []
        self.change_accumulators = []
        self.search_accumulators = []

        for ix in range(GRID_DIVISION):
            pixel_x = GRID_WIDTH * ix
            cx = 2.0 / GRID_DIVISION * (ix + 0.5) - 1.0
            
            for iy in range(GRID_DIVISION):
                pixel_y = GRID_WIDTH * iy
                cy = 2.0 / GRID_DIVISION * (iy + 0.5) - 1.0
                
                ex = -cx
                ey = -cy
                
                template_accumulator = TemplateAccumulator(pixel_x, pixel_y, ex, ey)
                self.template_accumulators.append(template_accumulator)
                direction_accumulator = DirectionAccumulator(pixel_x, pixel_y, ex, ey)
                self.direction_accumulators.append(direction_accumulator)
                change_accumulator = ChangeAccumulator(pixel_x, pixel_y, ex, ey)
                self.change_accumulators.append(change_accumulator)
                search_accumulator = SearchAccumulator(pixel_x, pixel_y, ex, ey)
                self.search_accumulators.append(search_accumulator)

                
    def __call__(self, inputs):
        if 'from_lip' not in inputs:
            raise Exception('FEF did not recieve from LIP')
        if 'from_vc' not in inputs:
            raise Exception('FEF did not recieve from VC')
        if 'from_pfc' not in inputs:
            raise Exception('FEF did not recieve from PFC')
        if 'from_bg' not in inputs:
            raise Exception('FEF did not recieve from BG')

        task, phase, internal_phase, target, angle, search_map, change_map = inputs['from_pfc']

        optical_flow, large_e_match_map, small_e_match_map, magenta_t_match_map, green_ball_match_map, blue_ball_match_map, black_ball_match_map = inputs['from_lip']
        retina_image = inputs['from_vc']

        # Calculate mean value of optical flow
        direction_x, direction_y = 0, 0
        if optical_flow is not None:
            direction_x, direction_y = self._get_center_optical_flow_mean(optical_flow)
        direction_angle = self._direction_to_angle(direction_x, direction_y, DIRECTION_TO_ANGLE_RATE)
        direction_map = self._get_direction_map(0.0 - direction_angle[0], 0.0 - direction_angle[1])

        # Choose template match map
        match_map = None
        max_match = 0.0
        if task == Task.POINT_TO_TARGET:
            max_large = np.max(large_e_match_map)
            max_small = np.max(small_e_match_map)
            match_map = large_e_match_map if max_large > max_small else small_e_match_map
            max_match = np.max([max_large, max_small])
        elif task == Task.VISUAL_SEARCH:
            match_map = magenta_t_match_map
            max_match = np.max(magenta_t_match_map)
        elif task == Task.MULTIPLE_OBJECT_TRACKING:
            if internal_phase == InternalPhase.LEARNING:
                match_map = green_ball_match_map
                max_match = np.max(green_ball_match_map)
            elif internal_phase == InternalPhase.INTERVAL:
                match_map = black_ball_match_map
                max_match = np.max(black_ball_match_map)
            elif internal_phase == InternalPhase.EVALUATION:
                match_map = blue_ball_match_map
                max_match = np.max(blue_ball_match_map)

        for template_accumulator in self.template_accumulators:
            if match_map is not None:
                template_accumulator.process(match_map, max_match)
        for direction_accumulator in self.direction_accumulators:
            direction_accumulator.process(direction_map)
        for change_accumulator in self.change_accumulators:
            change_accumulator.process(change_map)
        max_search = np.max(search_map)
        num_max_search = len(search_map[np.where(search_map == max_search)])
        if num_max_search > 100: # useless if too much max points
            max_search = 1.0
        for search_accumulator in self.search_accumulators:
            search_accumulator.process(search_map, max_search)

        for template_accumulator in self.template_accumulators:
            template_accumulator.post_process()
        for direction_accumulator in self.direction_accumulators:
            direction_accumulator.post_process()
        for change_accumulator in self.change_accumulators:
            change_accumulator.post_process()
        for search_accumulator in self.search_accumulators:
            search_accumulator.post_process()

        output = self._collect_output()

        target_angle = TARGETS[target]
        cb_output = np.array([target_angle[0] - angle[0], target_angle[1] - angle[1]])

        return dict(to_pfc=None,
                    to_bg=output,
                    to_sc=output,
                    to_cb=cb_output)

    def _collect_output(self):
        output = []
        for template_accumulator in self.template_accumulators:
            output.append(template_accumulator.output)
        for direction_accumulator in self.direction_accumulators:
            output.append(direction_accumulator.output)
        for change_accumulator in self.change_accumulators:
            output.append(change_accumulator.output)
        for search_accumulator in self.search_accumulators:
            output.append(search_accumulator.output)
        return np.array(output, dtype=np.float32)

    def _get_center_optical_flow_mean(self, optical_flow):
        h, w = optical_flow.shape[:2]
        mean_x = np.mean(optical_flow[w//4:w*3//4, h//4:h*3//4, 0])
        mean_y = np.mean(optical_flow[w//4:w*3//4, h//4:h*3//4, 1])
        return mean_x, mean_y

    def _direction_to_angle(self, direction_x, direction_y, rate):
        theta = np.arctan2(direction_y, direction_x) # [-pi, pi]
        theta = theta + 0.125 * np.pi # theta + pi / 8
        theta = (theta + 2 * np.pi) % (2 * np.pi) # [0, 2 * pi]
        discrete_theta = (2 * np.pi * 0.125) * np.digitize(theta, bins=np.linspace(0, 2 * np.pi, 8 + 1)[1:-1])
        #print(theta, discrete_theta)
        angle_h = rate * np.cos(discrete_theta)
        angle_v = rate * np.sin(discrete_theta)
        return angle_h, angle_v

    def _get_direction_map(self, angle_h, angle_v):
        direction_map = np.zeros((128, 128), np.float32)
        w, h = direction_map.shape
        x, y = self._angle_to_panel_coordinate(angle_h, angle_v, w, h)
        #print((angle_h, angle_v), self._panel_coordinate_to_angle(x, y, w, h))
        direction_map[y, x] = 1.0 # NOT direction_map[x, y]
        return direction_map

    def _angle_to_panel_coordinate(self, angle_h, angle_v, w, h):
        # x = (1.0 + angle_h / unit) * 0.5 * w
        angle_h_unit = CAMERA_HORIZONTAL_ANGLE_MAX * ANGLE_TO_PANEL_RATE
        angle_v_unit = CAMERA_VERTICAL_ANGLE_MAX * ANGLE_TO_PANEL_RATE
        x = (1.0 - angle_h / angle_h_unit) * 0.5 * w
        y = (1.0 - angle_v / angle_v_unit) * 0.5 * h
        x = np.clip(x, 0, w - 1).astype(np.uint8)
        y = np.clip(y, 0, h - 1).astype(np.uint8)
        return x, y

    def _panel_coordinate_to_angle(self, x, y, w, h):
        # angle_h = 2 * unit * (x / w - 0.5)
        angle_h_unit = CAMERA_HORIZONTAL_ANGLE_MAX * ANGLE_TO_PANEL_RATE
        angle_v_unit = CAMERA_VERTICAL_ANGLE_MAX * ANGLE_TO_PANEL_RATE
        angle_h = 2.0 * angle_h_unit * (0.5 - x / w)
        angle_v = 2.0 * angle_v_unit * (0.5 - y / h)
        return angle_h, angle_v
