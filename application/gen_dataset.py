# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# oculomotorデータセットの生成スクリプト

import numpy as np
import os
import cv2


class OpticalFlowManager(object):
    def __init__(self, height=32, width=32):
        """ Calculating optical flow.
        Input image can be retina image or saliency map.
        """
        self.last_gray_image = None
        self.hist_32 = np.zeros((height, width), np.float32)
        
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
        
    def process(self, gray_image):
        if gray_image is None:
            return

        if self.last_gray_image is not None:
            if self.flow is not None:
                self.flow = self.inst.calc(self.last_gray_image,
                                           gray_image,
                                           self._warp_flow(self.flow, self.flow))
            else:
                self.flow = self.inst.calc(self.last_gray_image,
                                           gray_image,
                                           None)
            # (height, width, 2)
        self.last_gray_image = gray_image
        return self.flow


def generate(base_data_dir, frame_size):
    file_name = "oculomotor"
    seq_length = 20

    infos_path = os.path.join(base_data_dir, "infos.npz")
    data = np.load(infos_path)
    
    dat_actions = data["actions"].astype(np.float32)[:-1] # (100000, 2)
    dat_angles  = data["angles"].astype(np.float32)[:-1]  # (100000, 2)
    dat_rewards = data["rewards"].astype(np.int32)[:-1]   # (100000,)
    dat_phases  = data["phases"].astype(np.int8)[:-1]     # (100000,)
    dat_targets = data["targets"].astype(np.float32)[:-1] # (100000, 3)
    dat_lures   = data["lures"].astype(np.float32)[:-1]   # (100000, 3)
    
    dat_actions = dat_actions.reshape((-1, seq_length, 2)) # (5000, 20, 2)
    dat_angles  = dat_angles.reshape((-1, seq_length, 2))  # (5000, 20, 2)
    dat_rewards = dat_rewards.reshape((-1, seq_length))    # (5000, 20)
    dat_phases  = dat_phases.reshape((-1, seq_length))     # (5000, 20)
    dat_targets = dat_targets.reshape((-1, seq_length, 3)) # (5000, 20, 3)
    dat_lures   = dat_lures.reshape((-1, seq_length, 3))   # (5000, 20, 3)
    
    dat_images = np.empty((frame_size, 64, 64), np.uint8)

    dir_path = os.path.join(base_data_dir, "dir0")
    file_size_in_dir = 1000
    
    for i in range(frame_size):
        if i % file_size_in_dir == 0:
            dir_path = os.path.join(base_data_dir,
                                    "dir{}".format(i // file_size_in_dir))
        file_path = "{}/image{}.png".format(dir_path, i)
        # (128)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64,64))
        # (64, 64) uint8
        dat_images[i] = image
        
        if i % 100 == 0:
            print("processes:{}".format(i))

    # (100000, 64, 64) uint8
    dat_images = dat_images.reshape((-1, seq_length, 64, 64))
    # (5000, 20, 64, 64) uint8

    # .npzを省いたパス
    file_path = os.path.join(".", file_name)

    # 圧縮して保存
    np.savez_compressed(file_path,
                        images=dat_images,
                        actions=dat_actions,
                        angles=dat_angles,
                        rewards=dat_rewards,
                        phases=dat_phases,
                        targets=dat_targets,
                        lures=dat_lures)


def generate_opt_flow(base_data_dir, frame_size):
    file_name = "oculomotor_opt_flow"
    seq_length = 20

    # 何故か32x32にするとcv2が落ちてしまう.
    opt_flow_width = 48

    dat_opt_flow = np.empty((frame_size, opt_flow_width, opt_flow_width, 2), np.float32)

    opt_flow_manager = OpticalFlowManager(opt_flow_width, opt_flow_width)

    dir_path = os.path.join(base_data_dir, "dir0")
    file_size_in_dir = 1000

    for i in range(frame_size):
        if i % file_size_in_dir == 0:
            dir_path = os.path.join(base_data_dir,
                                    "dir{}".format(i // file_size_in_dir))
        file_path = "{}/image{}.png".format(dir_path, i)
        # (128)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (opt_flow_width,opt_flow_width))
        # (32, 32) uint8
        dat_opt_flow[i] = opt_flow_manager.process(image)
        
        if i % 100 == 0:
            print("processes:{}".format(i))

    # 1枚目が空なので2枚目と同じものにしておく
    dat_opt_flow[0] = dat_opt_flow[1]

    # (100000, 32, 32, 2) float32
    dat_opt_flow = dat_opt_flow.reshape((-1, seq_length, opt_flow_width, opt_flow_width, 2))
    # (5000, 20, 32, 32, 2) float32
        
    # .npzを省いたパス
    file_path = os.path.join(".", file_name)

    # 圧縮して保存
    np.savez_compressed(file_path,
                        opt_flow=dat_opt_flow)
