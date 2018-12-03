# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# oculomotorデータセットの生成スクリプト

import numpy as np
import os
import cv2


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
