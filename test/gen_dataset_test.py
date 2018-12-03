# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
import imageio

DEBUG_SAVE_IMAGE = False


class GenDatasetTest(unittest.TestCase):
    def test_generate(self):
        data_path = "./oculomotor.npz"
        data_all = np.load(data_path)

        seq_size = 10000
        
        data_images  = data_all["images"]  # (seq_size, 20, 64, 64) uint8
        data_actions = data_all["actions"] # (seq_size, 20, 2)      float32

        data_actions = data_all["actions"] # (seq_size, 20, 2)      float32
        data_angles  = data_all["angles"]  # (seq_size, 20, 2)      float32
        data_rewards = data_all["rewards"] # (seq_size, 20)         int32
        data_phases  = data_all["phases"]  # (seq_size, 20)         int8
        data_targets = data_all["targets"] # (seq_size, 20, 3)      float32
        data_lures   = data_all["lures"]   # (seq_size, 20, 3)      float32

        self.assertEqual(data_images.shape,    (seq_size, 20, 64, 64))
        self.assertEqual(data_actions.shape,   (seq_size, 20, 2))
        self.assertEqual(data_angles.shape,    (seq_size, 20, 2))
        self.assertEqual(data_rewards.shape,   (seq_size, 20))
        self.assertEqual(data_phases.shape,    (seq_size, 20))
        self.assertEqual(data_targets.shape,   (seq_size, 20, 3))
        self.assertEqual(data_lures.shape,     (seq_size, 20, 3))
        
        self.assertEqual(data_images.dtype,  np.uint8)
        self.assertEqual(data_actions.dtype, np.float32)
        self.assertEqual(data_angles.dtype,  np.float32)
        self.assertEqual(data_rewards.dtype, np.int32)
        self.assertEqual(data_phases.dtype,  np.int8)
        self.assertEqual(data_targets.dtype, np.float32)
        self.assertEqual(data_lures.dtype,   np.float32)

        if DEBUG_SAVE_IMAGE:
            seq_index = 0
            for i in range(20):
                img = data_images[seq_index,i]
                imageio.imwrite("o_out_{0:02}_{1:02}.png".format(seq_index, i), img)


if __name__ == '__main__':
    unittest.main()
