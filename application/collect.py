# -*- coding: utf-8 -*-
"""
 Training script for oculomotor tasks.
"""

import argparse
import cv2
import os
import time
import numpy as np

from agent import Agent
from functions import BG, FEF, LIP, PFC, Retina, SC, VC, HP, CB
from oculoenv import Environment
from oculoenv import PointToTargetContent, ChangeDetectionContent, OddOneOutContent, VisualSearchContent, \
    MultipleObjectTrackingContent, RandomDotMotionDiscriminationContent
from logger import Logger
from gen_dataset import generate



class Contents(object):
    POINT_TO_TARGET = 1
    CHANGE_DETECTION = 2
    ODD_ONE_OUT = 3
    VISUAL_SEARCH = 4
    MULTIPLE_OBJECT_TRACKING = 5
    RANDOM_DOT_MOTION_DISCRIMINATION = 6


def get_content(content_type):
    if content_type == Contents.POINT_TO_TARGET:
        content = PointToTargetContent()
    elif content_type == Contents.CHANGE_DETECTION:
        content = ChangeDetectionContent()
    elif content_type == Contents.ODD_ONE_OUT:
        content = OddOneOutContent()
    elif content_type == Contents.VISUAL_SEARCH:
        content = VisualSearchContent()
    elif content_type == Contents.MULTIPLE_OBJECT_TRACKING:
        content = MultipleObjectTrackingContent()
    else:
        content = RandomDotMotionDiscriminationContent()
    return content


def collect(content, step_size):
    retina = Retina()
    lip = LIP()
    vc = VC()
    pfc = PFC()
    fef = FEF()
    bg = BG()
    sc = SC()
    hp = HP()
    cb = CB()
    
    agent = Agent(
        retina=retina,
        lip=lip,
        vc=vc,
        pfc=pfc,
        fef=fef,
        bg=bg,
        sc=sc,
        hp=hp,
        cb=cb
    )
    
    env = Environment(content)

    pfc.load_model('data/pfc_task_detection.pth')
    
    obs = env.reset()
    
    reward = 0
    done = False
    
    episode_reward = 0
    episode_count = 0

    actions = []
    angles = []
    phases = []
    rewards = []
    targets = []
    lures = []

    if not os.path.exists("base_data"):
        os.mkdir("base_data")

    start_time = time.time()

    file_size_in_dir = 1000

    for i in range(step_size):
        if i % file_size_in_dir == 0:
            dir_path = "base_data/dir{}".format(i // file_size_in_dir)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        
        image, angle = obs['screen'], obs['angle']
        # Choose action by the agent's decision
        action = agent(image, angle, reward, done)
        # Foward environment one step
        obs, reward, done, info = env.step(action)

        phase = info['phase']

        if 'target' in info:
            target = info['target']
        else:
            target = (0.0, 0.0, 0.0)
        if 'lure' in info:
            lure = info['lure']
        else:
            lure = (0.0, 0.0, 0.0)

        actions.append(action)
        angles.append(angle)
        phases.append(phase)
        rewards.append(reward)
        targets.append(target)
        lures.append(lure)

        file_name = "{}/image{}.png".format(dir_path, i)
        image = cv2.cvtColor(obs["screen"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_name, image)
        
        episode_reward += reward

        if i % 1000 == 0:
            print("step{}".format(i))
            elapsed_time = time.time() - start_time
            print("fps={}".format(i / elapsed_time))

        if done:
            obs = env.reset()
            print("episode reward={}".format(episode_reward))
            
            # Store log for tensorboard graph
            episode_count += 1
            
            episode_reward = 0

    np.savez_compressed("base_data/infos",
                        actions=actions,
                        angles=angles,
                        rewards=rewards,
                        phases=phases,
                        targets=targets,
                        lures=lures)
    
    print("collecting finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content",
                        help="1: Point To Target"
                        + " 2: Change Detection"
                        + " 3: Odd One Out"
                        + " 4: Visual Search"
                        + " 5: Multiple Object Tracking"
                        + " 6: Random Dot Motion Descrimination",
                        type=int,
                        default=1)
    
    # Small dataset version
    parser.add_argument("--step_size", help="Training step size", type=int, default=20*10000+1)
    
    args = parser.parse_args()
    
    content_type = args.content
    step_size = args.step_size

    # Create task content
    content = get_content(content_type)
    
    print("start collecting content: {} step_size={}".format(content_type, step_size))

    # Collect original images
    collect(content, step_size)

    # generate dataset
    generate("base_data", step_size-1)
    


if __name__ == '__main__':
    main()
