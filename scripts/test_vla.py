#!/usr/bin/env python3

import argparse
import csv
import os
import pickle
import sys
import time

from aloha.constants import (
    FOLLOWER_GRIPPER_JOINT_OPEN,
    FPS,
    IS_MOBILE,
)
from aloha.real_env import make_real_env
from aloha.robot_utils import move_grippers
from einops import rearrange
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_startup,
    robot_shutdown,
)
from interbotix_common_modules.common_robot.exceptions import InterbotixException
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import cv2


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image


def run_vla():

    # Load in Llava model
    

    # Initialize robot - using approach from replay_episodes.py
    node = create_interbotix_global_node('aloha')
    env = make_real_env(node, setup_robots=False, setup_base=IS_MOBILE)   
    env.base.base.set_motor_torque(True)
    robot_startup(node)
    env.setup_robots()
    
    print(f"Starting")
    ts = env.reset()
      
    # Run episode

    while True: 
        cmd = input("Enter command: ")
        if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
            break
        
        # Get observation
        obs = ts.observation
        qpos_numpy = np.array(obs['qpos'])
        qpos = pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
               
        curr_image = get_image(
            ts,
            camera_names,
            rand_crop_resize=(policy_class == 'Diffusion')
        )

        # Now we have the: (1) command, (2) qpos, (3) image(s)

        
       
        all_actions = policy(qpos, curr_image)
        raw_action = all_actions[:, 0] # Take the first, for now
                
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = post_process(raw_action)
        
        target_qpos = action[:-2]
        base_action = action[-2:]

        # Step environment with SCALED velocity
        ts = env.step(target_qpos, base_action, get_base_vel=True)


    # Open grippers at end
    move_grippers(
        [env.follower_bot_left, env.follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
        moving_time=0.5,
    )
    
    robot_shutdown(node)


if __name__ == '__main__':
    run_vla()