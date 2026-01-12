#!/usr/bin/env python3

import argparse
import csv
import os
import pickle
import sys
import time

# Add act_training_evaluation to path
sys.path.append('/home/aloha/act_training_evaluation')

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


def make_policy(policy_class, policy_config):
    # Import here to avoid triggering argument parsers at module level
    # Temporarily save and replace sys.argv with dummy args to satisfy DETR argparse

    original_argv = sys.argv.copy()
    sys.argv = [
        sys.argv[0],
        '--ckpt_dir', '/home/aloha/models-long',
        '--policy_class', 'ACT',
        '--task_name', 'aloha_mobile_elevator_press_arch',
        '--seed', '0',
        '--num_steps', '2500',
        '--temporal_agg'
    ]
    
    
    try:
        from act_plus_plus.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
        
        # Create policy while sys.argv is still modified
        if policy_class == 'ACT':
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        elif policy_class == 'Diffusion':
            policy = DiffusionPolicy(policy_config)
        else:
            raise NotImplementedError
        
        return policy
        
    except Exception as e:
        raise
    finally:
        sys.argv = original_argv


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


def save_images(ts, camera_names, save_dir, timestep):
    """
    Save camera images to disk (timestamped and in order).
    No display to avoid Qt/X11 issues when running headless.
    """
    for cam_name in camera_names:
        # Get raw image from observation (H, W, C) in BGR format
        img = ts.observation['images'][cam_name].copy()
        
        # Save individual camera image with timestep in filename
        save_path = os.path.join(save_dir, f'step_{timestep:04d}_{cam_name}.jpg')
        cv2.imwrite(save_path, img)  # Save in BGR (OpenCV default)


def run_policy(args):
    """
    Run a trained policy on the real robot with working base movement.
    Combines the policy inference from imitate_episodes.py with the 
    base control approach from replay_episodes.py.
    """
    
    # Import here to avoid triggering argument parsers at module level
    from act_plus_plus.utils import set_seed
    from act_plus_plus.detr.models.latent_model import Latent_Model_Transformer
    
    set_seed(1000)
    
    # Load configuration
    ckpt_dir = args['ckpt_dir']
    ckpt_name = args['ckpt_name']
    num_rollouts = args['num_rollouts']
    onscreen_render = args['onscreen_render']
    
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    config['temporal_agg'] = True 

    # Extract config parameters
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']
    vq = config['policy_config']['vq']
    
    # Load policy
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(f'Loaded policy: {loading_status}')
    policy.cuda()
    policy.eval()
    
    # Load VQ model if needed
    if vq:
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded latent model from: {latent_model_ckpt_path}')
    
    # Load dataset stats
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Pre/post processing functions
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # Initialize robot - using approach from replay_episodes.py
    node = create_interbotix_global_node('aloha')
    
    # Setup environment with base - similar to replay_episodes.py
    env = make_real_env(node, setup_robots=False, setup_base=IS_MOBILE)
    
    # CRITICAL: Enable motor torque for base (this is what replay_episodes does)
    if IS_MOBILE:
        env.base.base.set_motor_torque(True)
    robot_startup(node)
    
    env.setup_robots()
    
    # Policy inference parameters
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    
    # Base delay for real robot
    BASE_DELAY = 13 # USED TO BE 13 
    if not temporal_agg:
        query_frequency -= BASE_DELAY
    
    print(f"Starting {num_rollouts} rollout(s)")
    print(f"Query frequency: {query_frequency}, Base delay: {BASE_DELAY}")
    print(f"Temporal aggregation: {temporal_agg}")
    print(f"Policy will be queried every {query_frequency} timestep(s) = every {query_frequency * 0.02:.3f} seconds")
    
    # Run rollouts
    for rollout_id in range(num_rollouts):
        print(f"\n=== Rollout {rollout_id + 1}/{num_rollouts} ===")
        
        ts = env.reset()
        
        # Create directory for saving camera images
        image_save_dir = os.path.join(ckpt_dir, f'camera_images_rollout_{rollout_id}')
        os.makedirs(image_save_dir, exist_ok=True)
        print(f"Saving camera images to: {image_save_dir}")
        
        # Temporal aggregation buffer
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, 16]).cuda()
        
        # Setup CSV logging for base velocities
        csv_filename = os.path.join(ckpt_dir, f'run_policy_base_vel_rollout_{rollout_id}.csv')
        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['timestep', 'base_vel_x', 'base_vel_y', 'actual_dt'])
        
        # Run episode
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            prev_time = time0
            actual_dt_history = []
            
            for t in range(max_timesteps):
                time1 = time.time()
                actual_dt = time1 - prev_time if t > 0 else DT
                
                # Get observation
                obs = ts.observation
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                # Get image at query frequency
                if t % query_frequency == 0:
                    curr_image = get_image(
                        ts,
                        camera_names,
                        rand_crop_resize=(policy_class == 'Diffusion')
                    )
                
                # Save camera feed to disk (every timestep for debugging)
                save_images(ts, camera_names, image_save_dir, t)
                
                # Warm up network on first step
                if t == 0:
                    for _ in range(10):
                        policy(qpos, curr_image)
                    print('Network warm up done')
                    time1 = time.time()
                
                # Query policy
                if policy_class == "ACT":
                    if t % query_frequency == 0:
                        if vq:
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            all_actions = policy(qpos, curr_image)
                        
                        # Apply base delay compensation
                        all_actions = torch.cat(
                            [
                                all_actions[:, :-BASE_DELAY, :-2],
                                all_actions[:, BASE_DELAY:, -2:]
                            ],
                            dim=2
                        )
                    
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries - BASE_DELAY] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                
                elif policy_class == "Diffusion":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        all_actions = torch.cat(
                            [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]],
                            dim=2
                        )
                    raw_action = all_actions[:, t % query_frequency]
                
                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                
                # Post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                
                # CRITICAL: Clip actions to safe bounds from training data
                action_before_clip = action.copy()
                action = np.clip(action, stats['action_min'], stats['action_max'])
                
                target_qpos = action[:-2]
                base_action = action[-2:]
                
                # Log if base actions were clipped (indicates policy is outputting unsafe values)
                if not np.allclose(action_before_clip[-2:], base_action, atol=1e-6):
                    print(f"Step {t}: WARNING - base action clipped from {action_before_clip[-2:]} to {base_action}")
                
                print(f"Step {t}: base_action = {base_action}, actual_dt = {actual_dt:.3f}s")
                actual_dt_history.append(actual_dt)
                
                # Log to CSV
                csv_writer.writerow([
                    t,
                    float(base_action[0]),
                    float(base_action[1]),
                    actual_dt
                ])
                
                # Step environment
                if IS_MOBILE:
                    ts = env.step(target_qpos, base_action, get_base_vel=True)
                else:
                    ts = env.step(target_qpos, None, get_base_vel=False)
                
                # Timing control
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                time.sleep(sleep_time)
                prev_time = time.time()
                
                if duration >= DT:
                    print(f'Warning: step {t} took {duration:.3f}s (DT={DT:.3f}s)')
            
            avg_fps = max_timesteps / (time.time() - time0)
            avg_actual_dt = np.mean(actual_dt_history[1:]) if len(actual_dt_history) > 1 else DT
            print(f'Rollout {rollout_id} complete. Avg FPS: {avg_fps:.2f} (target: {FPS})')
            print(f'Avg actual DT: {avg_actual_dt:.3f}s (target: {DT:.3f}s), ratio: {avg_actual_dt/DT:.2f}x')
        
        # Close CSV file
        csv_file.close()
        print(f'Base velocity data saved to: {csv_filename}')
        
        # Open grippers at end
        move_grippers(
            [env.follower_bot_left, env.follower_bot_right],
            [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
            moving_time=0.5,
        )
    
    print("\nAll rollouts complete!")
    robot_shutdown(node)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Run a trained policy on the real robot with working base movement'
    )
    parser.add_argument(
        '--ckpt_dir',
        action='store',
        type=str,
        help='Checkpoint directory containing config.pkl and policy checkpoint',
        required=True,
    )
    parser.add_argument(
        '--ckpt_name',
        action='store',
        type=str,
        default='policy_best.ckpt',
        help='Name of the checkpoint file to load',
        required=False,
    )
    parser.add_argument(
        '--num_rollouts',
        action='store',
        type=int,
        default=1,
        help='Number of rollouts to execute',
        required=False,
    )
    parser.add_argument(
        '--onscreen_render',
        action='store_true',
        help='Render on screen (not implemented in this script)',
        required=False,
    )
    
    args = vars(parser.parse_args())
    
    try:
        run_policy(args)
    except KeyboardInterrupt:
        print("\n\n=== Interrupted by user (Ctrl+C) ===")
        print("Cleaning up...")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        raise
