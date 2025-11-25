#!/usr/bin/env python3

"""
VLM Controller Base Class for Mobile ALOHA

This module provides a base class for Vision-Language-Model based control
of the Mobile ALOHA robot. It integrates camera inputs, natural language
processing, and closed-loop reasoning for autonomous task execution.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from aloha.constants import DT, IS_MOBILE
from aloha.robot_utils import (
    ImageRecorder,
    get_arm_joint_positions,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_on,
    setup_follower_bot,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
)
from geometry_msgs.msg import Twist
import rclpy


class VLMAction:
    """Represents an action output from a VLM model."""
    
    def __init__(
        self,
        base_action: Optional[np.ndarray] = None,
        left_arm_action: Optional[np.ndarray] = None,
        right_arm_action: Optional[np.ndarray] = None,
        left_gripper_action: Optional[float] = None,
        right_gripper_action: Optional[float] = None,
        reasoning: str = "",
        confidence: float = 1.0,
        done: bool = False,
    ):
        """
        Initialize a VLM action.
        
        Args:
            base_action: [linear_x, angular_z] velocities for mobile base
            left_arm_action: 6-DOF joint positions for left arm
            right_arm_action: 6-DOF joint positions for right arm
            left_gripper_action: Gripper position for left gripper
            right_gripper_action: Gripper position for right gripper
            reasoning: Text explanation of the action
            confidence: Confidence score [0, 1]
            done: Whether the task is complete
        """
        self.base_action = base_action if base_action is not None else np.zeros(2)
        self.left_arm_action = left_arm_action
        self.right_arm_action = right_arm_action
        self.left_gripper_action = left_gripper_action
        self.right_gripper_action = right_gripper_action
        self.reasoning = reasoning
        self.confidence = confidence
        self.done = done


class VLMController(ABC):
    """
    Abstract base class for VLM-based robot controllers.
    
    This class provides the infrastructure for:
    - Multi-camera visual input
    - Natural language command processing
    - Closed-loop control with reasoning
    - Robot state management
    """
    
    def __init__(
        self,
        node=None,
        enable_base: bool = True,
        enable_arms: bool = True,
        max_episode_length: int = 1000,
        control_frequency: float = 10.0,
    ):
        """
        Initialize the VLM controller.
        
        Args:
            node: ROS2 node (created if None)
            enable_base: Whether to enable mobile base control
            enable_arms: Whether to enable arm control
            max_episode_length: Maximum steps per episode
            control_frequency: Control loop frequency in Hz
        """
        self.enable_base = enable_base and IS_MOBILE
        self.enable_arms = enable_arms
        self.max_episode_length = max_episode_length
        self.control_dt = 1.0 / control_frequency
        
        # Initialize ROS2 node if not provided
        if node is None:
            if not rclpy.ok():
                rclpy.init()
            self.node = create_interbotix_global_node('vlm_controller')
        else:
            self.node = node
        
        # Initialize camera system
        self.image_recorder = ImageRecorder(is_mobile=IS_MOBILE, node=self.node)
        
        # Initialize arms if enabled
        self.follower_bot_left = None
        self.follower_bot_right = None
        if self.enable_arms:
            self._setup_arms()
        
        # Initialize mobile base publisher if enabled
        self.base_publisher = None
        if self.enable_base:
            self.base_publisher = self.node.create_publisher(
                Twist, 'mobile_base/cmd_vel', 10
            )
            time.sleep(0.5)
        
        # Episode tracking
        self.current_episode_step = 0
        self.episode_history = []
        
        print(f"VLM Controller initialized:")
        print(f"  - Base control: {self.enable_base}")
        print(f"  - Arm control: {self.enable_arms}")
        print(f"  - Cameras: {self.image_recorder.camera_names}")
    
    def _setup_arms(self):
        """Initialize the follower arms."""
        self.follower_bot_left = InterbotixManipulatorXS(
            robot_model='vx300s',
            robot_name='follower_left',
            node=self.node,
            iterative_update_fk=False,
        )
        self.follower_bot_right = InterbotixManipulatorXS(
            robot_model='vx300s',
            robot_name='follower_right',
            node=self.node,
            iterative_update_fk=False,
        )
        
        # Setup follower bots
        setup_follower_bot(self.follower_bot_left)
        setup_follower_bot(self.follower_bot_right)
        
        print("Follower arms initialized and ready")
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation from cameras and robot state.
        
        Returns:
            Dictionary containing:
                - images: Dict of camera images
                - robot_state: Current joint positions and gripper states
                - timestamp: Current time
        """
        observation = {
            'images': self.image_recorder.get_images(),
            'timestamp': time.time(),
        }
        
        if self.enable_arms:
            observation['robot_state'] = {
                'left_arm_qpos': get_arm_joint_positions(self.follower_bot_left),
                'right_arm_qpos': get_arm_joint_positions(self.follower_bot_right),
                'left_gripper_qpos': get_arm_gripper_positions(self.follower_bot_left),
                'right_gripper_qpos': get_arm_gripper_positions(self.follower_bot_right),
            }
        
        return observation
    
    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> VLMAction:
        """
        Predict the next action given observation and task.
        
        Args:
            observation: Current observation from get_observation()
            task_description: Natural language task description
            history: List of previous (observation, action) pairs
            
        Returns:
            VLMAction object with predicted actions and reasoning
        """
        pass
    
    @abstractmethod
    def reason_about_task(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> str:
        """
        Generate reasoning about the current state and what to do next.
        
        Args:
            observation: Current observation
            task_description: Natural language task description
            history: Execution history
            
        Returns:
            Text reasoning about the situation
        """
        pass
    
    def execute_action(self, action: VLMAction) -> bool:
        """
        Execute the given action on the robot.
        
        Args:
            action: VLMAction to execute
            
        Returns:
            True if execution successful, False otherwise
        """
        try:
            # Execute base action
            if self.enable_base and self.base_publisher is not None:
                twist = Twist()
                twist.linear.x = float(action.base_action[0])
                twist.angular.z = float(action.base_action[1])
                self.base_publisher.publish(twist)
            
            # Execute arm actions
            if self.enable_arms:
                if action.left_arm_action is not None:
                    self.follower_bot_left.arm.set_joint_positions(
                        action.left_arm_action, blocking=False
                    )
                
                if action.right_arm_action is not None:
                    self.follower_bot_right.arm.set_joint_positions(
                        action.right_arm_action, blocking=False
                    )
                
                # Execute gripper actions
                if action.left_gripper_action is not None:
                    from interbotix_xs_msgs.msg import JointSingleCommand
                    gripper_cmd = JointSingleCommand(name='gripper')
                    gripper_cmd.cmd = float(action.left_gripper_action)
                    self.follower_bot_left.gripper.core.pub_single.publish(gripper_cmd)
                
                if action.right_gripper_action is not None:
                    from interbotix_xs_msgs.msg import JointSingleCommand
                    gripper_cmd = JointSingleCommand(name='gripper')
                    gripper_cmd.cmd = float(action.right_gripper_action)
                    self.follower_bot_right.gripper.core.pub_single.publish(gripper_cmd)
            
            return True
            
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
    
    def run_episode(
        self,
        task_description: str,
        max_steps: Optional[int] = None,
        visualize: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a complete episode for the given task.
        
        Args:
            task_description: Natural language description of the task
            max_steps: Maximum number of steps (uses default if None)
            visualize: Whether to show visual feedback
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with episode results
        """
        if max_steps is None:
            max_steps = self.max_episode_length
        
        self.current_episode_step = 0
        self.episode_history = []
        
        print(f"\n{'='*60}")
        print(f"Starting new episode: {task_description}")
        print(f"{'='*60}\n")
        
        step = 0
        done = False
        
        while step < max_steps and not done and rclpy.ok():
            step_start_time = time.time()
            
            # Get observation
            observation = self.get_observation()
            
            # Predict action with reasoning
            action = self.predict_action(
                observation, task_description, self.episode_history
            )
            
            if verbose:
                print(f"\n--- Step {step + 1} ---")
                print(f"Reasoning: {action.reasoning}")
                print(f"Confidence: {action.confidence:.2f}")
                if self.enable_base:
                    print(f"Base: linear={action.base_action[0]:.3f}, "
                          f"angular={action.base_action[1]:.3f}")
            
            # Execute action
            success = self.execute_action(action)
            
            if not success:
                print("Failed to execute action, stopping episode")
                break
            
            # Save to history
            self.episode_history.append({
                'step': step,
                'observation': observation,
                'action': action,
                'timestamp': time.time(),
            })
            
            # Check if done
            done = action.done
            
            # Visualize if requested
            if visualize:
                self._visualize_step(observation, action, task_description)
            
            # Sleep to maintain control frequency
            elapsed = time.time() - step_start_time
            if elapsed < self.control_dt:
                time.sleep(self.control_dt - elapsed)
            
            step += 1
        
        # Stop the robot
        self.stop()
        
        print(f"\n{'='*60}")
        print(f"Episode completed: {step} steps")
        print(f"Task: {task_description}")
        print(f"Status: {'SUCCESS' if done else 'INCOMPLETE'}")
        print(f"{'='*60}\n")
        
        return {
            'task': task_description,
            'steps': step,
            'done': done,
            'history': self.episode_history,
        }
    
    def stop(self):
        """Stop all robot motion."""
        if self.enable_base and self.base_publisher is not None:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.base_publisher.publish(twist)
        
        # Arms naturally stop when no new commands are sent
        print("Robot stopped")
    
    def _visualize_step(
        self,
        observation: Dict[str, Any],
        action: VLMAction,
        task_description: str,
    ):
        """Visualize the current step (optional implementation)."""
        images = observation['images']
        
        # Create a visualization window with the main camera view
        if 'cam_high' in images and images['cam_high'] is not None:
            img = images['cam_high'].copy()
            
            # Add text overlay
            cv2.putText(
                img, f"Task: {task_description[:50]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.putText(
                img, f"Step: {self.current_episode_step}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.putText(
                img, f"Conf: {action.confidence:.2f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
            cv2.imshow('VLM Controller', img)
            cv2.waitKey(1)
    
    def shutdown(self):
        """Clean shutdown of the controller."""
        self.stop()
        cv2.destroyAllWindows()
        print("VLM Controller shutdown complete")
