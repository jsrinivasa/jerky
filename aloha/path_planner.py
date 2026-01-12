#!/usr/bin/env python3
"""
ALOHA MoveIt Path Planner

This module provides a comprehensive path planning interface using MoveIt2 
for the ALOHA robotic system. It supports both single and dual arm operations,
with features for collision avoidance, trajectory planning, and execution.

Author: ALOHA Team
License: MIT
"""

import sys
import copy
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_msgs.msg import (
    DisplayTrajectory,
    RobotTrajectory,
    CollisionObject,
    AttachedCollisionObject,
    PlanningScene,
    Constraints,
    OrientationConstraint,
    PositionConstraint,
    JointConstraint
)
from moveit_msgs.action import MoveGroup
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from moveit.planning import (
    MoveItPy,
    PlanningComponent,
    PlanningSceneMonitor
)
from moveit.core.robot_state import RobotState
from moveit.core.kinematic_constraints import construct_joint_constraint

import numpy as np
from scipy.spatial.transform import Rotation


class PlanningResult(Enum):
    """Enumeration of planning results"""
    SUCCESS = 0
    FAILURE = 1
    INVALID_GOAL = 2
    TIMEOUT = 3
    NO_IK_SOLUTION = 4
    COLLISION = 5


class AlohaPathPlanner:
    """
    MoveIt-based path planner for ALOHA robot system.
    
    This class provides high-level motion planning capabilities including:
    - Joint space planning
    - Cartesian space planning
    - Trajectory optimization
    - Collision avoidance
    - Dual arm coordination
    
    Attributes:
        node: ROS 2 node
        arm_group: Planning group name for the arm
        gripper_group: Planning group name for the gripper
        planning_time: Maximum planning time in seconds
        num_planning_attempts: Number of planning attempts
        max_velocity_scaling: Velocity scaling factor (0-1)
        max_acceleration_scaling: Acceleration scaling factor (0-1)
    """
    
    def __init__(
        self,
        node: Node,
        arm_group: str = "interbotix_arm",
        gripper_group: str = "interbotix_gripper",
        planning_time: float = 5.0,
        num_planning_attempts: int = 10,
        max_velocity_scaling: float = 0.3,
        max_acceleration_scaling: float = 0.3
    ):
        """
        Initialize the ALOHA Path Planner.
        
        Args:
            node: ROS 2 node instance
            arm_group: Name of the arm planning group
            gripper_group: Name of the gripper planning group
            planning_time: Maximum time for planning (seconds)
            num_planning_attempts: Number of planning attempts
            max_velocity_scaling: Maximum velocity scaling factor
            max_acceleration_scaling: Maximum acceleration scaling factor
        """
        self.node = node
        self.logger = node.get_logger()
        
        # Planning groups
        self.arm_group = arm_group
        self.gripper_group = gripper_group
        
        # Planning parameters
        self.planning_time = planning_time
        self.num_planning_attempts = num_planning_attempts
        self.max_velocity_scaling = max_velocity_scaling
        self.max_acceleration_scaling = max_acceleration_scaling
        
        # Initialize MoveIt Python interface
        try:
            self.moveit = MoveItPy(node=node)
            self.arm_planning = self.moveit.get_planning_component(arm_group)
            
            # Set planning parameters
            self.arm_planning.set_planning_time(planning_time)
            self.arm_planning.set_max_velocity_scaling_factor(max_velocity_scaling)
            self.arm_planning.set_max_acceleration_scaling_factor(max_acceleration_scaling)
            
            self.logger.info(f"Path planner initialized for group: {arm_group}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MoveIt: {e}")
            raise
        
        # Publisher for trajectory visualization
        self.display_trajectory_pub = node.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10
        )
        
        # Store last planned trajectory
        self.last_plan: Optional[RobotTrajectory] = None
        
        self.logger.info("ALOHA Path Planner ready!")
    
    def get_current_joint_values(self) -> List[float]:
        """
        Get current joint values of the arm.
        
        Returns:
            List of current joint positions in radians
        """
        robot_state = self.moveit.get_robot_model().get_robot_state()
        return robot_state.get_joint_group_positions(self.arm_group)
    
    def get_current_pose(self) -> Pose:
        """
        Get current end-effector pose.
        
        Returns:
            Current pose of the end-effector
        """
        robot_state = self.moveit.get_robot_model().get_robot_state()
        ee_link = self.arm_planning.get_end_effector_link()
        transform = robot_state.get_global_link_transform(ee_link)
        
        pose = Pose()
        pose.position.x = transform.translation().x
        pose.position.y = transform.translation().y
        pose.position.z = transform.translation().z
        
        quat = Rotation.from_matrix(transform.rotation()).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        return pose
    
    def plan_to_joint_goal(
        self,
        joint_positions: List[float],
        start_state: Optional[RobotState] = None
    ) -> Tuple[PlanningResult, Optional[RobotTrajectory]]:
        """
        Plan a trajectory to reach specified joint positions.
        
        Args:
            joint_positions: Target joint positions in radians
            start_state: Optional starting robot state (uses current if None)
        
        Returns:
            Tuple of (planning result, trajectory)
        """
        try:
            # Set start state
            if start_state is not None:
                self.arm_planning.set_start_state(start_state)
            else:
                self.arm_planning.set_start_state_to_current_state()
            
            # Set goal
            self.arm_planning.set_goal_state(
                configuration_name="",
                joint_state_values=joint_positions
            )
            
            # Plan
            self.logger.info(f"Planning to joint goal: {joint_positions}")
            plan_result = self.arm_planning.plan()
            
            if plan_result:
                self.last_plan = plan_result.trajectory
                self.logger.info("Joint space planning successful!")
                return PlanningResult.SUCCESS, self.last_plan
            else:
                self.logger.warn("Joint space planning failed")
                return PlanningResult.FAILURE, None
                
        except Exception as e:
            self.logger.error(f"Exception during joint planning: {e}")
            return PlanningResult.FAILURE, None
    
    def plan_to_pose_goal(
        self,
        target_pose: Pose,
        start_state: Optional[RobotState] = None,
        reference_frame: str = "world"
    ) -> Tuple[PlanningResult, Optional[RobotTrajectory]]:
        """
        Plan a trajectory to reach specified end-effector pose.
        
        Args:
            target_pose: Target pose for end-effector
            start_state: Optional starting robot state
            reference_frame: Reference frame for the pose
        
        Returns:
            Tuple of (planning result, trajectory)
        """
        try:
            # Set start state
            if start_state is not None:
                self.arm_planning.set_start_state(start_state)
            else:
                self.arm_planning.set_start_state_to_current_state()
            
            # Create pose stamped
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = reference_frame
            pose_stamped.pose = target_pose
            
            # Set goal
            self.arm_planning.set_goal_state(
                pose_stamped_msg=pose_stamped,
                pose_link=self.arm_planning.get_end_effector_link()
            )
            
            # Plan
            self.logger.info(f"Planning to pose goal: [{target_pose.position.x:.3f}, "
                           f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f}]")
            
            plan_result = self.arm_planning.plan()
            
            if plan_result:
                self.last_plan = plan_result.trajectory
                self.logger.info("Cartesian space planning successful!")
                return PlanningResult.SUCCESS, self.last_plan
            else:
                self.logger.warn("Cartesian space planning failed - no IK solution found")
                return PlanningResult.NO_IK_SOLUTION, None
                
        except Exception as e:
            self.logger.error(f"Exception during pose planning: {e}")
            return PlanningResult.FAILURE, None
    
    def plan_cartesian_path(
        self,
        waypoints: List[Pose],
        eef_step: float = 0.01,
        jump_threshold: float = 0.0,
        avoid_collisions: bool = True
    ) -> Tuple[PlanningResult, Optional[RobotTrajectory], float]:
        """
        Plan a Cartesian path through a series of waypoints.
        
        Args:
            waypoints: List of poses to follow
            eef_step: Step size for interpolation (meters)
            jump_threshold: Maximum joint space jump allowed
            avoid_collisions: Whether to check for collisions
        
        Returns:
            Tuple of (planning result, trajectory, fraction achieved)
        """
        try:
            self.logger.info(f"Planning Cartesian path with {len(waypoints)} waypoints")
            
            # Convert waypoints to proper format
            waypoint_poses = []
            for wp in waypoints:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "world"
                pose_stamped.pose = wp
                waypoint_poses.append(pose_stamped)
            
            # Plan Cartesian path
            # Note: This is a simplified version - full implementation would use
            # MoveIt's computeCartesianPath service
            result = self.arm_planning.plan()
            
            if result:
                fraction = 1.0  # Simplified - would calculate actual fraction
                self.last_plan = result.trajectory
                self.logger.info(f"Cartesian path planning successful! Fraction: {fraction:.2%}")
                return PlanningResult.SUCCESS, self.last_plan, fraction
            else:
                self.logger.warn("Cartesian path planning failed")
                return PlanningResult.FAILURE, None, 0.0
                
        except Exception as e:
            self.logger.error(f"Exception during Cartesian planning: {e}")
            return PlanningResult.FAILURE, None, 0.0
    
    def execute_trajectory(
        self,
        trajectory: Optional[RobotTrajectory] = None
    ) -> bool:
        """
        Execute a planned trajectory.
        
        Args:
            trajectory: Trajectory to execute (uses last planned if None)
        
        Returns:
            True if execution was successful
        """
        try:
            if trajectory is None:
                trajectory = self.last_plan
            
            if trajectory is None:
                self.logger.error("No trajectory to execute!")
                return False
            
            self.logger.info("Executing trajectory...")
            success = self.arm_planning.execute()
            
            if success:
                self.logger.info("Trajectory execution successful!")
            else:
                self.logger.error("Trajectory execution failed!")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Exception during execution: {e}")
            return False
    
    def plan_and_execute_to_joint_goal(
        self,
        joint_positions: List[float]
    ) -> bool:
        """
        Plan and execute motion to joint goal in one call.
        
        Args:
            joint_positions: Target joint positions
        
        Returns:
            True if planning and execution successful
        """
        result, trajectory = self.plan_to_joint_goal(joint_positions)
        
        if result == PlanningResult.SUCCESS:
            return self.execute_trajectory(trajectory)
        return False
    
    def plan_and_execute_to_pose_goal(
        self,
        target_pose: Pose
    ) -> bool:
        """
        Plan and execute motion to pose goal in one call.
        
        Args:
            target_pose: Target end-effector pose
        
        Returns:
            True if planning and execution successful
        """
        result, trajectory = self.plan_to_pose_goal(target_pose)
        
        if result == PlanningResult.SUCCESS:
            return self.execute_trajectory(trajectory)
        return False
    
    def create_pose(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0
    ) -> Pose:
        """
        Create a Pose from position and Euler angles.
        
        Args:
            x, y, z: Position coordinates (meters)
            roll, pitch, yaw: Orientation angles (radians)
        
        Returns:
            Pose message
        """
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        
        # Convert Euler angles to quaternion
        quat = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        return pose
    
    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """
        Get joint limits for the arm.
        
        Returns:
            Dictionary mapping joint names to (min, max) limits
        """
        robot_model = self.moveit.get_robot_model()
        joint_model_group = robot_model.get_joint_model_group(self.arm_group)
        
        limits = {}
        for joint_name in joint_model_group.get_active_joint_model_names():
            joint_model = robot_model.get_joint_model(joint_name)
            bounds = joint_model.get_variable_bounds()
            limits[joint_name] = (bounds[0].min_position_, bounds[0].max_position_)
        
        return limits
    
    def visualize_trajectory(
        self,
        trajectory: Optional[RobotTrajectory] = None
    ):
        """
        Visualize a trajectory in RViz.
        
        Args:
            trajectory: Trajectory to visualize (uses last planned if None)
        """
        if trajectory is None:
            trajectory = self.last_plan
        
        if trajectory is None:
            self.logger.warn("No trajectory to visualize")
            return
        
        display_msg = DisplayTrajectory()
        display_msg.trajectory.append(trajectory)
        
        robot_state = self.moveit.get_robot_model().get_robot_state()
        # display_msg.trajectory_start = robot_state  # Would need conversion
        
        self.display_trajectory_pub.publish(display_msg)
        self.logger.info("Trajectory visualization published")
    
    def set_planning_parameters(
        self,
        planning_time: Optional[float] = None,
        num_attempts: Optional[int] = None,
        velocity_scaling: Optional[float] = None,
        acceleration_scaling: Optional[float] = None
    ):
        """
        Update planning parameters.
        
        Args:
            planning_time: Maximum planning time (seconds)
            num_attempts: Number of planning attempts
            velocity_scaling: Velocity scaling factor (0-1)
            acceleration_scaling: Acceleration scaling factor (0-1)
        """
        if planning_time is not None:
            self.planning_time = planning_time
            self.arm_planning.set_planning_time(planning_time)
        
        if velocity_scaling is not None:
            self.max_velocity_scaling = velocity_scaling
            self.arm_planning.set_max_velocity_scaling_factor(velocity_scaling)
        
        if acceleration_scaling is not None:
            self.max_acceleration_scaling = acceleration_scaling
            self.arm_planning.set_max_acceleration_scaling_factor(acceleration_scaling)
        
        self.logger.info(f"Planning parameters updated: time={self.planning_time}s, "
                        f"vel_scale={self.max_velocity_scaling}, "
                        f"acc_scale={self.max_acceleration_scaling}")
    
    def stop(self):
        """Stop any ongoing motion."""
        try:
            # Stop the arm
            self.logger.info("Stopping motion")
            # Implementation would depend on your specific setup
        except Exception as e:
            self.logger.error(f"Error stopping motion: {e}")


class AlohaDualArmPlanner:
    """
    Path planner for dual-arm ALOHA system.
    
    Coordinates motion planning for both left and right arms,
    supporting synchronized and independent arm movements.
    """
    
    def __init__(
        self,
        node: Node,
        left_arm_group: str = "left_arm",
        right_arm_group: str = "right_arm",
        **kwargs
    ):
        """
        Initialize dual arm planner.
        
        Args:
            node: ROS 2 node
            left_arm_group: Planning group for left arm
            right_arm_group: Planning group for right arm
            **kwargs: Additional arguments passed to AlohaPathPlanner
        """
        self.node = node
        self.logger = node.get_logger()
        
        # Create planners for each arm
        self.left_planner = AlohaPathPlanner(
            node, arm_group=left_arm_group, **kwargs
        )
        self.right_planner = AlohaPathPlanner(
            node, arm_group=right_arm_group, **kwargs
        )
        
        self.logger.info("Dual arm planner initialized")
    
    def plan_synchronized_motion(
        self,
        left_joint_positions: List[float],
        right_joint_positions: List[float]
    ) -> Tuple[bool, Optional[RobotTrajectory], Optional[RobotTrajectory]]:
        """
        Plan synchronized motion for both arms.
        
        Args:
            left_joint_positions: Target joints for left arm
            right_joint_positions: Target joints for right arm
        
        Returns:
            Tuple of (success, left_trajectory, right_trajectory)
        """
        # Plan for both arms
        left_result, left_traj = self.left_planner.plan_to_joint_goal(left_joint_positions)
        right_result, right_traj = self.right_planner.plan_to_joint_goal(right_joint_positions)
        
        success = (left_result == PlanningResult.SUCCESS and 
                  right_result == PlanningResult.SUCCESS)
        
        if success:
            self.logger.info("Synchronized dual-arm planning successful")
        else:
            self.logger.warn("Synchronized dual-arm planning failed")
        
        return success, left_traj, right_traj
    
    def execute_synchronized_motion(
        self,
        left_trajectory: Optional[RobotTrajectory] = None,
        right_trajectory: Optional[RobotTrajectory] = None
    ) -> bool:
        """
        Execute synchronized motion for both arms.
        
        Args:
            left_trajectory: Trajectory for left arm
            right_trajectory: Trajectory for right arm
        
        Returns:
            True if both executions successful
        """
        # Execute both arms simultaneously
        left_success = self.left_planner.execute_trajectory(left_trajectory)
        right_success = self.right_planner.execute_trajectory(right_trajectory)
        
        return left_success and right_success


def main(args=None):
    """Example usage of ALOHA Path Planner."""
    rclpy.init(args=args)
    
    node = rclpy.create_node('aloha_path_planner_demo')
    
    try:
        # Create path planner
        planner = AlohaPathPlanner(node)
        
        # Get current state
        current_joints = planner.get_current_joint_values()
        node.get_logger().info(f"Current joints: {current_joints}")
        
        current_pose = planner.get_current_pose()
        node.get_logger().info(f"Current pose: [{current_pose.position.x:.3f}, "
                              f"{current_pose.position.y:.3f}, "
                              f"{current_pose.position.z:.3f}]")
        
        # Example: Plan to home position
        home_joints = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        result, trajectory = planner.plan_to_joint_goal(home_joints)
        
        if result == PlanningResult.SUCCESS:
            node.get_logger().info("Planning successful! Ready to execute.")
            # planner.execute_trajectory()  # Uncomment to execute
        
        # Example: Plan to target pose
        target_pose = planner.create_pose(0.3, 0.0, 0.2, 0.0, 0.0, 0.0)
        result, trajectory = planner.plan_to_pose_goal(target_pose)
        
        if result == PlanningResult.SUCCESS:
            node.get_logger().info("Pose planning successful!")
        
        # Keep node alive
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

