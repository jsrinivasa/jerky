#!/usr/bin/env python3
"""
Simple ALOHA Path Planner using MoveIt Commander

A simplified interface using the standard moveit_commander for easier compatibility.
This version is recommended for getting started quickly.

Author: ALOHA Team
License: MIT
"""

from typing import List, Tuple, Optional
import numpy as np

import rclpy
from rclpy.node import Node

try:
    import moveit_commander
    from moveit_commander.conversions import pose_to_list
except ImportError:
    print("Error: moveit_commander not found. Install with:")
    print("  sudo apt install ros-${ROS_DISTRO}-moveit-commander")
    raise

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from scipy.spatial.transform import Rotation


class SimplePathPlanner:
    """
    Simplified MoveIt-based path planner using moveit_commander.
    
    This class provides an easy-to-use interface for motion planning
    with the ALOHA system using the standard MoveIt Commander API.
    
    Example:
        planner = SimplePathPlanner(node, "interbotix_arm")
        success = planner.move_to_joint_goal([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    """
    
    def __init__(
        self,
        node: Node,
        group_name: str = "interbotix_arm",
        velocity_scaling: float = 0.3,
        acceleration_scaling: float = 0.3,
        planning_time: float = 5.0
    ):
        """
        Initialize the simple path planner.
        
        Args:
            node: ROS 2 node
            group_name: MoveIt planning group name
            velocity_scaling: Max velocity scaling (0-1)
            acceleration_scaling: Max acceleration scaling (0-1)
            planning_time: Maximum planning time in seconds
        """
        self.node = node
        self.logger = node.get_logger()
        
        # Initialize moveit_commander
        moveit_commander.roscpp_initialize([])
        
        # Create robot interface
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Create move group
        self.group_name = group_name
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        
        # Set planning parameters
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_planning_time(planning_time)
        
        # Get basic information
        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        
        self.logger.info(f"Simple Path Planner initialized")
        self.logger.info(f"  Planning group: {group_name}")
        self.logger.info(f"  Planning frame: {self.planning_frame}")
        self.logger.info(f"  End effector: {self.eef_link}")
    
    def get_current_joint_values(self) -> List[float]:
        """Get current joint values."""
        return self.move_group.get_current_joint_values()
    
    def get_current_pose(self) -> Pose:
        """Get current end-effector pose."""
        return self.move_group.get_current_pose().pose
    
    def move_to_joint_goal(
        self,
        joint_values: List[float],
        wait: bool = True
    ) -> bool:
        """
        Move to specified joint configuration.
        
        Args:
            joint_values: Target joint angles in radians
            wait: Whether to wait for motion to complete
        
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Moving to joint goal: {[f'{j:.3f}' for j in joint_values]}")
            self.move_group.go(joint_values, wait=wait)
            self.move_group.stop()
            
            # Verify we reached the goal
            current = self.get_current_joint_values()
            success = self._all_close(joint_values, current, tolerance=0.01)
            
            if success:
                self.logger.info("✓ Reached joint goal")
            else:
                self.logger.warn("⚠ Did not fully reach joint goal")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error moving to joint goal: {e}")
            return False
    
    def move_to_pose_goal(
        self,
        target_pose: Pose,
        wait: bool = True
    ) -> bool:
        """
        Move to specified end-effector pose.
        
        Args:
            target_pose: Target pose for end-effector
            wait: Whether to wait for motion to complete
        
        Returns:
            True if successful
        """
        try:
            self.logger.info(
                f"Moving to pose: [{target_pose.position.x:.3f}, "
                f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f}]"
            )
            
            self.move_group.set_pose_target(target_pose)
            success = self.move_group.go(wait=wait)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            if success:
                self.logger.info("✓ Reached pose goal")
            else:
                self.logger.warn("⚠ Did not reach pose goal")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error moving to pose goal: {e}")
            return False
    
    def plan_to_joint_goal(
        self,
        joint_values: List[float]
    ) -> Tuple[bool, Optional[object]]:
        """
        Plan (but don't execute) motion to joint goal.
        
        Args:
            joint_values: Target joint angles
        
        Returns:
            (success, plan) tuple
        """
        try:
            self.move_group.set_joint_value_target(joint_values)
            plan = self.move_group.plan()
            
            # Handle different return formats from different MoveIt versions
            if isinstance(plan, tuple):
                success, trajectory, planning_time, error_code = plan
            else:
                success = bool(plan)
                trajectory = plan
            
            return success, trajectory if success else None
            
        except Exception as e:
            self.logger.error(f"Error planning to joint goal: {e}")
            return False, None
    
    def plan_to_pose_goal(
        self,
        target_pose: Pose
    ) -> Tuple[bool, Optional[object]]:
        """
        Plan (but don't execute) motion to pose goal.
        
        Args:
            target_pose: Target end-effector pose
        
        Returns:
            (success, plan) tuple
        """
        try:
            self.move_group.set_pose_target(target_pose)
            plan = self.move_group.plan()
            self.move_group.clear_pose_targets()
            
            # Handle different return formats
            if isinstance(plan, tuple):
                success, trajectory, planning_time, error_code = plan
            else:
                success = bool(plan)
                trajectory = plan
            
            return success, trajectory if success else None
            
        except Exception as e:
            self.logger.error(f"Error planning to pose goal: {e}")
            return False, None
    
    def execute_plan(self, plan) -> bool:
        """
        Execute a previously computed plan.
        
        Args:
            plan: Previously computed plan
        
        Returns:
            True if successful
        """
        try:
            success = self.move_group.execute(plan, wait=True)
            if success:
                self.logger.info("✓ Plan executed successfully")
            else:
                self.logger.error("✗ Plan execution failed")
            return success
        except Exception as e:
            self.logger.error(f"Error executing plan: {e}")
            return False
    
    def plan_cartesian_path(
        self,
        waypoints: List[Pose],
        eef_step: float = 0.01,
        jump_threshold: float = 0.0
    ) -> Tuple[object, float]:
        """
        Plan a Cartesian path through waypoints.
        
        Args:
            waypoints: List of poses to follow
            eef_step: Step size for interpolation
            jump_threshold: Max joint space jump
        
        Returns:
            (plan, fraction) tuple where fraction is percentage of path achieved
        """
        try:
            plan, fraction = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step,
                jump_threshold
            )
            
            self.logger.info(f"Cartesian path computed: {fraction*100:.1f}% achieved")
            return plan, fraction
            
        except Exception as e:
            self.logger.error(f"Error planning Cartesian path: {e}")
            return None, 0.0
    
    def create_pose(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0
    ) -> Pose:
        """
        Create a Pose from position and Euler angles.
        
        Args:
            x, y, z: Position in meters
            roll, pitch, yaw: Orientation in radians
        
        Returns:
            Pose message
        """
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        
        # Convert Euler to quaternion
        quat = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        return pose
    
    def set_velocity_scaling(self, factor: float):
        """Set velocity scaling factor (0-1)."""
        self.move_group.set_max_velocity_scaling_factor(factor)
        self.logger.info(f"Velocity scaling set to {factor}")
    
    def set_acceleration_scaling(self, factor: float):
        """Set acceleration scaling factor (0-1)."""
        self.move_group.set_max_acceleration_scaling_factor(factor)
        self.logger.info(f"Acceleration scaling set to {factor}")
    
    def set_planning_time(self, seconds: float):
        """Set maximum planning time."""
        self.move_group.set_planning_time(seconds)
        self.logger.info(f"Planning time set to {seconds}s")
    
    def stop(self):
        """Stop any ongoing motion."""
        self.move_group.stop()
        self.logger.info("Motion stopped")
    
    def _all_close(
        self,
        goal: List[float],
        actual: List[float],
        tolerance: float = 0.01
    ) -> bool:
        """Check if two lists of values are close within tolerance."""
        if len(goal) != len(actual):
            return False
        return all(abs(g - a) <= tolerance for g, a in zip(goal, actual))


class SimpleDualArmPlanner:
    """
    Simple dual-arm planner.
    
    Manages two SimplePathPlanner instances for coordinated dual-arm control.
    """
    
    def __init__(
        self,
        node: Node,
        left_group: str = "left_arm",
        right_group: str = "right_arm",
        **kwargs
    ):
        """
        Initialize dual arm planner.
        
        Args:
            node: ROS 2 node
            left_group: Left arm planning group
            right_group: Right arm planning group
            **kwargs: Additional args for SimplePathPlanner
        """
        self.node = node
        self.logger = node.get_logger()
        
        self.left_planner = SimplePathPlanner(node, left_group, **kwargs)
        self.right_planner = SimplePathPlanner(node, right_group, **kwargs)
        
        self.logger.info("Dual arm planner initialized")
    
    def move_both_to_joint_goals(
        self,
        left_joints: List[float],
        right_joints: List[float]
    ) -> bool:
        """
        Move both arms to specified joint configurations.
        
        Note: This executes sequentially, not simultaneously.
        For truly synchronized motion, use the more advanced AlohaPathPlanner.
        
        Args:
            left_joints: Left arm target joints
            right_joints: Right arm target joints
        
        Returns:
            True if both successful
        """
        left_success = self.left_planner.move_to_joint_goal(left_joints)
        right_success = self.right_planner.move_to_joint_goal(right_joints)
        
        return left_success and right_success


def main():
    """Example usage of Simple Path Planner."""
    rclpy.init()
    node = rclpy.create_node('simple_path_planner_demo')
    
    try:
        # Create planner
        planner = SimplePathPlanner(node, "interbotix_arm")
        
        # Get current state
        current_joints = planner.get_current_joint_values()
        node.get_logger().info(f"Current joints: {[f'{j:.3f}' for j in current_joints]}")
        
        current_pose = planner.get_current_pose()
        node.get_logger().info(
            f"Current pose: [{current_pose.position.x:.3f}, "
            f"{current_pose.position.y:.3f}, {current_pose.position.z:.3f}]"
        )
        
        # Example: Move to home position
        home_joints = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        
        input("Press Enter to move to home position...")
        success = planner.move_to_joint_goal(home_joints)
        
        if success:
            node.get_logger().info("✓ Successfully moved to home!")
        else:
            node.get_logger().error("✗ Failed to move to home")
        
        # Example: Move forward 10cm
        input("Press Enter to move forward 10cm...")
        current = planner.get_current_pose()
        target = planner.create_pose(
            current.position.x + 0.1,
            current.position.y,
            current.position.z,
            0.0, np.pi/2, 0.0
        )
        
        success = planner.move_to_pose_goal(target)
        
        if success:
            node.get_logger().info("✓ Successfully moved forward!")
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

