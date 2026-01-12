#!/usr/bin/env python3
"""
ALOHA Dual Arm Path Planner Example

Demonstrates coordinated motion planning for both arms:
- Synchronized motion
- Bimanual manipulation
- Coordinated pick and place

Usage:
    ros2 run aloha dual_arm_path_planner_example.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
from aloha.path_planner import AlohaDualArmPlanner, PlanningResult
import time


class DualArmDemo(Node):
    """Demo node for dual-arm coordination with path planning."""
    
    def __init__(self):
        super().__init__('dual_arm_demo')
        
        self.get_logger().info("Initializing Dual Arm Path Planner Demo...")
        
        # Create dual arm planner
        # Adjust these group names based on your robot configuration
        self.planner = AlohaDualArmPlanner(
            node=self,
            left_arm_group="left_arm",
            right_arm_group="right_arm",
            planning_time=5.0,
            max_velocity_scaling=0.3,
            max_acceleration_scaling=0.3
        )
        
        self.get_logger().info("Dual Arm Planner ready!")
    
    def demo_synchronized_home(self):
        """Move both arms to home position simultaneously."""
        self.get_logger().info("\n=== Demo 1: Synchronized Home Position ===")
        
        # Define home positions for both arms
        left_home = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        right_home = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        
        self.get_logger().info("Planning synchronized motion to home...")
        
        success, left_traj, right_traj = self.planner.plan_synchronized_motion(
            left_home, right_home
        )
        
        if success:
            self.get_logger().info("✓ Synchronized planning successful!")
            
            # Visualize both trajectories
            self.planner.left_planner.visualize_trajectory(left_traj)
            self.planner.right_planner.visualize_trajectory(right_traj)
            
            user_input = input("Execute synchronized motion? (y/n): ")
            if user_input.lower() == 'y':
                success = self.planner.execute_synchronized_motion(left_traj, right_traj)
                if success:
                    self.get_logger().info("✓ Synchronized execution successful!")
                else:
                    self.get_logger().error("✗ Synchronized execution failed!")
        else:
            self.get_logger().error("✗ Synchronized planning failed!")
    
    def demo_mirrored_motion(self):
        """Create mirrored motion for both arms."""
        self.get_logger().info("\n=== Demo 2: Mirrored Motion ===")
        
        # Get current positions
        left_current = self.planner.left_planner.get_current_joint_values()
        right_current = self.planner.right_planner.get_current_joint_values()
        
        # Create mirrored target positions
        # Mirror the waist joint (first joint)
        left_target = left_current.copy()
        right_target = right_current.copy()
        
        # Move left arm waist to +30 degrees, right to -30 degrees
        left_target[0] = np.deg2rad(30)
        right_target[0] = np.deg2rad(-30)
        
        self.get_logger().info("Planning mirrored motion...")
        
        success, left_traj, right_traj = self.planner.plan_synchronized_motion(
            left_target, right_target
        )
        
        if success:
            self.get_logger().info("✓ Mirrored motion planning successful!")
            
            user_input = input("Execute mirrored motion? (y/n): ")
            if user_input.lower() == 'y':
                self.planner.execute_synchronized_motion(left_traj, right_traj)
        else:
            self.get_logger().error("✗ Mirrored motion planning failed!")
    
    def demo_bimanual_coordination(self):
        """Demonstrate bimanual coordinated manipulation."""
        self.get_logger().info("\n=== Demo 3: Bimanual Coordination ===")
        
        # Get current poses
        left_pose = self.planner.left_planner.get_current_pose()
        right_pose = self.planner.right_planner.get_current_pose()
        
        # Create target poses that bring hands together
        meet_x = (left_pose.position.x + right_pose.position.x) / 2
        meet_y = 0.0  # Center between arms
        meet_z = 0.15
        
        left_target = self.planner.left_planner.create_pose(
            meet_x, meet_y + 0.1, meet_z, 0, np.pi/2, 0
        )
        
        right_target = self.planner.right_planner.create_pose(
            meet_x, meet_y - 0.1, meet_z, 0, np.pi/2, 0
        )
        
        self.get_logger().info("Planning to bring hands together...")
        
        # Plan for both arms
        left_result, left_traj = self.planner.left_planner.plan_to_pose_goal(left_target)
        right_result, right_traj = self.planner.right_planner.plan_to_pose_goal(right_target)
        
        if (left_result == PlanningResult.SUCCESS and 
            right_result == PlanningResult.SUCCESS):
            self.get_logger().info("✓ Bimanual coordination planning successful!")
            
            user_input = input("Execute bimanual motion? (y/n): ")
            if user_input.lower() == 'y':
                self.planner.execute_synchronized_motion(left_traj, right_traj)
        else:
            self.get_logger().error("✗ Bimanual coordination planning failed!")
    
    def demo_sequential_handoff(self):
        """Demonstrate object handoff between arms."""
        self.get_logger().info("\n=== Demo 4: Sequential Handoff ===")
        
        # Define handoff sequence
        handoff_x, handoff_y, handoff_z = 0.25, 0.0, 0.15
        
        motions = [
            ("Left arm to handoff position", 
             "left",
             self.planner.left_planner.create_pose(
                 handoff_x, handoff_y + 0.05, handoff_z, 0, np.pi/2, 0)),
            
            ("Right arm to handoff position",
             "right",
             self.planner.right_planner.create_pose(
                 handoff_x, handoff_y - 0.05, handoff_z, 0, np.pi/2, 0)),
            
            # At this point, gripper actions would occur
            
            ("Left arm retract",
             "left",
             self.planner.left_planner.create_pose(
                 handoff_x - 0.1, handoff_y + 0.1, handoff_z + 0.05, 0, np.pi/2, 0)),
            
            ("Right arm move away",
             "right",
             self.planner.right_planner.create_pose(
                 handoff_x + 0.1, handoff_y - 0.1, handoff_z, 0, np.pi/2, 0)),
        ]
        
        for step_name, arm, target_pose in motions:
            self.get_logger().info(f"\n{step_name}")
            
            if arm == "left":
                result, traj = self.planner.left_planner.plan_to_pose_goal(target_pose)
                planner = self.planner.left_planner
            else:
                result, traj = self.planner.right_planner.plan_to_pose_goal(target_pose)
                planner = self.planner.right_planner
            
            if result == PlanningResult.SUCCESS:
                self.get_logger().info(f"✓ {step_name} planning successful")
                planner.visualize_trajectory(traj)
                
                user_input = input(f"Execute {step_name}? (y/n/q): ")
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 'y':
                    success = planner.execute_trajectory(traj)
                    if not success:
                        self.get_logger().error(f"✗ {step_name} execution failed!")
                        break
                    time.sleep(0.5)
            else:
                self.get_logger().error(f"✗ {step_name} planning failed!")
                break
    
    def demo_symmetric_pattern(self):
        """Create symmetric motion patterns for both arms."""
        self.get_logger().info("\n=== Demo 5: Symmetric Pattern ===")
        
        # Get current poses
        left_pose = self.planner.left_planner.get_current_pose()
        right_pose = self.planner.right_planner.get_current_pose()
        
        # Create circular patterns (mirrored)
        radius = 0.05
        num_points = 12
        
        left_waypoints = []
        right_waypoints = []
        
        for i in range(num_points + 1):
            angle = 2 * np.pi * i / num_points
            
            # Left arm: clockwise circle
            left_x = left_pose.position.x + radius * np.cos(angle)
            left_y = left_pose.position.y + radius * np.sin(angle)
            left_waypoints.append(
                self.planner.left_planner.create_pose(
                    left_x, left_y, left_pose.position.z, 0, 0, 0
                )
            )
            
            # Right arm: counter-clockwise circle
            right_x = right_pose.position.x + radius * np.cos(-angle)
            right_y = right_pose.position.y + radius * np.sin(-angle)
            right_waypoints.append(
                self.planner.right_planner.create_pose(
                    right_x, right_y, right_pose.position.z, 0, 0, 0
                )
            )
        
        self.get_logger().info("Planning symmetric circular patterns...")
        
        # Plan Cartesian paths
        left_result, left_traj, left_frac = self.planner.left_planner.plan_cartesian_path(
            left_waypoints, eef_step=0.005
        )
        
        right_result, right_traj, right_frac = self.planner.right_planner.plan_cartesian_path(
            right_waypoints, eef_step=0.005
        )
        
        if (left_result == PlanningResult.SUCCESS and 
            right_result == PlanningResult.SUCCESS):
            self.get_logger().info(
                f"✓ Symmetric pattern planning successful! "
                f"Left: {left_frac:.1%}, Right: {right_frac:.1%}"
            )
            
            # Visualize
            self.planner.left_planner.visualize_trajectory(left_traj)
            self.planner.right_planner.visualize_trajectory(right_traj)
            
            user_input = input("Execute symmetric patterns? (y/n): ")
            if user_input.lower() == 'y':
                self.planner.execute_synchronized_motion(left_traj, right_traj)
        else:
            self.get_logger().error("✗ Symmetric pattern planning failed!")
    
    def demo_wide_to_narrow_motion(self):
        """Move arms from wide stance to narrow stance."""
        self.get_logger().info("\n=== Demo 6: Wide to Narrow Motion ===")
        
        # Wide position: arms spread apart
        wide_y_offset = 0.3
        narrow_y_offset = 0.1
        
        x = 0.3
        z = 0.15
        
        # Create poses for wide position
        left_wide = self.planner.left_planner.create_pose(
            x, wide_y_offset, z, 0, np.pi/2, 0
        )
        right_wide = self.planner.right_planner.create_pose(
            x, -wide_y_offset, z, 0, np.pi/2, 0
        )
        
        # Create poses for narrow position
        left_narrow = self.planner.left_planner.create_pose(
            x, narrow_y_offset, z, 0, np.pi/2, 0
        )
        right_narrow = self.planner.right_planner.create_pose(
            x, -narrow_y_offset, z, 0, np.pi/2, 0
        )
        
        # Plan wide position
        self.get_logger().info("Planning to wide position...")
        left_result1, left_traj1 = self.planner.left_planner.plan_to_pose_goal(left_wide)
        right_result1, right_traj1 = self.planner.right_planner.plan_to_pose_goal(right_wide)
        
        if (left_result1 == PlanningResult.SUCCESS and 
            right_result1 == PlanningResult.SUCCESS):
            
            user_input = input("Execute wide position? (y/n): ")
            if user_input.lower() == 'y':
                self.planner.execute_synchronized_motion(left_traj1, right_traj1)
                time.sleep(1.0)
                
                # Plan narrow position
                self.get_logger().info("Planning to narrow position...")
                left_result2, left_traj2 = self.planner.left_planner.plan_to_pose_goal(left_narrow)
                right_result2, right_traj2 = self.planner.right_planner.plan_to_pose_goal(right_narrow)
                
                if (left_result2 == PlanningResult.SUCCESS and 
                    right_result2 == PlanningResult.SUCCESS):
                    
                    user_input = input("Execute narrow position? (y/n): ")
                    if user_input.lower() == 'y':
                        self.planner.execute_synchronized_motion(left_traj2, right_traj2)
    
    def run_all_demos(self):
        """Run demo selection menu."""
        demos = [
            ("Synchronized Home", self.demo_synchronized_home),
            ("Mirrored Motion", self.demo_mirrored_motion),
            ("Bimanual Coordination", self.demo_bimanual_coordination),
            ("Sequential Handoff", self.demo_sequential_handoff),
            ("Symmetric Pattern", self.demo_symmetric_pattern),
            ("Wide to Narrow", self.demo_wide_to_narrow_motion),
        ]
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("ALOHA Dual Arm Path Planner Demo Suite")
        self.get_logger().info("="*60)
        
        for i, (name, _) in enumerate(demos, 1):
            print(f"{i}. {name}")
        print("0. Run all demos")
        print("q. Quit")
        
        while True:
            choice = input("\nSelect demo (0-6, q): ").strip()
            
            if choice == 'q':
                break
            elif choice == '0':
                for name, demo_func in demos:
                    try:
                        demo_func()
                        input("\nPress Enter to continue to next demo...")
                    except Exception as e:
                        self.get_logger().error(f"Demo failed: {e}")
                        break
            elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                try:
                    name, demo_func = demos[int(choice) - 1]
                    demo_func()
                except Exception as e:
                    self.get_logger().error(f"Demo failed: {e}")
            else:
                print("Invalid choice. Please try again.")


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        demo = DualArmDemo()
        demo.run_all_demos()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

