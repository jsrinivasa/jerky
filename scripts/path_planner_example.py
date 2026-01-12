#!/usr/bin/env python3
"""
ALOHA Path Planner Example Script

Demonstrates various path planning capabilities:
- Joint space planning
- Cartesian space planning
- Waypoint following
- Collision avoidance

Usage:
    ros2 run aloha path_planner_example.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
from aloha.path_planner import AlohaPathPlanner, PlanningResult
import time


class PathPlannerDemo(Node):
    """Demo node for ALOHA path planning examples."""
    
    def __init__(self):
        super().__init__('path_planner_demo')
        
        self.get_logger().info("Initializing Path Planner Demo...")
        
        # Create path planner with custom parameters
        self.planner = AlohaPathPlanner(
            node=self,
            arm_group="interbotix_arm",
            planning_time=5.0,
            max_velocity_scaling=0.3,
            max_acceleration_scaling=0.3
        )
        
        self.get_logger().info("Path Planner Demo ready!")
    
    def demo_joint_space_planning(self):
        """Demonstrate planning in joint space."""
        self.get_logger().info("\n=== Demo 1: Joint Space Planning ===")
        
        # Get current joint values
        current = self.planner.get_current_joint_values()
        self.get_logger().info(f"Current joints: {[f'{j:.3f}' for j in current]}")
        
        # Define some target joint configurations
        home_position = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        upright_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Plan to home position
        self.get_logger().info("Planning to home position...")
        result, trajectory = self.planner.plan_to_joint_goal(home_position)
        
        if result == PlanningResult.SUCCESS:
            self.get_logger().info("✓ Planning successful!")
            
            # Visualize in RViz
            self.planner.visualize_trajectory(trajectory)
            
            # Optionally execute
            user_input = input("Execute trajectory? (y/n): ")
            if user_input.lower() == 'y':
                success = self.planner.execute_trajectory(trajectory)
                if success:
                    self.get_logger().info("✓ Execution successful!")
                else:
                    self.get_logger().error("✗ Execution failed!")
        else:
            self.get_logger().error(f"✗ Planning failed: {result}")
    
    def demo_cartesian_planning(self):
        """Demonstrate planning in Cartesian space."""
        self.get_logger().info("\n=== Demo 2: Cartesian Space Planning ===")
        
        # Get current pose
        current_pose = self.planner.get_current_pose()
        self.get_logger().info(f"Current pose: x={current_pose.position.x:.3f}, "
                              f"y={current_pose.position.y:.3f}, "
                              f"z={current_pose.position.z:.3f}")
        
        # Create target pose (move forward 10cm in x)
        target_pose = self.planner.create_pose(
            x=current_pose.position.x + 0.1,
            y=current_pose.position.y,
            z=current_pose.position.z,
            roll=0.0, pitch=0.0, yaw=0.0
        )
        
        self.get_logger().info(f"Target pose: x={target_pose.position.x:.3f}, "
                              f"y={target_pose.position.y:.3f}, "
                              f"z={target_pose.position.z:.3f}")
        
        # Plan to target pose
        result, trajectory = self.planner.plan_to_pose_goal(target_pose)
        
        if result == PlanningResult.SUCCESS:
            self.get_logger().info("✓ Cartesian planning successful!")
            self.planner.visualize_trajectory(trajectory)
            
            user_input = input("Execute trajectory? (y/n): ")
            if user_input.lower() == 'y':
                self.planner.execute_trajectory(trajectory)
        elif result == PlanningResult.NO_IK_SOLUTION:
            self.get_logger().error("✗ No inverse kinematics solution found!")
        else:
            self.get_logger().error(f"✗ Planning failed: {result}")
    
    def demo_waypoint_following(self):
        """Demonstrate following a series of waypoints."""
        self.get_logger().info("\n=== Demo 3: Waypoint Following ===")
        
        # Get current pose as starting point
        current_pose = self.planner.get_current_pose()
        
        # Create a square pattern of waypoints
        waypoints = []
        base_x = current_pose.position.x
        base_y = current_pose.position.y
        base_z = current_pose.position.z
        
        # Square corners
        square_size = 0.1  # 10cm square
        
        waypoints.append(self.planner.create_pose(
            base_x + square_size, base_y, base_z, 0, 0, 0))
        waypoints.append(self.planner.create_pose(
            base_x + square_size, base_y + square_size, base_z, 0, 0, 0))
        waypoints.append(self.planner.create_pose(
            base_x, base_y + square_size, base_z, 0, 0, 0))
        waypoints.append(self.planner.create_pose(
            base_x, base_y, base_z, 0, 0, 0))
        
        self.get_logger().info(f"Planning path through {len(waypoints)} waypoints...")
        
        # Plan Cartesian path
        result, trajectory, fraction = self.planner.plan_cartesian_path(
            waypoints,
            eef_step=0.01,
            jump_threshold=0.0
        )
        
        if result == PlanningResult.SUCCESS:
            self.get_logger().info(f"✓ Waypoint planning successful! Achieved: {fraction:.1%}")
            self.planner.visualize_trajectory(trajectory)
            
            if fraction >= 0.95:
                user_input = input("Execute trajectory? (y/n): ")
                if user_input.lower() == 'y':
                    self.planner.execute_trajectory(trajectory)
            else:
                self.get_logger().warn(f"Only {fraction:.1%} of path achieved - not executing")
        else:
            self.get_logger().error(f"✗ Waypoint planning failed: {result}")
    
    def demo_circular_motion(self):
        """Demonstrate circular motion in Cartesian space."""
        self.get_logger().info("\n=== Demo 4: Circular Motion ===")
        
        # Get current pose
        current_pose = self.planner.get_current_pose()
        
        # Create circular waypoints
        center_x = current_pose.position.x
        center_y = current_pose.position.y
        z = current_pose.position.z
        
        radius = 0.05  # 5cm radius
        num_points = 16
        
        waypoints = []
        for i in range(num_points + 1):
            angle = 2 * np.pi * i / num_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            waypoints.append(self.planner.create_pose(x, y, z, 0, 0, 0))
        
        self.get_logger().info(f"Planning circular path with {len(waypoints)} waypoints...")
        
        result, trajectory, fraction = self.planner.plan_cartesian_path(
            waypoints,
            eef_step=0.005
        )
        
        if result == PlanningResult.SUCCESS:
            self.get_logger().info(f"✓ Circular motion planned! Fraction: {fraction:.1%}")
            self.planner.visualize_trajectory(trajectory)
            
            user_input = input("Execute trajectory? (y/n): ")
            if user_input.lower() == 'y':
                self.planner.execute_trajectory(trajectory)
        else:
            self.get_logger().error(f"✗ Circular motion planning failed: {result}")
    
    def demo_pick_and_place(self):
        """Demonstrate a simple pick and place motion."""
        self.get_logger().info("\n=== Demo 5: Pick and Place ===")
        
        # Define pick and place poses
        # These would need to be adjusted for your specific setup
        approach_height = 0.15
        grasp_height = 0.05
        
        # Pick location
        pick_x, pick_y = 0.3, 0.1
        
        # Place location
        place_x, place_y = 0.3, -0.1
        
        # Create motion sequence
        motions = [
            ("Approach pick", self.planner.create_pose(
                pick_x, pick_y, approach_height, 0, np.pi/2, 0)),
            ("Grasp", self.planner.create_pose(
                pick_x, pick_y, grasp_height, 0, np.pi/2, 0)),
            ("Lift", self.planner.create_pose(
                pick_x, pick_y, approach_height, 0, np.pi/2, 0)),
            ("Approach place", self.planner.create_pose(
                place_x, place_y, approach_height, 0, np.pi/2, 0)),
            ("Place", self.planner.create_pose(
                place_x, place_y, grasp_height, 0, np.pi/2, 0)),
            ("Retract", self.planner.create_pose(
                place_x, place_y, approach_height, 0, np.pi/2, 0)),
        ]
        
        for name, target_pose in motions:
            self.get_logger().info(f"\nPlanning: {name}")
            result, trajectory = self.planner.plan_to_pose_goal(target_pose)
            
            if result == PlanningResult.SUCCESS:
                self.get_logger().info(f"✓ {name} planning successful")
                self.planner.visualize_trajectory(trajectory)
                
                user_input = input(f"Execute {name}? (y/n/q): ")
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 'y':
                    success = self.planner.execute_trajectory(trajectory)
                    if not success:
                        self.get_logger().error(f"✗ {name} execution failed!")
                        break
                    time.sleep(0.5)  # Brief pause between motions
            else:
                self.get_logger().error(f"✗ {name} planning failed: {result}")
                break
    
    def demo_parameter_tuning(self):
        """Demonstrate adjusting planning parameters."""
        self.get_logger().info("\n=== Demo 6: Parameter Tuning ===")
        
        # Show current parameters
        self.get_logger().info(f"Current planning time: {self.planner.planning_time}s")
        self.get_logger().info(f"Current velocity scaling: {self.planner.max_velocity_scaling}")
        
        # Try planning with different speeds
        velocities = [0.1, 0.3, 0.5, 0.8]
        target_joints = [0.0, -0.5, 0.8, 0.0, -0.3, 0.0]
        
        for vel in velocities:
            self.get_logger().info(f"\n--- Testing with velocity scaling: {vel} ---")
            self.planner.set_planning_parameters(velocity_scaling=vel)
            
            result, trajectory = self.planner.plan_to_joint_goal(target_joints)
            if result == PlanningResult.SUCCESS:
                # Would measure execution time here
                self.get_logger().info(f"✓ Planning successful with vel={vel}")
            else:
                self.get_logger().error(f"✗ Planning failed with vel={vel}")
    
    def run_all_demos(self):
        """Run all demonstration examples."""
        demos = [
            ("Joint Space Planning", self.demo_joint_space_planning),
            ("Cartesian Planning", self.demo_cartesian_planning),
            ("Waypoint Following", self.demo_waypoint_following),
            ("Circular Motion", self.demo_circular_motion),
            ("Pick and Place", self.demo_pick_and_place),
            ("Parameter Tuning", self.demo_parameter_tuning),
        ]
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("ALOHA Path Planner Demo Suite")
        self.get_logger().info("="*60)
        
        for i, (name, demo_func) in enumerate(demos, 1):
            print(f"\n{i}. {name}")
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
        demo = PathPlannerDemo()
        demo.run_all_demos()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

