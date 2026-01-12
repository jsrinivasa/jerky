#!/usr/bin/env python3
"""
Quick Start Path Planner Example

The simplest possible example to get you started with path planning.
This script demonstrates the most common operations.

Usage:
    ros2 run aloha quick_start_planner
    
Or run directly:
    python3 quick_start_planner.py
"""

import rclpy
from rclpy.node import Node
import numpy as np

# Import the simple path planner
from aloha.simple_path_planner import SimplePathPlanner


def main():
    """Quick start example."""
    
    # Initialize ROS 2
    rclpy.init()
    node = rclpy.create_node('quick_start_planner')
    logger = node.get_logger()
    
    logger.info("="*60)
    logger.info("  ALOHA Path Planner - Quick Start")
    logger.info("="*60)
    
    try:
        # Step 1: Create the planner
        logger.info("\n[1/5] Creating path planner...")
        planner = SimplePathPlanner(
            node,
            group_name="interbotix_arm",
            velocity_scaling=0.3,
            acceleration_scaling=0.3
        )
        logger.info("✓ Planner created!")
        
        # Step 2: Check current state
        logger.info("\n[2/5] Getting current robot state...")
        current_joints = planner.get_current_joint_values()
        current_pose = planner.get_current_pose()
        
        logger.info(f"  Joint angles: {[f'{j:.3f}' for j in current_joints]}")
        logger.info(f"  Position: x={current_pose.position.x:.3f}, "
                   f"y={current_pose.position.y:.3f}, "
                   f"z={current_pose.position.z:.3f}")
        logger.info("✓ Current state retrieved!")
        
        # Step 3: Plan and move to home position
        logger.info("\n[3/5] Moving to home position...")
        home_position = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        
        input("  Press Enter to execute motion (or Ctrl+C to skip)...")
        
        success = planner.move_to_joint_goal(home_position)
        if success:
            logger.info("✓ Successfully reached home position!")
        else:
            logger.warn("⚠ Could not reach home position (this is OK for demo)")
        
        # Step 4: Move to a Cartesian pose
        logger.info("\n[4/5] Moving to a target pose...")
        
        # Create a target pose 30cm forward, centered, 20cm high
        target_pose = planner.create_pose(
            x=0.3,          # 30cm forward
            y=0.0,          # centered
            z=0.2,          # 20cm high
            roll=0.0,       # no roll
            pitch=np.pi/2,  # pointing down
            yaw=0.0         # no yaw
        )
        
        logger.info(f"  Target: x={target_pose.position.x:.3f}, "
                   f"y={target_pose.position.y:.3f}, "
                   f"z={target_pose.position.z:.3f}")
        
        input("  Press Enter to execute motion (or Ctrl+C to skip)...")
        
        # First, let's just plan (not execute) to see if it's reachable
        can_reach, plan = planner.plan_to_pose_goal(target_pose)
        
        if can_reach:
            logger.info("✓ Target is reachable! Executing...")
            success = planner.execute_plan(plan)
            if success:
                logger.info("✓ Successfully reached target pose!")
        else:
            logger.warn("⚠ Target pose not reachable (may be outside workspace)")
        
        # Step 5: Draw a small circle
        logger.info("\n[5/5] Drawing a circle with Cartesian path...")
        
        current = planner.get_current_pose()
        
        # Create circular waypoints (5cm radius, 12 points)
        radius = 0.05
        num_points = 12
        waypoints = []
        
        for i in range(num_points + 1):
            angle = 2 * np.pi * i / num_points
            x = current.position.x + radius * np.cos(angle)
            y = current.position.y + radius * np.sin(angle)
            z = current.position.z
            
            waypoint = planner.create_pose(x, y, z, 0.0, np.pi/2, 0.0)
            waypoints.append(waypoint)
        
        logger.info(f"  Planning path with {len(waypoints)} waypoints...")
        
        input("  Press Enter to execute circular motion (or Ctrl+C to skip)...")
        
        plan, fraction = planner.plan_cartesian_path(waypoints, eef_step=0.005)
        
        if fraction > 0.95:
            logger.info(f"✓ Path computed! {fraction*100:.1f}% achieved")
            logger.info("  Executing circular motion...")
            success = planner.execute_plan(plan)
            if success:
                logger.info("✓ Circle completed!")
        else:
            logger.warn(f"⚠ Could only achieve {fraction*100:.1f}% of path")
        
        # Done!
        logger.info("\n" + "="*60)
        logger.info("  Quick Start Complete! 🎉")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("  1. Try: ros2 run aloha path_planner_example.py")
        logger.info("  2. Read: src/aloha/docs/PATH_PLANNER_GUIDE.md")
        logger.info("  3. Experiment with your own waypoints!")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"\nError: {e}")
        logger.error("Make sure:")
        logger.error("  1. Robot is launched: ros2 launch aloha aloha_bringup.launch.py")
        logger.error("  2. MoveIt is running")
        logger.error("  3. Planning group name is correct")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

