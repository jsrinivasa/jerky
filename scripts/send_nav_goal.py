#!/usr/bin/env python3
"""
Example: Send navigation goals programmatically

This example shows how to send goal poses to the navigation planner
without using RViz2.

Usage:
    python3 send_nav_goal.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math


class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')
        
        # Publisher for goal poses
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )
        
        self.get_logger().info('Goal Sender Node Started')
        
    def send_goal(self, x, y, yaw=0.0):
        """
        Send a navigation goal.
        
        Args:
            x: X coordinate in map frame (meters)
            y: Y coordinate in map frame (meters)
            yaw: Orientation in radians (default: 0.0)
        """
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.goal_pub.publish(goal)
        
        self.get_logger().info(
            f'Sent goal: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°'
        )


def main():
    rclpy.init()
    node = GoalSender()
    
    try:
        # Example 1: Send goal to (2.0, 1.0)
        print("\n=== Example 1: Simple Goal ===")
        print("Sending robot to x=2.0, y=1.0")
        input("Press Enter to send goal...")
        node.send_goal(2.0, 1.0)
        
        # Example 2: Send goal with specific orientation
        print("\n=== Example 2: Goal with Orientation ===")
        print("Sending robot to x=1.0, y=2.0, facing 90 degrees")
        input("Press Enter to send goal...")
        node.send_goal(1.0, 2.0, yaw=math.pi/2)
        
        # Example 3: Send multiple waypoints
        print("\n=== Example 3: Multiple Waypoints ===")
        waypoints = [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, math.pi/2),
            (0.0, 1.0, math.pi),
            (0.0, 0.0, -math.pi/2),
        ]
        
        for i, (x, y, yaw) in enumerate(waypoints):
            print(f"\nWaypoint {i+1}/{len(waypoints)}: ({x}, {y})")
            input("Press Enter to send next waypoint...")
            node.send_goal(x, y, yaw)
            
            # Wait for robot to reach goal
            print("Wait for robot to reach goal before sending next...")
            input("Press Enter when ready for next waypoint...")
        
        print("\n=== All examples complete! ===")
        
        # Keep node alive
        print("\nNode will continue running. Press Ctrl+C to exit.")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

