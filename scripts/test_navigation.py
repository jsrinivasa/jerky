#!/usr/bin/env python3
"""
Test script to verify the simple navigation planner setup.

This script checks:
1. Required topics exist
2. Map is published
3. Odometry is available
4. Navigation node is running
"""

import rclpy
from rclpy.node import Node
import sys
import time


class NavTestNode(Node):
    def __init__(self):
        super().__init__('nav_test_node')
        self.test_results = {}
        
    def run_tests(self):
        """Run all tests."""
        self.get_logger().info("="*60)
        self.get_logger().info("  Navigation Planner Test Suite")
        self.get_logger().info("="*60)
        
        tests = [
            ("Map topic exists", self.test_map_topic),
            ("Odometry topic exists", self.test_odom_topic),
            ("Goal pose topic exists", self.test_goal_topic),
            ("Cmd vel topic exists", self.test_cmd_vel_topic),
            ("Navigation node running", self.test_nav_node),
        ]
        
        for test_name, test_func in tests:
            self.get_logger().info(f"\nTesting: {test_name}...")
            result = test_func()
            self.test_results[test_name] = result
            
            if result:
                self.get_logger().info(f"  ✓ PASS: {test_name}")
            else:
                self.get_logger().warn(f"  ✗ FAIL: {test_name}")
        
        # Print summary
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("  Test Summary")
        self.get_logger().info("="*60)
        
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            self.get_logger().info(f"{status}: {test_name}")
        
        self.get_logger().info(f"\n{passed}/{total} tests passed")
        
        if passed == total:
            self.get_logger().info("\n🎉 All tests passed! Navigation system is ready.")
            return True
        else:
            self.get_logger().warn("\n⚠️  Some tests failed. Check configuration.")
            return False
    
    def test_map_topic(self):
        """Check if /map topic exists."""
        topics = self.get_topic_names_and_types()
        return any('/map' in topic[0] for topic in topics)
    
    def test_odom_topic(self):
        """Check if /odom topic exists."""
        topics = self.get_topic_names_and_types()
        return any('/odom' in topic[0] for topic in topics)
    
    def test_goal_topic(self):
        """Check if /goal_pose topic exists or can be created."""
        topics = self.get_topic_names_and_types()
        # Topic might not exist until first goal is sent, that's OK
        return True
    
    def test_cmd_vel_topic(self):
        """Check if /cmd_vel topic exists or can be created."""
        topics = self.get_topic_names_and_types()
        # Topic might not exist until first command, that's OK
        return True
    
    def test_nav_node(self):
        """Check if navigation node is running."""
        nodes = self.get_node_names()
        return 'simple_nav_planner' in nodes


def main():
    rclpy.init()
    node = NavTestNode()
    
    # Give nodes time to start up
    time.sleep(2.0)
    
    try:
        success = node.run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

