#!/usr/bin/env python3
"""
Fake Odometry Node for Simulation

Simulates robot odometry by integrating velocity commands.
Subscribes to /cmd_vel and publishes /odom and TF.

This is for testing navigation without real hardware.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster
import math


class FakeOdometry(Node):
    def __init__(self):
        super().__init__('fake_odometry')
        
        # Parameters
        self.declare_parameter('update_rate', 30.0)
        
        self.update_rate = self.get_parameter('update_rate').value
        
        # Robot state (pose and velocity)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.vtheta = 0.0
        
        # Current command
        self.cmd_vel = Twist()
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Timer for updating odometry
        self.timer = self.create_timer(1.0 / self.update_rate, self.update_odometry)
        
        self.get_logger().info('Fake Odometry Node Started')
        self.get_logger().info('  Publishing /odom and odom->base_footprint TF')
        self.get_logger().info('  Subscribing to /cmd_vel for simulation')
        self.get_logger().info(f'  Update rate: {self.update_rate} Hz')
    
    def cmd_vel_callback(self, msg: Twist):
        """Store the latest velocity command."""
        self.vx = msg.linear.x
        self.vtheta = msg.angular.z
    
    def update_odometry(self):
        """Update robot pose based on velocity and publish odometry."""
        current_time = self.get_clock().now()
        dt = 1.0 / self.update_rate
        
        # Update pose (simple integration)
        # This is a simplified model - real robots have slip, acceleration limits, etc.
        delta_x = self.vx * math.cos(self.theta) * dt
        delta_y = self.vx * math.sin(self.theta) * dt
        delta_theta = self.vtheta * dt
        
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Create and publish odometry message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        
        # Position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        
        # Orientation (convert theta to quaternion)
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.theta / 2.0)
        
        # Velocity
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = self.vtheta
        
        # Publish odometry
        self.odom_pub.publish(odom)
        
        # Publish TF
        tf = TransformStamped()
        tf.header.stamp = current_time.to_msg()
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_footprint'
        
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = 0.0
        
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = math.sin(self.theta / 2.0)
        tf.transform.rotation.w = math.cos(self.theta / 2.0)
        
        self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = FakeOdometry()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

