#!/usr/bin/env python3

"""
Autonomous Localization

Monitors RTAB-Map for visual loop closures.  If the robot hasn't localized
within a timeout, slowly rotates in place to show the camera different views.
Once a loop closure is detected, stops rotating and reports success.

Publishes to /nav_cmd_vel (NOT /mobile_base/cmd_vel directly) -- that
intermediate topic is what nav_deadman.py gates behind holding L2. Publishing
straight to /mobile_base/cmd_vel would let this spin the robot with no
deadman/kill-switch at all, the same class of bug as teleop_twist_joy fighting
nav_deadman (see nav-wall-collision-teleop-fight memory) except worse -- no
second publisher needed to cause harm, this alone would be enough.
2026-07-20 fix: was publishing directly to /mobile_base/cmd_vel.

Usage (after launching navigate_mission.launch.py / rtabmap_localization.launch.py):
    ros2 run aloha auto_localize
    # Hold L2 on the controller while this runs -- it moves nothing until you do.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rtabmap_msgs.msg import Info


class AutoLocalize(Node):

    def __init__(self):
        super().__init__('auto_localize')

        self.declare_parameter('rotation_speed', 0.3)
        self.declare_parameter('timeout_before_rotate', 3.0)
        self.declare_parameter('max_rotation_time', 90.0)

        self._rotation_speed = self.get_parameter('rotation_speed').value
        self._timeout = self.get_parameter('timeout_before_rotate').value
        self._max_time = self.get_parameter('max_rotation_time').value

        self._localized = False
        self._closure_count = 0
        self._start_time = self.get_clock().now()
        self._rotating = False

        self._cmd_pub = self.create_publisher(
            Twist, '/nav_cmd_vel', 10
        )
        self.create_subscription(
            Info, '/rtabmap/info', self._info_cb, 10
        )
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Waiting for visual localization '
            f'(will rotate after {self._timeout:.0f}s if needed) -- '
            'HOLD L2 on the controller, nav_deadman gates this...'
        )

    def _info_cb(self, msg):
        if msg.loop_closure_id > 0 or msg.proximity_detection_id > 0:
            self._closure_count += 1
            source = (
                f'loop closure (node {msg.loop_closure_id})'
                if msg.loop_closure_id > 0
                else f'proximity detection (node {msg.proximity_detection_id})'
            )
            self.get_logger().info(
                f'Visual match #{self._closure_count}: {source}'
            )

            if self._closure_count >= 2 and not self._localized:
                self._localized = True
                self._stop_robot()
                elapsed = (
                    self.get_clock().now() - self._start_time
                ).nanoseconds / 1e9
                self.get_logger().info(
                    '========================================\n'
                    f'  LOCALIZED in {elapsed:.1f} seconds\n'
                    f'  ({self._closure_count} visual matches confirmed)\n'
                    '========================================'
                )

    def _control_loop(self):
        if self._localized:
            return

        elapsed = (
            self.get_clock().now() - self._start_time
        ).nanoseconds / 1e9

        if elapsed > self._max_time:
            self._stop_robot()
            self.get_logger().warn(
                f'Failed to localize after {self._max_time:.0f}s. '
                'Try starting in a mapped area with visible features.'
            )
            raise SystemExit(1)

        if elapsed > self._timeout and not self._rotating:
            self._rotating = True
            self.get_logger().info(
                'No match yet -- rotating to find visual landmarks... '
                '(hold L2 or this goes nowhere)'
            )

        if self._rotating:
            twist = Twist()
            twist.angular.z = self._rotation_speed
            self._cmd_pub.publish(twist)

    def _stop_robot(self):
        self._rotating = False
        self._cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = AutoLocalize()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
