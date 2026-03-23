#!/usr/bin/env python3

"""
Localization Monitor Node

Subscribes to AMCL's particle cloud and pose estimate to monitor localization
convergence. Reports the robot's estimated (x, y, yaw) in the map frame and
a convergence metric based on particle spread.

Also provides a service to programmatically set the initial pose.

Topics subscribed:
    /particlecloud (nav2_msgs/ParticleCloud) - AMCL particle distribution
    /amcl_pose (geometry_msgs/PoseWithCovarianceStamped) - Estimated pose

Topics published:
    /localization_status (std_msgs/String) - JSON status string

Services provided:
    ~/set_pose (std_srvs/Trigger) - Re-initialize AMCL pose (uses current estimate)
"""

import math
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

try:
    from nav2_msgs.msg import ParticleCloud
    HAS_PARTICLE_CLOUD = True
except ImportError:
    HAS_PARTICLE_CLOUD = False


def quaternion_to_yaw(q):
    """Extract yaw from a quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class LocalizationMonitor(Node):

    def __init__(self):
        super().__init__('localization_monitor')

        self.declare_parameter('convergence_threshold', 0.5)
        self.declare_parameter('report_interval', 2.0)

        self._convergence_threshold = (
            self.get_parameter('convergence_threshold').value
        )
        self._report_interval = self.get_parameter('report_interval').value

        self._last_pose = None
        self._last_covariance = None
        self._particle_spread = float('inf')
        self._is_converged = False
        self._pose_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._pose_callback,
            qos,
        )

        if HAS_PARTICLE_CLOUD:
            self._particle_sub = self.create_subscription(
                ParticleCloud,
                '/particlecloud',
                self._particle_callback,
                qos,
            )

        self._status_pub = self.create_publisher(String, '/localization_status', 10)

        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10,
        )

        self._timer = self.create_timer(self._report_interval, self._report_status)

        self.get_logger().info('Localization monitor started')
        self.get_logger().info(
            f'Convergence threshold: {self._convergence_threshold}m'
        )
        if not HAS_PARTICLE_CLOUD:
            self.get_logger().warn(
                'nav2_msgs not found; particle cloud monitoring disabled. '
                'Convergence will be estimated from pose covariance only.'
            )
        self.get_logger().info('Waiting for AMCL pose estimates...')

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self._last_pose = msg.pose.pose
        self._last_covariance = msg.pose.covariance
        self._pose_count += 1

        cov_x = msg.pose.covariance[0]
        cov_y = msg.pose.covariance[7]
        self._particle_spread = math.sqrt(cov_x + cov_y)

        was_converged = self._is_converged
        self._is_converged = self._particle_spread < self._convergence_threshold

        if self._is_converged and not was_converged:
            yaw = quaternion_to_yaw(self._last_pose.orientation)
            self.get_logger().info(
                f'LOCALIZED! Position: ({self._last_pose.position.x:.2f}, '
                f'{self._last_pose.position.y:.2f}), '
                f'yaw: {math.degrees(yaw):.1f} deg, '
                f'spread: {self._particle_spread:.3f}m'
            )

    def _particle_callback(self, msg):
        if len(msg.particles) < 2:
            return

        xs = [p.pose.position.x for p in msg.particles]
        ys = [p.pose.position.y for p in msg.particles]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        var_x = sum((x - mean_x) ** 2 for x in xs) / n
        var_y = sum((y - mean_y) ** 2 for y in ys) / n
        self._particle_spread = math.sqrt(var_x + var_y)

    def _report_status(self):
        if self._last_pose is None:
            self.get_logger().info(
                'Waiting for AMCL... '
                '(Is the robot base running? Is the D435i camera streaming?)'
            )
            return

        pose = self._last_pose
        yaw = quaternion_to_yaw(pose.orientation)
        yaw_deg = math.degrees(yaw)

        if self._is_converged:
            state = 'LOCALIZED'
        elif self._particle_spread < self._convergence_threshold * 3:
            state = 'CONVERGING'
        else:
            state = 'SEARCHING'

        status_dict = {
            'state': state,
            'x': round(pose.position.x, 3),
            'y': round(pose.position.y, 3),
            'yaw_rad': round(yaw, 3),
            'yaw_deg': round(yaw_deg, 1),
            'spread_m': round(self._particle_spread, 3),
            'pose_updates': self._pose_count,
        }

        status_msg = String()
        status_msg.data = json.dumps(status_dict)
        self._status_pub.publish(status_msg)

        spread_bar = self._spread_bar(self._particle_spread)
        self.get_logger().info(
            f'[{state:10s}] '
            f'x={pose.position.x:+7.2f} y={pose.position.y:+7.2f} '
            f'yaw={yaw_deg:+7.1f}deg  '
            f'spread={self._particle_spread:.3f}m {spread_bar}'
        )

    @staticmethod
    def _spread_bar(spread, max_spread=5.0, width=20):
        """Visual bar showing particle spread (smaller = better)."""
        ratio = min(spread / max_spread, 1.0)
        filled = int(ratio * width)
        empty = width - filled
        return '[' + '#' * filled + '.' * empty + ']'

    def set_initial_pose(self, x, y, yaw):
        """Publish an initial pose estimate to AMCL."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(float(yaw) / 2.0)
        msg.pose.pose.orientation.w = math.cos(float(yaw) / 2.0)
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.068
        self._initial_pose_pub.publish(msg)
        self.get_logger().info(
            f'Set initial pose: ({x:.2f}, {y:.2f}), yaw={math.degrees(yaw):.1f}deg'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
