#!/usr/bin/env python3

"""
Laser Scan Merger

Merges two LaserScan topics (front + rear cameras) into a single scan
in the base_link frame.  Each incoming beam is projected into base_link
using the TF tree, binned by angle, and the nearest range per bin wins.

The output scan covers the full -pi..pi range so AMCL can correctly
interpret the beam angles, but diagnostics are logged periodically to
help verify that both cameras are contributing valid data.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
import tf2_ros


class LaserScanMerger(Node):

    def __init__(self):
        super().__init__('laser_scan_merger')

        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('angle_min', -math.pi)
        self.declare_parameter('angle_max', math.pi)
        self.declare_parameter('angle_increment', 0.005)
        self.declare_parameter('range_min', 0.1)
        self.declare_parameter('range_max', 5.0)
        self.declare_parameter('publish_rate', 15.0)

        self._target_frame = self.get_parameter('target_frame').value
        self._angle_min = self.get_parameter('angle_min').value
        self._angle_max = self.get_parameter('angle_max').value
        self._angle_inc = self.get_parameter('angle_increment').value
        self._range_min = self.get_parameter('range_min').value
        self._range_max = self.get_parameter('range_max').value
        publish_rate = self.get_parameter('publish_rate').value

        self._num_beams = int(
            (self._angle_max - self._angle_min) / self._angle_inc
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._latest_scans = {}
        self._diag_counter = 0
        self._diag_interval = int(publish_rate * 5)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        self.create_subscription(
            LaserScan, '/scan_front', self._make_cb('front'), qos)
        self.create_subscription(
            LaserScan, '/scan_rear', self._make_cb('rear'), qos)

        scan_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )
        self._pub = self.create_publisher(LaserScan, '/scan', scan_pub_qos)
        self.create_timer(1.0 / publish_rate, self._merge_and_publish)

        self.get_logger().info(
            f'Merging /scan_front + /scan_rear -> /scan '
            f'({self._num_beams} bins, frame={self._target_frame})'
        )

    def _make_cb(self, key):
        def cb(msg):
            self._latest_scans[key] = msg
        return cb

    def _merge_and_publish(self):
        if not self._latest_scans:
            return

        merged = [float('inf')] * self._num_beams
        per_source_counts = {}

        for name, scan in self._latest_scans.items():
            try:
                tf_stamped = self._tf_buffer.lookup_transform(
                    self._target_frame,
                    scan.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                if self._diag_counter % self._diag_interval == 0:
                    self.get_logger().warn(
                        f'[{name}] TF lookup failed '
                        f'({self._target_frame} <- {scan.header.frame_id}): {e}'
                    )
                per_source_counts[name] = 0
                continue

            q = tf_stamped.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            tx = tf_stamped.transform.translation.x
            ty = tf_stamped.transform.translation.y

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            count = 0
            angle = scan.angle_min
            for r in scan.ranges:
                if (not math.isnan(r)
                        and not math.isinf(r)
                        and scan.range_min <= r <= scan.range_max):
                    px = r * math.cos(angle)
                    py = r * math.sin(angle)

                    px_t = cos_yaw * px - sin_yaw * py + tx
                    py_t = sin_yaw * px + cos_yaw * py + ty

                    angle_t = math.atan2(py_t, px_t)
                    range_t = math.hypot(px_t, py_t)

                    if self._range_min <= range_t <= self._range_max:
                        idx = int(
                            (angle_t - self._angle_min) / self._angle_inc
                        )
                        if 0 <= idx < self._num_beams:
                            if range_t < merged[idx]:
                                merged[idx] = range_t
                            count += 1

                angle += scan.angle_increment

            per_source_counts[name] = count

            if self._diag_counter % self._diag_interval == 0:
                yaw_deg = math.degrees(yaw)
                self.get_logger().info(
                    f'[{name}] frame={scan.header.frame_id} '
                    f'yaw={yaw_deg:.1f}deg tx={tx:.3f} ty={ty:.3f} '
                    f'valid_beams={count}/{len(scan.ranges)}'
                )

        total_valid = sum(per_source_counts.values())

        if self._diag_counter % self._diag_interval == 0:
            self.get_logger().info(
                f'Merged scan: {total_valid} valid beams out of '
                f'{self._num_beams} bins'
            )

        self._diag_counter += 1

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._target_frame
        msg.angle_min = self._angle_min
        msg.angle_max = self._angle_max
        msg.angle_increment = self._angle_inc
        msg.range_min = self._range_min
        msg.range_max = self._range_max
        msg.ranges = [
            r if r != float('inf') else float('nan') for r in merged
        ]
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
