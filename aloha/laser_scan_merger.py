#!/usr/bin/env python3

"""
Laser Scan Merger

Merges any number of LaserScan topics (configured via the `input_topics`
parameter -- e.g. an RPLIDAR plus a depthimage_to_laserscan-derived scan
from a depth camera) into a single scan in the base_link frame.  Each
incoming beam is projected into base_link using the TF tree, binned by
angle, and the nearest range per bin wins across all sources.

The output scan covers the full -pi..pi range so downstream consumers can
correctly interpret the beam angles, but diagnostics are logged
periodically to help verify each source is contributing valid data.

`self_occlusion_ranges` optionally blanks out known angle sectors (in the
target_frame, radians, flat list of [min1, max1, min2, max2, ...] pairs)
where the robot's own chassis sits in a sensor's field of view -- e.g. a
lidar mounted near the front will usually see part of the body behind it.
Points in these sectors are dropped before merging, from every source.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
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
        self.declare_parameter(
            'input_topics', ['/scan_rplidar', '/scan_depth_cam_high'])
        self.declare_parameter(
            'self_occlusion_ranges', [],
            ParameterDescriptor(dynamic_typing=True),
        )  # e.g. [2.6, 3.14, -3.14, -2.6]

        self._target_frame = self.get_parameter('target_frame').value
        self._angle_min = self.get_parameter('angle_min').value
        self._angle_max = self.get_parameter('angle_max').value
        self._angle_inc = self.get_parameter('angle_increment').value
        self._range_min = self.get_parameter('range_min').value
        self._range_max = self.get_parameter('range_max').value
        publish_rate = self.get_parameter('publish_rate').value
        input_topics = self.get_parameter('input_topics').value
        occlusion_flat = list(self.get_parameter('self_occlusion_ranges').value)
        if len(occlusion_flat) % 2 != 0:
            raise ValueError(
                'self_occlusion_ranges must be an even-length flat list of '
                '[min1, max1, min2, max2, ...] pairs'
            )
        self._occlusion_ranges = list(
            zip(occlusion_flat[0::2], occlusion_flat[1::2])
        )

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

        for topic in input_topics:
            self.create_subscription(
                LaserScan, topic, self._make_cb(topic), qos)

        self._pub = self.create_publisher(LaserScan, '/scan', 10)
        self.create_timer(1.0 / publish_rate, self._merge_and_publish)

        self.get_logger().info(
            f'Merging {input_topics} -> /scan '
            f'({self._num_beams} bins, frame={self._target_frame}'
            + (f', {len(self._occlusion_ranges)} self-occlusion sector(s)'
               if self._occlusion_ranges else '')
            + ')'
        )

    def _make_cb(self, key):
        def cb(msg):
            self._latest_scans[key] = msg
        return cb

    def _in_occlusion_sector(self, angle: float) -> bool:
        """True if `angle` (target_frame bearing, radians) falls inside a
        configured self_occlusion_ranges pair -- the chassis, not a real
        obstacle, is expected there."""
        for lo, hi in self._occlusion_ranges:
            if lo <= hi:
                if lo <= angle <= hi:
                    return True
            else:  # pair wraps across +/-pi, e.g. [3.0, -3.0]
                if angle >= lo or angle <= hi:
                    return True
        return False

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

                    if (self._range_min <= range_t <= self._range_max
                            and not self._in_occlusion_sector(angle_t)):
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
