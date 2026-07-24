#!/usr/bin/env python3
"""
Depth-image border mask.

Blanks a configurable fraction of columns at the left and right edges of a
depth image, then republishes it -- so depthimage_to_laserscan (and
anything downstream of it: /scan, laser_scan_merger, RTAB-Map's grid,
obstacle avoidance) doesn't treat the robot's own arm/hand -- which tends
to sit near the image edges when it's in frame at all -- as a real
external obstacle.

Blanked columns are set to each encoding's own "no valid reading" value
(0 for 16UC1, NaN for 32FC1) -- the same convention depth sensors already
use for holes/low-confidence pixels, so depthimage_to_laserscan already
drops them with no config change beyond remapping its input topic to this
node's output.

Only touches whatever topic is configured as input_topic (normally the
depthimage_to_laserscan-facing .../depth/image_rect_raw stream) -- NOT the
separate aligned_depth_to_color stream rgbd_sync/RTAB-Map use for visual
registration/loop-closure, so that's unaffected unless explicitly pointed
here too.

No cv_bridge dependency (not a declared package dependency here) -- same
manual, step-aware numpy decode/re-encode pattern nav_web_viewer.py uses
for camera JPEGs.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

_DTYPE_BY_ENCODING = {
    '16UC1': (np.uint16, 0),
    '32FC1': (np.float32, float('nan')),
}


class DepthBorderMask(Node):

    def __init__(self):
        super().__init__('depth_border_mask')
        self.declare_parameter('input_topic', '/cam_high/camera/depth/image_rect_raw')
        self.declare_parameter('output_topic', '/cam_high/camera/depth/image_rect_raw_masked')
        self.declare_parameter('border_fraction', 0.05)

        self._border_fraction = float(self.get_parameter('border_fraction').value)
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self._pub = self.create_publisher(Image, output_topic, qos_profile_sensor_data)
        self.create_subscription(Image, input_topic, self._callback, qos_profile_sensor_data)
        self.get_logger().info(
            f'Masking outer {self._border_fraction:.0%} of columns on each '
            f'side: {input_topic} -> {output_topic}')

    def _callback(self, msg: Image):
        dtype_invalid = _DTYPE_BY_ENCODING.get(msg.encoding)
        if dtype_invalid is None:
            self.get_logger().warn(
                f"unsupported depth encoding '{msg.encoding}' -- passing "
                'through unmasked', throttle_duration_sec=30.0)
            self._pub.publish(msg)
            return
        dtype, invalid = dtype_invalid

        border = max(1, int(round(msg.width * self._border_fraction)))
        row_stride = msg.step // dtype().itemsize
        # step-aware reshape (row_stride, not width) handles row padding
        # correctly; .copy() because frombuffer's array is read-only.
        arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_stride).copy()
        arr[:, :border] = invalid
        arr[:, msg.width - border:msg.width] = invalid

        out = Image()
        out.header = msg.header
        out.height = msg.height
        out.width = msg.width
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = msg.step
        out.data = arr.tobytes()
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DepthBorderMask()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
