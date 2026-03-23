#!/usr/bin/env python3
"""
Publishes the base_link -> camera_link static transform using the depth image
header stamp, so the transform exists at exactly the timestamps used by the scan
(depthimage_to_laserscan copies the depth stamp to the scan). This avoids
"timestamp earlier than all the data in the transform cache" in slam_toolbox
and RViz.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class CameraStaticTFPublisher(Node):
    def __init__(self):
        super().__init__('camera_static_tf_publisher')

        self.declare_parameter('x', 0.2098050)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 1.031778)
        self.declare_parameter('yaw', 1.5708)
        self.declare_parameter('pitch', 0.0)
        self.declare_parameter('roll', 0.0)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('child_frame_id', 'camera_link')
        self.declare_parameter('depth_topic', '/cam_high/camera/depth/image_rect_raw')

        self.tf_broadcaster = TransformBroadcaster(self)
        depth_topic = self.get_parameter('depth_topic').value
        self._depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10,
        )
        # Fallback: publish at 10 Hz with stamp=now() so TF exists before first depth
        self._fallback_timer = self.create_timer(0.1, self.fallback_publish)
        self.get_logger().info(
            f'Publishing {self.get_parameter("frame_id").value} -> '
            f'{self.get_parameter("child_frame_id").value} using depth stamp from {depth_topic}'
        )

    def depth_callback(self, msg: Image):
        """Publish base_link -> camera_link with the depth image's stamp so TF is valid at scan time."""
        x = self.get_parameter('x').value
        y = self.get_parameter('y').value
        z = self.get_parameter('z').value
        yaw = self.get_parameter('yaw').value
        pitch = self.get_parameter('pitch').value
        roll = self.get_parameter('roll').value
        frame_id = self.get_parameter('frame_id').value
        child_frame_id = self.get_parameter('child_frame_id').value

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        q = euler_to_quaternion(roll, pitch, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def fallback_publish(self):
        """Publish with stamp=now() so buffer has a transform before depth arrives."""
        x = self.get_parameter('x').value
        y = self.get_parameter('y').value
        z = self.get_parameter('z').value
        yaw = self.get_parameter('yaw').value
        pitch = self.get_parameter('pitch').value
        roll = self.get_parameter('roll').value
        frame_id = self.get_parameter('frame_id').value
        child_frame_id = self.get_parameter('child_frame_id').value
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        q = euler_to_quaternion(roll, pitch, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def main(args=None):
    rclpy.init(args=args)
    node = CameraStaticTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
