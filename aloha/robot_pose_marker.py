#!/usr/bin/env python3
"""
Robot Pose Marker Publisher

Publishes highly visible RViz markers for the robot's current position and
heading based on the map → base_link TF transform.  Markers include a
to-scale rectangular body footprint (22in x 28in) and a bold arrow
(heading vector).
"""

import math

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf2_ros


class RobotPoseMarker(Node):

    def __init__(self):
        super().__init__('robot_pose_marker')

        # Measured robot footprint: ~22in wide x 28in long.
        self.declare_parameter('robot_length', 0.7112)  # 28in, along forward (x)
        self.declare_parameter('robot_width', 0.5588)   # 22in, lateral (y)
        self.declare_parameter('arrow_length', 0.6)
        self.declare_parameter('publish_rate', 10.0)

        self._robot_length = self.get_parameter('robot_length').value
        self._robot_width = self.get_parameter('robot_width').value
        self._arrow_length = self.get_parameter('arrow_length').value
        rate = self.get_parameter('publish_rate').value

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._pub = self.create_publisher(MarkerArray, '/robot_pose_markers', 10)
        self._tf_was_missing = False
        self.create_timer(1.0 / rate, self._publish_markers)

    def _publish_markers(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            # 2026-07-21: used to just return here, leaving whatever was
            # published last on screen -- these markers use lifetime=0
            # (never auto-expire in RViz), so a TF hiccup made the footprint
            # silently freeze in place with zero indication it had gone
            # stale (looked identical to a real, current pose). Explicitly
            # clear it instead, so "no marker" honestly means "no current
            # pose" rather than "last known pose, possibly minutes old."
            if not self._tf_was_missing:
                clear = MarkerArray()
                for ns in ('robot_body', 'robot_heading'):
                    d = Marker()
                    d.header.frame_id = 'map'
                    d.ns = ns
                    d.action = Marker.DELETEALL
                    clear.markers.append(d)
                self._pub.publish(clear)
                self._tf_was_missing = True
            return
        self._tf_was_missing = False

        stamp = self.get_clock().now().to_msg()
        px = tf.transform.translation.x
        py = tf.transform.translation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

        markers = MarkerArray()

        # Rectangle markers need the robot's actual heading (a circle didn't
        # care about orientation, but a to-scale footprint box does).
        body_orientation_z = math.sin(yaw / 2.0)
        body_orientation_w = math.cos(yaw / 2.0)

        # --- To-scale rectangular footprint (22in x 28in) for the robot body ---
        body = Marker()
        body.header.frame_id = 'map'
        body.header.stamp = stamp
        body.ns = 'robot_body'
        body.id = 0
        body.type = Marker.CUBE
        body.action = Marker.ADD
        body.pose.position.x = px
        body.pose.position.y = py
        body.pose.position.z = 0.02
        body.pose.orientation.z = body_orientation_z
        body.pose.orientation.w = body_orientation_w
        body.scale.x = self._robot_length
        body.scale.y = self._robot_width
        body.scale.z = 0.04
        body.color = ColorRGBA(r=0.0, g=1.0, b=0.3, a=0.85)
        body.lifetime.sec = 0
        markers.markers.append(body)

        # --- Bright outline around the footprint ---
        ring = Marker()
        ring.header.frame_id = 'map'
        ring.header.stamp = stamp
        ring.ns = 'robot_body'
        ring.id = 1
        ring.type = Marker.CUBE
        ring.action = Marker.ADD
        ring.pose.position.x = px
        ring.pose.position.y = py
        ring.pose.position.z = 0.03
        ring.pose.orientation.z = body_orientation_z
        ring.pose.orientation.w = body_orientation_w
        ring.scale.x = self._robot_length + 0.08
        ring.scale.y = self._robot_width + 0.08
        ring.scale.z = 0.02
        ring.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        ring.lifetime.sec = 0
        markers.markers.append(ring)

        # --- Heading arrow ---
        arrow = Marker()
        arrow.header.frame_id = 'map'
        arrow.header.stamp = stamp
        arrow.ns = 'robot_heading'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        tail = Point()
        tail.x = px
        tail.y = py
        tail.z = 0.06

        tip = Point()
        length = self._arrow_length
        tip.x = px + length * math.cos(yaw)
        tip.y = py + length * math.sin(yaw)
        tip.z = 0.06

        arrow.points = [tail, tip]
        arrow.scale.x = 0.12          # shaft diameter
        arrow.scale.y = 0.22          # head diameter
        arrow.scale.z = 0.18          # head length
        arrow.color = ColorRGBA(r=1.0, g=0.2, b=0.0, a=1.0)
        arrow.lifetime.sec = 0
        markers.markers.append(arrow)

        # --- Small centre dot ---
        dot = Marker()
        dot.header.frame_id = 'map'
        dot.header.stamp = stamp
        dot.ns = 'robot_body'
        dot.id = 2
        dot.type = Marker.CYLINDER
        dot.action = Marker.ADD
        dot.pose.position.x = px
        dot.pose.position.y = py
        dot.pose.position.z = 0.04
        dot.pose.orientation.w = 1.0
        dot.scale.x = 0.08
        dot.scale.y = 0.08
        dot.scale.z = 0.02
        dot.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
        dot.lifetime.sec = 0
        markers.markers.append(dot)

        self._pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = RobotPoseMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
