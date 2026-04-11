#!/usr/bin/env python3
"""
Room Label Marker Publisher

Publishes RViz text markers for room labels extracted from an SVG floor
plan.  The markers are purely visual — they do NOT affect the occupancy
grid or navigation stack.

Usage (standalone):
    ros2 run aloha room_label_publisher \
        --ros-args -p rooms_json:=/path/to/rooms.json

Or include in a launch file — see navigate_mission.launch.py.
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


# Colour palette per category (RGBA, 0-1)
CATEGORY_COLORS = {
    'conference_room':    ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0),   # cyan
    'restroom':           ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0),   # orange
    'vertical_transport': ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),   # yellow
    'corridor':           ColorRGBA(r=0.6, g=1.0, b=0.6, a=0.9),   # light green
    'shared_space':       ColorRGBA(r=0.8, g=0.6, b=1.0, a=0.9),   # purple
    'utility':            ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8),   # grey
    'numbered_room':      ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.7),   # white
    'named_space':        ColorRGBA(r=0.9, g=0.9, b=0.9, a=0.5),   # dim white
}
DEFAULT_COLOR = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.6)

# Text height (metres) per category — important rooms get bigger labels
CATEGORY_SCALE = {
    'conference_room':    0.8,
    'restroom':           0.7,
    'vertical_transport': 0.6,
    'corridor':           0.6,
    'shared_space':       0.4,
    'utility':            0.4,
    'numbered_room':      0.35,
    'named_space':        0.3,
}
DEFAULT_SCALE = 0.35


class RoomLabelPublisher(Node):

    def __init__(self):
        super().__init__('room_label_publisher')

        self.declare_parameter('rooms_json', '')
        self.declare_parameter('publish_rate', 0.5)      # Hz (labels are static)
        self.declare_parameter('marker_z', 0.0)           # height above ground
        self.declare_parameter('categories', [''])         # empty = all

        rooms_json = self.get_parameter('rooms_json').value
        rate = self.get_parameter('publish_rate').value
        self._marker_z = self.get_parameter('marker_z').value
        cat_filter = self.get_parameter('categories').value

        if not rooms_json:
            self.get_logger().error(
                "Parameter 'rooms_json' is required. "
                "Set it to the path of the rooms JSON file.")
            return

        # Load rooms
        try:
            with open(rooms_json, 'r') as f:
                self._rooms = json.load(f)
        except Exception as e:
            self.get_logger().error(f"Could not load rooms JSON: {e}")
            return

        # Filter categories if specified
        if cat_filter and cat_filter != ['']:
            self._rooms = [r for r in self._rooms
                           if r['category'] in cat_filter]

        self.get_logger().info(
            f"Loaded {len(self._rooms)} room labels from {rooms_json}")

        # Use transient-local durability so RViz sees labels even if it
        # subscribes after the first publish
        latching_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._pub = self.create_publisher(
            MarkerArray, '/room_labels', latching_qos)

        # Build the MarkerArray once (labels are static)
        self._marker_array = self._build_markers()

        # Publish periodically (low rate since it's static data)
        self.create_timer(1.0 / rate, self._publish)
        # Also publish immediately
        self._publish()

    def _build_markers(self) -> MarkerArray:
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i, room in enumerate(self._rooms):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'room_labels'
            m.id = i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD

            m.pose.position.x = room['map_x']
            m.pose.position.y = room['map_y']
            m.pose.position.z = self._marker_z
            m.pose.orientation.w = 1.0

            cat = room.get('category', '')
            scale = CATEGORY_SCALE.get(cat, DEFAULT_SCALE)
            m.scale.z = scale  # text height in metres

            m.color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
            m.text = room['label']

            # Keep markers alive forever
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0

            ma.markers.append(m)

        return ma

    def _publish(self):
        self._pub.publish(self._marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = RoomLabelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
