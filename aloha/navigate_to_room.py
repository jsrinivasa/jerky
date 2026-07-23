#!/usr/bin/env python3

"""
Navigate to a Room by Name

NOT part of the current map-free flow (see docs/MAPFREE_NAV.md) -- kept as
a reference implementation. Its find_room_entrance() doorway-detection
algorithm (free->wall->free ray-cast) was adapted into that flow's own
attempt at a "stand outside the door" mode, which was tried, found
unreliable in live testing, and dropped in favor of letting
simple_nav_planner's own A* degrade gracefully instead (see MAPFREE_NAV.md's
"Search by room/desk code" and "What was tried and dropped" sections).

Also: DEFAULT_MAP_YAML/DEFAULT_ROOMS_JSON below point at files that no
longer exist on disk (superseded by floorplan_real_2_nav_rooms_corrected.json
and the _walls_inflated.{pgm,yaml} variants) -- running this as-is will fail
to load until pointed at current files via --map-yaml/--rooms-json. It also
publishes /goal_pose using room map_x/map_y AS-IS in the 'map' frame, which
is only correct under the older prebuilt-map /calibrate flow, not the
map-free anchor architecture (frame-A floorplan coordinates there are tied
to frame-B robot coordinates only through a runtime anchor, not directly).

Looks up a room by name (fuzzy match) from the rooms JSON, computes
the entrance/approach point using the occupancy grid (ray-casting to
find the doorway), and publishes a goal pose to /goal_pose.

The approach point is placed in the corridor just outside the room's
door, so the robot stops in front of the room rather than entering it.

Usage:
    # Navigate to JAWS conference room:
    ros2 run aloha navigate_to_room -- --room "jaws"

    # Navigate to room 350:
    ros2 run aloha navigate_to_room -- --room "350"

    # Override standoff distance (default 0.5m):
    ros2 run aloha navigate_to_room -- --room "jaws" --standoff 0.8

    # List all matching rooms without navigating:
    ros2 run aloha navigate_to_room -- --room "conf" --list

    # Use custom map/rooms files:
    ros2 run aloha navigate_to_room -- --room "jaws" \
        --map-yaml /path/to/map.yaml \
        --rooms-json /path/to/rooms.json
"""

import argparse
import json
import math
import os
import sys
import time
import yaml

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401


# =====================================================================
# Room entrance finder
# =====================================================================

def load_map_image(map_yaml_path: str):
    """Load the PGM image and metadata from a ROS map YAML file."""
    with open(map_yaml_path) as f:
        meta = yaml.safe_load(f)

    pgm_path = meta['image']
    if not os.path.isabs(pgm_path):
        pgm_path = os.path.join(os.path.dirname(map_yaml_path), pgm_path)

    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read map image: {pgm_path}')

    resolution = float(meta['resolution'])
    origin_x = float(meta['origin'][0])
    origin_y = float(meta['origin'][1])
    return img, resolution, origin_x, origin_y


def find_room_entrance(
    map_img: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    room_map_x: float,
    room_map_y: float,
    standoff: float = 0.5,
    wall_thresh: int = 128,
    free_thresh: int = 200,
    n_rays: int = 72,
    max_ray_px: int = 150,
):
    """
    Find the approach point (corridor side of the door) for a room.

    Algorithm:
      1. From the room center, cast rays outward every (360/n_rays) degrees.
      2. For each ray, detect the pattern: free → wall → free (a doorway).
      3. Pick the ray with the shortest total distance (nearest door).
      4. Place the approach point ``standoff`` meters past the wall into
         the corridor, facing toward the door.

    Returns:
        (approach_x, approach_y, facing_yaw) in map coordinates,
        or None if no entrance found.
    """
    h, w = map_img.shape

    # Room center in grid coords (PGM row 0 = top of image = max Y in map)
    gx = int((room_map_x - origin_x) / resolution)
    gy = h - 1 - int((room_map_y - origin_y) / resolution)

    best = None  # (total_dist_px, approach_mx, approach_my, facing_yaw)

    for i in range(n_rays):
        angle = 2.0 * math.pi * i / n_rays
        dx = math.cos(angle)
        dy = math.sin(angle)

        wall_start = None
        wall_end = None

        for step in range(1, max_ray_px):
            px = int(gx + dx * step)
            py = int(gy + dy * step)
            if not (0 <= px < w and 0 <= py < h):
                break

            pixel = map_img[py, px]

            if wall_start is None:
                if pixel < wall_thresh:
                    wall_start = step
            else:
                if pixel >= free_thresh:
                    wall_end = step
                    break

        if wall_start is not None and wall_end is not None:
            wall_thickness = wall_end - wall_start
            if wall_thickness < 20:  # reasonable wall (< ~50 cm)
                total_dist = wall_end
                if best is None or total_dist < best[0]:
                    standoff_cells = int(standoff / resolution)
                    approach_step = wall_end + standoff_cells
                    ax = gx + dx * approach_step
                    ay = gy + dy * approach_step
                    approach_mx = ax * resolution + origin_x
                    approach_my = (h - 1 - ay) * resolution + origin_y
                    # Facing yaw: point toward the door (opposite of ray direction)
                    # dy is in image coords (down=+), map coords flip Y
                    facing_yaw = math.atan2(dy, -dx) + math.pi
                    # Normalize to [-pi, pi]
                    facing_yaw = math.atan2(math.sin(facing_yaw), math.cos(facing_yaw))
                    best = (total_dist, approach_mx, approach_my, facing_yaw)

    if best is None:
        return None
    return best[1], best[2], best[3]


def fuzzy_match_rooms(rooms: list, query: str) -> list:
    """Return rooms whose label contains the query (case-insensitive)."""
    q = query.strip().upper()
    return [r for r in rooms if q in r.get('label', '').upper()]


# =====================================================================
# ROS2 node
# =====================================================================

class NavigateToRoom(Node):

    def __init__(self, room_name: str, rooms_json: str, map_yaml: str,
                 standoff: float, list_only: bool):
        super().__init__('navigate_to_room')

        # ── Load rooms ──
        with open(rooms_json) as f:
            all_rooms = json.load(f)

        matches = fuzzy_match_rooms(all_rooms, room_name)
        if not matches:
            self.get_logger().error(f'No room matching "{room_name}" found.')
            self.get_logger().info('Tip: try a shorter name, e.g. "jaws" or "350"')
            raise SystemExit(1)

        if list_only:
            self.get_logger().info(f'Found {len(matches)} rooms matching "{room_name}":')
            for r in matches:
                self.get_logger().info(
                    f'  {r.get("label",""):35s}  ({r.get("map_x",0):7.2f}, {r.get("map_y",0):7.2f})  [{r.get("category","")}]'
                )
            raise SystemExit(0)

        # Pick best match: prefer exact match, then shortest label
        exact = [r for r in matches if r['label'].upper() == room_name.strip().upper()]
        room = exact[0] if exact else sorted(matches, key=lambda r: len(r['label']))[0]

        if len(matches) > 1:
            self.get_logger().info(
                f'{len(matches)} rooms match "{room_name}". Using: {room["label"]}'
            )

        self.get_logger().info(
            f'Target room: {room["label"]}  '
            f'center=({room.get("map_x",0):.2f}, {room.get("map_y",0):.2f})'
        )

        # ── Compute entrance ──
        map_img, resolution, ox, oy = load_map_image(map_yaml)
        entrance = find_room_entrance(
            map_img, resolution, ox, oy,
            room['map_x'], room['map_y'],
            standoff=standoff,
        )

        if entrance is None:
            self.get_logger().warn(
                'Could not find entrance on map — navigating to room center instead.'
            )
            self._goal_x = room['map_x']
            self._goal_y = room['map_y']
            self._goal_yaw = 0.0
        else:
            self._goal_x, self._goal_y, self._goal_yaw = entrance
            self.get_logger().info(
                f'Entrance: ({self._goal_x:.2f}, {self._goal_y:.2f})  '
                f'facing {math.degrees(self._goal_yaw):.0f}deg  '
                f'(standoff={standoff:.1f}m from door)'
            )

        self._room_label = room['label']
        self._arrived = False

        # ── TF ──
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._current_pose = None

        # ── Pubs / Subs ──
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.create_subscription(Odometry, '/mobile_base/odom', self._odom_cb, 10)

        # Wait a moment for planner to be ready, then publish
        self._publish_count = 0
        self._goal_tolerance = 0.3
        self._timer = self.create_timer(1.0, self._tick)
        self.get_logger().info('Waiting for planner...')

    def _odom_cb(self, msg: Odometry):
        try:
            ps = PoseStamped()
            ps.header.frame_id = msg.header.frame_id
            ps.header.stamp = rclpy.time.Time().to_msg()
            ps.pose = msg.pose.pose
            transformed = self._tf_buffer.transform(
                ps, 'map', timeout=Duration(seconds=0.2)
            )
            self._current_pose = transformed.pose
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    def _publish_goal(self):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = self._goal_x
        goal.pose.position.y = self._goal_y
        goal.pose.orientation.z = math.sin(self._goal_yaw / 2.0)
        goal.pose.orientation.w = math.cos(self._goal_yaw / 2.0)
        self._goal_pub.publish(goal)

    def _tick(self):
        if self._arrived:
            return

        # Publish goal a few times to make sure planner receives it
        if self._publish_count < 5:
            self._publish_goal()
            self._publish_count += 1
            if self._publish_count == 1:
                self.get_logger().info(
                    f'Navigating to {self._room_label} → '
                    f'({self._goal_x:.2f}, {self._goal_y:.2f})'
                )

        # Check arrival
        if self._current_pose is not None:
            dx = self._current_pose.position.x - self._goal_x
            dy = self._current_pose.position.y - self._goal_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self._goal_tolerance:
                self._arrived = True
                self.get_logger().info(
                    '========================================\n'
                    f'  ARRIVED at {self._room_label}\n'
                    f'  Position: ({self._current_pose.position.x:.2f}, '
                    f'{self._current_pose.position.y:.2f})\n'
                    '========================================'
                )
                raise SystemExit(0)


# =====================================================================
# Main
# =====================================================================

DEFAULT_MAP_YAML = '/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_walls.yaml'
DEFAULT_ROOMS_JSON = '/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_rooms.json'


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Navigate the robot to a room by name.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ros2 run aloha navigate_to_room -- --room "jaws"
  ros2 run aloha navigate_to_room -- --room "350"
  ros2 run aloha navigate_to_room -- --room "crystal pier" --standoff 0.8
  ros2 run aloha navigate_to_room -- --room "conf" --list
        """,
    )
    parser.add_argument('--room', '-r', required=True,
                        help='Room name (or partial name) to navigate to')
    parser.add_argument('--standoff', '-s', type=float, default=0.5,
                        help='Distance in meters to stand back from the door (default: 0.5)')
    parser.add_argument('--map-yaml', default=DEFAULT_MAP_YAML,
                        help='Path to the map YAML file')
    parser.add_argument('--rooms-json', default=DEFAULT_ROOMS_JSON,
                        help='Path to the rooms JSON file')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List matching rooms and exit (no navigation)')

    # Parse only known args (ROS may inject extra ones)
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = NavigateToRoom(
        room_name=parsed.room,
        rooms_json=parsed.rooms_json,
        map_yaml=parsed.map_yaml,
        standoff=parsed.standoff,
        list_only=parsed.list,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
