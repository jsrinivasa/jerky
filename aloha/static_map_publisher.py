#!/usr/bin/env python3

"""
Static Map Publisher

Reads a ROS-style map YAML + PGM and publishes it as a nav_msgs/OccupancyGrid
on /map with TRANSIENT_LOCAL (latched) QoS.  Any node that subscribes after
startup still receives the map immediately.

No lifecycle management needed — the map is published as soon as the node
starts and re-published every `republish_interval` seconds so any late
subscriber (e.g. RViz opened after launch) always picks it up.

Usage (replace nav2_map_server in navigate_mission.launch.py):
    ros2 run aloha static_map_publisher \
        --ros-args -p map_file:=/home/aloha/maps/mymap.yaml

Parameters
----------
map_file          : str   path to the .yaml map descriptor
republish_interval: float seconds between re-publishes (default 5.0)
                          set to 0 to publish once only
"""

import os
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

try:
    import yaml
except ImportError:
    raise ImportError('PyYAML is required: pip install pyyaml')

try:
    from PIL import Image
    import numpy as np
except ImportError:
    raise ImportError('Pillow and numpy are required: pip install pillow numpy')


TRANSIENT_LOCAL_QOS = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def _load_map(yaml_path: str) -> OccupancyGrid:
    """Parse a nav2-style map YAML + PGM into an OccupancyGrid message."""
    yaml_path = os.path.expanduser(yaml_path)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f'Map YAML not found: {yaml_path}')

    with open(yaml_path, 'r') as fh:
        meta = yaml.safe_load(fh)

    # Resolve image path relative to YAML location
    image_field = meta.get('image', '')
    if not os.path.isabs(image_field):
        image_field = os.path.join(os.path.dirname(yaml_path), image_field)
    image_field = os.path.expanduser(image_field)

    if not os.path.isfile(image_field):
        raise FileNotFoundError(f'Map image not found: {image_field}')

    resolution   = float(meta.get('resolution', 0.05))
    origin       = meta.get('origin', [0.0, 0.0, 0.0])
    negate       = int(meta.get('negate', 0))
    occ_thresh   = float(meta.get('occupied_thresh', 0.65))
    free_thresh  = float(meta.get('free_thresh', 0.25))
    mode         = meta.get('mode', 'trinary')

    img = Image.open(image_field)
    # PGM images may be P or L mode; convert to grayscale
    img = img.convert('L')
    pixels = np.array(img, dtype=np.uint8)

    # ROS maps are stored with (0,0) at bottom-left; images have (0,0) top-left
    pixels = np.flipud(pixels)

    height, width = pixels.shape

    # Convert pixel values to OccupancyGrid values
    # Pixel 255 = white = free, 0 = black = occupied, 205 = unknown (default)
    data = np.full(height * width, -1, dtype=np.int8)  # default: unknown

    # Normalise to [0, 1] probability of being occupied
    if negate:
        prob = pixels.astype(float) / 255.0
    else:
        prob = 1.0 - pixels.astype(float) / 255.0

    flat_prob = prob.flatten()

    if mode == 'raw':
        # raw mode: pixel value IS the occupancy value (0-100)
        data = np.clip(pixels.flatten().astype(np.int8), 0, 100)
    else:
        # trinary / scale mode
        free_mask = flat_prob < free_thresh
        occ_mask  = flat_prob > occ_thresh
        data[free_mask] = 0
        data[occ_mask]  = 100
        # unknown stays -1

    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = resolution
    msg.info.width      = width
    msg.info.height     = height
    msg.info.origin.position.x = float(origin[0])
    msg.info.origin.position.y = float(origin[1])
    msg.info.origin.position.z = float(origin[2]) if len(origin) > 2 else 0.0

    # origin yaw encoded as quaternion (origin[2] in YAML is z-position, not yaw;
    # yaw is always 0 in standard nav2 maps)
    msg.info.origin.orientation.w = 1.0

    msg.data = data.tolist()
    return msg


class StaticMapPublisher(Node):

    def __init__(self):
        super().__init__('static_map_publisher')

        self.declare_parameter('map_file', '')
        self.declare_parameter('republish_interval', 5.0)

        map_file = self.get_parameter('map_file').value
        republish_interval = self.get_parameter('republish_interval').value

        if not map_file:
            self.get_logger().fatal(
                'Parameter map_file is required.  Pass it with:\n'
                '  --ros-args -p map_file:=/path/to/map.yaml'
            )
            raise SystemExit(1)

        self.get_logger().info(f'Loading map from: {map_file}')
        try:
            self._map_msg = _load_map(map_file)
        except (FileNotFoundError, Exception) as exc:
            self.get_logger().fatal(f'Failed to load map: {exc}')
            raise SystemExit(1)

        w = self._map_msg.info.width
        h = self._map_msg.info.height
        res = self._map_msg.info.resolution
        self.get_logger().info(
            f'Map loaded: {w}x{h} px  resolution={res}m/px  '
            f'({w*res:.1f}x{h*res:.1f} m)'
        )

        self._pub = self.create_publisher(OccupancyGrid, '/map', TRANSIENT_LOCAL_QOS)

        self._publish()

        if republish_interval > 0:
            self.create_timer(republish_interval, self._publish)

    def _publish(self):
        self._map_msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._map_msg)
        self.get_logger().info('Map published on /map', once=True)


def main(args=None):
    rclpy.init(args=args)
    node = StaticMapPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
