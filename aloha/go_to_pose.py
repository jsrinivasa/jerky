#!/usr/bin/env python3

"""
Go To Named Pose

Localizes the robot (using RTAB-Map loop closures) then navigates to any
named location defined in <package_share>/config/named_poses.yaml.

Prerequisites (must already be running):
  ros2 launch aloha navigate_mission.launch.py map_file:=<your_map>.yaml

Usage:
  ros2 run aloha go_to_pose --ros-args -p pose_name:=elevator
  ros2 run aloha go_to_pose --ros-args -p pose_name:=conference_room
  ros2 run aloha go_to_pose --ros-args -p pose_name:=team_room_31
  ros2 run aloha go_to_pose --ros-args -p pose_name:=parking

List all available poses:
  ros2 run aloha go_to_pose --ros-args -p pose_name:=list

Optional parameters:
  -p rotation_speed:=0.3
  -p timeout_before_rotate:=3.0
  -p max_localization_time:=90.0
  -p localization_match_threshold:=2
  -p goal_tolerance:=0.35
  -p waypoint_timeout:=120.0
  -p poses_file:=/absolute/path/to/named_poses.yaml
"""

import math
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import Info
import tf2_ros
import tf2_geometry_msgs  # noqa: F401

try:
    import yaml
except ImportError:
    raise ImportError('PyYAML is required: pip install pyyaml')

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


def _default_poses_file() -> str:
    if get_package_share_directory is not None:
        try:
            share = get_package_share_directory('aloha')
            candidate = os.path.join(share, 'config', 'named_poses.yaml')
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
    src_candidate = Path(__file__).parent.parent / 'config' / 'named_poses.yaml'
    return str(src_candidate)


def _load_all_poses(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"named_poses.yaml not found at '{path}'.")
    with open(path, 'r') as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"named_poses.yaml is empty or malformed: {path}")
    return data


def _parse_pose(data: dict, name: str) -> tuple[float, float, float]:
    if name not in data:
        available = ', '.join(data.keys())
        raise KeyError(
            f"Pose '{name}' not found in named_poses.yaml.\n"
            f"Available poses: {available}"
        )
    p = data[name]
    try:
        x = float(p['x'])
        y = float(p['y'])
        yaw = float(p['yaw'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pose '{name}': {exc}") from exc

    if x == 0.0 and y == 0.0 and yaw == 0.0:
        import warnings
        warnings.warn(
            f"Pose '{name}' is still at (0, 0, 0) — the placeholder value.\n"
            "Drive to the location and record coordinates with:\n"
            "  ros2 run tf2_ros tf2_echo map base_link",
            stacklevel=2,
        )
    return x, y, yaw


class GoToPose(Node):

    def __init__(self):
        super().__init__('go_to_pose')

        self.declare_parameter('pose_name', '')
        self.declare_parameter('rotation_speed', 0.3)
        self.declare_parameter('timeout_before_rotate', 3.0)
        self.declare_parameter('max_localization_time', 90.0)
        self.declare_parameter('localization_match_threshold', 2)
        self.declare_parameter('goal_tolerance', 0.35)
        self.declare_parameter('waypoint_timeout', 120.0)
        self.declare_parameter('poses_file', _default_poses_file())

        pose_name   = self.get_parameter('pose_name').value
        poses_file  = self.get_parameter('poses_file').value

        self._rotation_speed = self.get_parameter('rotation_speed').value
        self._loc_timeout    = self.get_parameter('timeout_before_rotate').value
        self._loc_max_time   = self.get_parameter('max_localization_time').value
        self._match_thresh   = self.get_parameter('localization_match_threshold').value
        self._goal_tol       = self.get_parameter('goal_tolerance').value
        self._wp_timeout     = self.get_parameter('waypoint_timeout').value

        try:
            all_poses = _load_all_poses(poses_file)
        except FileNotFoundError as exc:
            self.get_logger().fatal(str(exc))
            raise SystemExit(1)

        # Special command: list available poses and exit
        if pose_name in ('list', ''):
            names = list(all_poses.keys())
            self.get_logger().info(
                f'Available poses in {poses_file}:\n' +
                '\n'.join(f'  • {n}' for n in names) +
                '\n\nUsage:  ros2 run aloha go_to_pose --ros-args -p pose_name:=<name>'
            )
            raise SystemExit(0)

        try:
            self._goal_x, self._goal_y, self._goal_yaw = _parse_pose(all_poses, pose_name)
        except (KeyError, ValueError) as exc:
            self.get_logger().fatal(str(exc))
            raise SystemExit(1)

        self._pose_name = pose_name
        self.get_logger().info(
            f"Navigating to '{pose_name}': "
            f"x={self._goal_x:.2f}  y={self._goal_y:.2f}  "
            f"yaw={math.degrees(self._goal_yaw):.1f}°"
        )

        # ---- State ---------------------------------------------------------
        self._phase          = 'LOCALIZE'
        self._closure_count  = 0
        self._start_time     = self.get_clock().now()
        self._rotating       = False
        self._goal_sent      = False
        self._goal_start_time    = None
        self._goal_last_publish  = None
        self._goal_accepted      = False
        self._goal_initial_dist  = None
        self._current_pose       = None

        # ---- TF ------------------------------------------------------------
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ---- Pubs / Subs ---------------------------------------------------
        self._cmd_pub  = self.create_publisher(Twist, '/mobile_base/cmd_vel', 10)
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.create_subscription(Info, '/rtabmap/info', self._info_cb, 10)
        self.create_subscription(Odometry, '/mobile_base/odom', self._odom_cb, 10)
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f'Will rotate after {self._loc_timeout:.0f}s without a visual match. '
            'Localizing...'
        )

    # ================================================================
    # Callbacks
    # ================================================================

    def _info_cb(self, msg: Info):
        if self._phase != 'LOCALIZE':
            return
        if msg.loop_closure_id > 0 or msg.proximity_detection_id > 0:
            self._closure_count += 1
            source = (
                f'loop closure (node {msg.loop_closure_id})'
                if msg.loop_closure_id > 0
                else f'proximity (node {msg.proximity_detection_id})'
            )
            self.get_logger().info(f'Visual match #{self._closure_count}: {source}')
            if self._closure_count >= self._match_thresh:
                self._finish_localization()

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

    # ================================================================
    # Tick
    # ================================================================

    def _tick(self):
        if self._phase == 'LOCALIZE':
            self._localize_tick()
        elif self._phase == 'NAVIGATE':
            self._navigate_tick()

    # ================================================================
    # Phase 1 -- Localize
    # ================================================================

    def _localize_tick(self):
        elapsed = self._elapsed(self._start_time)
        if elapsed > self._loc_max_time:
            self._stop_robot()
            self.get_logger().error(
                f'Failed to localize after {self._loc_max_time:.0f}s.'
            )
            raise SystemExit(1)
        if elapsed > self._loc_timeout and not self._rotating:
            self._rotating = True
            self.get_logger().info('No match yet — rotating to find landmarks...')
        if self._rotating:
            twist = Twist()
            twist.angular.z = self._rotation_speed
            self._cmd_pub.publish(twist)

    def _finish_localization(self):
        self._stop_robot()
        elapsed = self._elapsed(self._start_time)
        self.get_logger().info(
            '========================================\n'
            f'  LOCALIZED in {elapsed:.1f}s  '
            f'({self._closure_count} visual matches)\n'
            '========================================\n'
            f"Navigating to '{self._pose_name}': "
            f'({self._goal_x:.2f}, {self._goal_y:.2f}, '
            f'yaw={math.degrees(self._goal_yaw):.1f}°)...'
        )
        self._phase = 'NAVIGATE'

    # ================================================================
    # Phase 2 -- Navigate
    # ================================================================

    def _navigate_tick(self):
        if not self._goal_sent:
            self._send_goal()
            return

        if not self._goal_accepted and self._elapsed(self._goal_last_publish) > 3.0:
            if self._elapsed(self._goal_start_time) < 15.0:
                self._republish_goal()
            else:
                self._goal_accepted = True
                self.get_logger().info('Assuming planner accepted goal (timeout-based).')

        if self._current_pose is None:
            return

        dx = self._current_pose.position.x - self._goal_x
        dy = self._current_pose.position.y - self._goal_y
        dist = math.sqrt(dx * dx + dy * dy)

        if not self._goal_accepted:
            if self._goal_initial_dist is None:
                self._goal_initial_dist = dist
            elif dist < self._goal_initial_dist - 0.05:
                self._goal_accepted = True
                self.get_logger().info(
                    f'Planner accepted goal (robot moving, dist={dist:.2f}m).'
                )

        if dist < self._goal_tol:
            self._stop_robot()
            self.get_logger().info(
                '========================================\n'
                f"  ARRIVED at '{self._pose_name}'  (dist={dist:.2f}m)\n"
                '========================================'
            )
            raise SystemExit(0)

        if self._elapsed(self._goal_start_time) > self._wp_timeout:
            self._stop_robot()
            self.get_logger().error(
                f"Failed to reach '{self._pose_name}' after "
                f'{self._wp_timeout:.0f}s (dist={dist:.2f}m).'
            )
            raise SystemExit(1)

    def _build_goal(self) -> PoseStamped:
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = self._goal_x
        goal.pose.position.y = self._goal_y
        goal.pose.orientation.z = math.sin(self._goal_yaw / 2.0)
        goal.pose.orientation.w = math.cos(self._goal_yaw / 2.0)
        return goal

    def _send_goal(self):
        self._goal_pub.publish(self._build_goal())
        self._goal_sent = True
        self._goal_start_time   = self.get_clock().now()
        self._goal_last_publish = self.get_clock().now()
        self.get_logger().info(
            f"Goal sent: '{self._pose_name}' at "
            f'({self._goal_x:.2f}, {self._goal_y:.2f}, '
            f'yaw={math.degrees(self._goal_yaw):.1f}°)'
        )

    def _republish_goal(self):
        self._goal_pub.publish(self._build_goal())
        self._goal_last_publish = self.get_clock().now()
        self.get_logger().info(
            f"Re-publishing goal '{self._pose_name}' "
            '(planner may not have been ready)...'
        )

    # ================================================================
    # Helpers
    # ================================================================

    def _stop_robot(self):
        self._rotating = False
        self._cmd_pub.publish(Twist())

    def _elapsed(self, stamp) -> float:
        return (self.get_clock().now() - stamp).nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = GoToPose()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node._stop_robot()
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
