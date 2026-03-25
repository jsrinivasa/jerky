#!/usr/bin/env python3

"""
Autonomous Frontier Exploration

Detects frontiers (boundaries between free and unknown space) on the live
RTAB-Map occupancy grid and drives the robot toward them, building a
complete map without manual teleoperation.

The node is camera-FOV-aware: it prefers frontiers roughly in front of the
robot so the single forward-facing depth camera actually observes new area
when it arrives.  After reaching each frontier it does a full rotation so
the camera sweeps the surroundings before picking the next target.

Usage (after launching autonomous_mapping.launch.py):
    ros2 run aloha auto_explore
"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped for TF transforms

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry


class ExplorationState:
    IDLE = 'idle'
    NAVIGATING = 'navigating'
    ROTATING = 'rotating'
    RETURNING = 'returning'
    DONE = 'done'


class AutoExplore(Node):

    def __init__(self):
        super().__init__('auto_explore')

        # ---- Parameters ----
        self.declare_parameter('min_frontier_size', 0.75)
        self.declare_parameter('exploration_rate', 0.2)
        self.declare_parameter('goal_timeout', 30.0)
        self.declare_parameter('return_to_start', True)
        self.declare_parameter('fov_bonus_weight', 1.5)
        self.declare_parameter('rotation_speed', 0.4)
        self.declare_parameter('goal_tolerance', 0.40)
        self.declare_parameter('blacklist_radius', 1.0)
        self.declare_parameter('consecutive_fail_limit', 3)

        self._min_frontier_m = self.get_parameter('min_frontier_size').value
        self._rate = self.get_parameter('exploration_rate').value
        self._goal_timeout = self.get_parameter('goal_timeout').value
        self._return_to_start = self.get_parameter('return_to_start').value
        self._fov_weight = self.get_parameter('fov_bonus_weight').value
        self._rotation_speed = self.get_parameter('rotation_speed').value
        self._goal_tol = self.get_parameter('goal_tolerance').value
        self._blacklist_radius = self.get_parameter('blacklist_radius').value
        self._consec_fail_limit = self.get_parameter('consecutive_fail_limit').value

        # ---- State ----
        self._state = ExplorationState.IDLE
        self._map: Optional[OccupancyGrid] = None
        self._robot_pose = None  # geometry_msgs/Pose in map frame
        self._start_pose = None
        self._current_goal: Optional[Tuple[float, float]] = None
        self._goal_sent_time: Optional[float] = None
        self._blacklist: List[Tuple[float, float]] = []
        self._consecutive_failures = 0
        self._rotate_start_yaw: Optional[float] = None

        # ---- TF ----
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ---- Subscribers ----
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, map_qos
        )
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10
        )

        # ---- Publishers ----
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ---- Timer ----
        period = 1.0 / max(self._rate, 0.01)
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            'AutoExplore started — waiting for map and robot pose...'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _map_cb(self, msg: OccupancyGrid):
        self._map = msg

    def _odom_cb(self, msg: Odometry):
        try:
            ps = PoseStamped()
            ps.header.frame_id = msg.header.frame_id
            ps.header.stamp = rclpy.time.Time().to_msg()
            ps.pose = msg.pose.pose
            transformed = self._tf_buffer.transform(
                ps, 'map', timeout=Duration(seconds=0.2)
            )
            self._robot_pose = transformed.pose
            if self._start_pose is None:
                self._start_pose = transformed.pose
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _tick(self):
        if self._map is None or self._robot_pose is None:
            return

        if self._state == ExplorationState.DONE:
            return

        # --- ROTATING: 360-deg sweep after reaching a frontier ---
        if self._state == ExplorationState.ROTATING:
            self._do_rotation()
            return

        # --- RETURNING to start ---
        if self._state == ExplorationState.RETURNING:
            if self._reached_point(
                self._start_pose.position.x, self._start_pose.position.y
            ):
                self._stop()
                self._state = ExplorationState.DONE
                self.get_logger().info('Returned to start — exploration complete.')
            elif self._goal_timed_out():
                self._stop()
                self._state = ExplorationState.DONE
                self.get_logger().warn(
                    'Timed out returning to start — stopping.'
                )
            return

        # --- NAVIGATING to a frontier ---
        if self._state == ExplorationState.NAVIGATING:
            if self._current_goal and self._reached_point(*self._current_goal):
                self.get_logger().info('Frontier reached — rotating to scan.')
                self._begin_rotation()
                return

            if self._goal_timed_out():
                self.get_logger().warn('Goal timed out — blacklisting frontier.')
                if self._current_goal:
                    self._blacklist.append(self._current_goal)
                self._consecutive_failures += 1
                self._state = ExplorationState.IDLE

                if self._consecutive_failures >= self._consec_fail_limit:
                    self.get_logger().warn(
                        f'{self._consecutive_failures} consecutive failures — '
                        f'doing recovery rotation.'
                    )
                    self._begin_rotation()
                    self._consecutive_failures = 0
                    return
            else:
                return

        # --- IDLE: pick the next frontier ---
        frontiers = self._detect_frontiers()
        if not frontiers:
            self.get_logger().info(
                'No frontiers found — exploration may be complete.'
            )
            if self._return_to_start and self._start_pose is not None:
                self.get_logger().info('Returning to starting position...')
                self._send_goal(
                    self._start_pose.position.x,
                    self._start_pose.position.y,
                )
                self._state = ExplorationState.RETURNING
            else:
                self._state = ExplorationState.DONE
                self.get_logger().info('Exploration complete.')
            return

        best = self._score_frontiers(frontiers)
        if best is None:
            self.get_logger().warn('All frontiers blacklisted — clearing list.')
            self._blacklist.clear()
            return

        cx, cy = best
        self.get_logger().info(
            f'Navigating to frontier at ({cx:.2f}, {cy:.2f}) — '
            f'{len(frontiers)} frontier(s) remaining'
        )
        self._send_goal(cx, cy)
        self._state = ExplorationState.NAVIGATING
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Frontier detection
    # ------------------------------------------------------------------

    def _detect_frontiers(self) -> List[Tuple[float, float, int]]:
        """Return list of (center_x, center_y, size_in_cells) for each
        frontier cluster above the minimum size threshold."""
        grid = self._map
        w, h = grid.info.width, grid.info.height
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y

        data = np.array(grid.data, dtype=np.int8).reshape((h, w))

        free_mask = (data >= 0) & (data < 50)
        unknown_mask = data == -1

        # A free cell is a frontier if any 4-connected neighbour is unknown
        shifted_up = np.zeros_like(unknown_mask)
        shifted_down = np.zeros_like(unknown_mask)
        shifted_left = np.zeros_like(unknown_mask)
        shifted_right = np.zeros_like(unknown_mask)
        shifted_up[:-1, :] = unknown_mask[1:, :]
        shifted_down[1:, :] = unknown_mask[:-1, :]
        shifted_left[:, :-1] = unknown_mask[:, 1:]
        shifted_right[:, 1:] = unknown_mask[:, :-1]
        adjacent_unknown = shifted_up | shifted_down | shifted_left | shifted_right

        frontier_mask = free_mask & adjacent_unknown

        if not np.any(frontier_mask):
            return []

        labels, num_labels = ndimage.label(frontier_mask)

        min_cells = int(self._min_frontier_m / res)
        frontiers = []

        for label_id in range(1, num_labels + 1):
            ys, xs = np.where(labels == label_id)
            size = len(xs)
            if size < min_cells:
                continue

            mean_gx = float(np.mean(xs))
            mean_gy = float(np.mean(ys))
            cx = ox + (mean_gx + 0.5) * res
            cy = oy + (mean_gy + 0.5) * res
            frontiers.append((cx, cy, size))

        return frontiers

    # ------------------------------------------------------------------
    # Frontier scoring
    # ------------------------------------------------------------------

    def _score_frontiers(
        self, frontiers: List[Tuple[float, float, int]]
    ) -> Optional[Tuple[float, float]]:
        """Pick the best frontier considering size, distance, camera FOV
        alignment, and the blacklist."""
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        yaw = self._yaw()

        best_score = -float('inf')
        best_center = None

        for cx, cy, size in frontiers:
            if self._is_blacklisted(cx, cy):
                continue

            dist = math.hypot(cx - rx, cy - ry)
            if dist < 0.3:
                continue

            angle_to = math.atan2(cy - ry, cx - rx)
            angle_diff = abs(self._normalize(angle_to - yaw))

            # FOV bonus: full bonus within ±60 deg, linear decay to 0 at 180 deg
            fov_bonus = max(0.0, 1.0 - angle_diff / math.pi)

            score = (
                0.4 * math.log1p(size)
                - 0.3 * dist
                + self._fov_weight * fov_bonus
            )

            if score > best_score:
                best_score = score
                best_center = (cx, cy)

        return best_center

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _begin_rotation(self):
        self._state = ExplorationState.ROTATING
        self._rotate_start_yaw = self._yaw()
        self._rotate_accumulated = 0.0
        self._rotate_last_yaw = self._rotate_start_yaw

    def _do_rotation(self):
        """Rotate ~360 deg in place then return to IDLE."""
        current_yaw = self._yaw()
        delta = self._normalize(current_yaw - self._rotate_last_yaw)
        self._rotate_accumulated += abs(delta)
        self._rotate_last_yaw = current_yaw

        if self._rotate_accumulated >= 2.0 * math.pi - 0.15:
            self._stop()
            self._state = ExplorationState.IDLE
            self.get_logger().info('Rotation complete — picking next frontier.')
            return

        cmd = Twist()
        cmd.angular.z = self._rotation_speed
        self._cmd_pub.publish(cmd)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_goal(self, x: float, y: float):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)
        self._current_goal = (x, y)
        self._goal_sent_time = time.monotonic()

    def _reached_point(self, x: float, y: float) -> bool:
        if self._robot_pose is None:
            return False
        dx = self._robot_pose.position.x - x
        dy = self._robot_pose.position.y - y
        return math.hypot(dx, dy) < self._goal_tol

    def _goal_timed_out(self) -> bool:
        if self._goal_sent_time is None:
            return False
        return (time.monotonic() - self._goal_sent_time) > self._goal_timeout

    def _is_blacklisted(self, x: float, y: float) -> bool:
        for bx, by in self._blacklist:
            if math.hypot(x - bx, y - by) < self._blacklist_radius:
                return True
        return False

    def _yaw(self) -> float:
        q = self._robot_pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _stop(self):
        self._cmd_pub.publish(Twist())

    @staticmethod
    def _normalize(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = AutoExplore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
