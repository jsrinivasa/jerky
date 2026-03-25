#!/usr/bin/env python3

"""
Autonomous Navigate Mission

Phase 1 -- Localize:  Monitors /rtabmap/info for visual loop closures.
                      Slowly rotates in place if no match is found within a
                      timeout.  Confirms localization after >=2 visual matches.

Phase 2 -- Navigate:  Iterates through hardcoded (x, y, yaw) waypoints,
                      publishing each as a PoseStamped to /goal_pose for the
                      simple_nav_planner to execute.  Monitors /mobile_base/odom
                      to detect arrival (within goal_tolerance) or timeout.

Usage (after launching navigate_mission.launch.py):
    ros2 run aloha navigate_mission
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import Info
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped transform


# =====================================================================
# EDIT THESE WAYPOINTS
# Each tuple is (x, y, yaw_radians) in the *map* frame.
# Hover over the map in RViz to read coordinates.
# =====================================================================
WAYPOINTS: list[tuple[float, float, float]] = [
    # (x, y, yaw)
    (1.0, 0.0, 0.0),
    (2.0, 1.0, 1.57),
]


class NavigateMission(Node):

    def __init__(self):
        super().__init__('navigate_mission')

        # ---- Parameters ------------------------------------------------
        self.declare_parameter('rotation_speed', 0.3)
        self.declare_parameter('timeout_before_rotate', 3.0)
        self.declare_parameter('max_localization_time', 90.0)
        self.declare_parameter('localization_match_threshold', 2)
        self.declare_parameter('goal_tolerance', 0.25)
        self.declare_parameter('waypoint_timeout', 120.0)

        self._rotation_speed = self.get_parameter('rotation_speed').value
        self._loc_timeout = self.get_parameter('timeout_before_rotate').value
        self._loc_max_time = self.get_parameter('max_localization_time').value
        self._match_thresh = self.get_parameter('localization_match_threshold').value
        self._goal_tol = self.get_parameter('goal_tolerance').value
        self._wp_timeout = self.get_parameter('waypoint_timeout').value

        # ---- State -----------------------------------------------------
        self._phase = 'LOCALIZE'
        self._closure_count = 0
        self._start_time = self.get_clock().now()
        self._rotating = False

        self._waypoints = list(WAYPOINTS)
        self._wp_index = 0
        self._wp_sent = False
        self._wp_start_time = None
        self._wp_last_publish = None
        self._wp_accepted = False
        self._wp_initial_dist = None
        self._current_pose = None

        # ---- TF for map-frame pose lookup --------------------------------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ---- Pubs / Subs -----------------------------------------------
        self._cmd_pub = self.create_publisher(Twist, '/mobile_base/cmd_vel', 10)
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.create_subscription(Info, '/rtabmap/info', self._info_cb, 10)
        self.create_subscription(Odometry, '/mobile_base/odom', self._odom_cb, 10)

        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f'Mission loaded with {len(self._waypoints)} waypoints.  '
            f'Localizing first (will rotate after {self._loc_timeout:.0f}s)...'
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
            self.get_logger().info(
                f'Visual match #{self._closure_count}: {source}'
            )
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
    # Main tick (10 Hz)
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
        elapsed = self._elapsed_since(self._start_time)

        if elapsed > self._loc_max_time:
            self._stop_robot()
            self.get_logger().error(
                f'Failed to localize after {self._loc_max_time:.0f}s.  '
                'Try starting in a mapped area with visible features.'
            )
            raise SystemExit(1)

        if elapsed > self._loc_timeout and not self._rotating:
            self._rotating = True
            self.get_logger().info(
                'No match yet -- rotating to find visual landmarks...'
            )

        if self._rotating:
            twist = Twist()
            twist.angular.z = self._rotation_speed
            self._cmd_pub.publish(twist)

    def _finish_localization(self):
        self._stop_robot()
        elapsed = self._elapsed_since(self._start_time)
        self.get_logger().info(
            '========================================\n'
            f'  LOCALIZED in {elapsed:.1f}s  '
            f'({self._closure_count} visual matches)\n'
            '========================================\n'
            'Starting waypoint navigation...'
        )
        self._phase = 'NAVIGATE'

    # ================================================================
    # Phase 2 -- Navigate
    # ================================================================

    def _navigate_tick(self):
        if self._wp_index >= len(self._waypoints):
            self._stop_robot()
            self.get_logger().info(
                '========================================\n'
                '  MISSION COMPLETE\n'
                f'  All {len(self._waypoints)} waypoints visited.\n'
                '========================================'
            )
            raise SystemExit(0)

        if not self._wp_sent:
            self._send_waypoint()
        elif not self._wp_accepted and self._elapsed_since(self._wp_last_publish) > 3.0:
            if self._elapsed_since(self._wp_start_time) < 15.0:
                self._republish_waypoint()
            else:
                self._wp_accepted = True
                self.get_logger().info(
                    'Assuming planner accepted waypoint (timeout-based).'
                )

        if self._current_pose is None:
            return

        wx, wy, _ = self._waypoints[self._wp_index]
        dx = self._current_pose.position.x - wx
        dy = self._current_pose.position.y - wy
        dist = math.sqrt(dx * dx + dy * dy)

        if not self._wp_accepted:
            if self._wp_initial_dist is None:
                self._wp_initial_dist = dist
            elif dist < self._wp_initial_dist - 0.05:
                self._wp_accepted = True
                self.get_logger().info(
                    f'Planner accepted waypoint (robot moving, dist={dist:.2f}m).'
                )

        if dist < self._goal_tol:
            self.get_logger().info(
                f'Waypoint {self._wp_index + 1}/{len(self._waypoints)} reached '
                f'(dist={dist:.2f}m).'
            )
            self._advance_waypoint()
            return

        if self._elapsed_since(self._wp_start_time) > self._wp_timeout:
            self.get_logger().warn(
                f'Waypoint {self._wp_index + 1}/{len(self._waypoints)} timed out '
                f'after {self._wp_timeout:.0f}s (dist={dist:.2f}m).  Skipping.'
            )
            self._advance_waypoint()

    def _build_goal_msg(self):
        x, y, yaw = self._waypoints[self._wp_index]
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        return goal

    def _send_waypoint(self):
        goal = self._build_goal_msg()
        self._goal_pub.publish(goal)
        self._wp_sent = True
        self._wp_start_time = self.get_clock().now()
        self._wp_last_publish = self.get_clock().now()

        x, y, yaw = self._waypoints[self._wp_index]
        self.get_logger().info(
            f'Sending waypoint {self._wp_index + 1}/{len(self._waypoints)}: '
            f'({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.0f}deg)'
        )

    def _republish_waypoint(self):
        """Re-publish the current goal in case the planner wasn't ready."""
        goal = self._build_goal_msg()
        self._goal_pub.publish(goal)
        self._wp_last_publish = self.get_clock().now()
        self.get_logger().info(
            f'Re-publishing waypoint {self._wp_index + 1}/{len(self._waypoints)} '
            '(planner may not have been ready)...'
        )

    def _advance_waypoint(self):
        self._wp_index += 1
        self._wp_sent = False
        self._wp_start_time = None
        self._wp_last_publish = None
        self._wp_accepted = False
        self._wp_initial_dist = None

    # ================================================================
    # Helpers
    # ================================================================

    def _stop_robot(self):
        self._rotating = False
        self._cmd_pub.publish(Twist())

    def _elapsed_since(self, stamp):
        return (self.get_clock().now() - stamp).nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = NavigateMission()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
