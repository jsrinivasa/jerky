#!/usr/bin/env python3
"""
High-level task API for Mobile ALOHA.

Thin, dependency-light wrappers over the existing nav stack + ACT policies:

    go_to_location(x, y, yaw)   -> publish PoseStamped to /goal_pose, wait for arrival
    go_to_room("jaws")          -> resolve room name -> (x, y) -> go_to_location
    pick_up("cup")              -> run a trained ACT policy (via run_policy.py)

Navigation contract (from navigate_mission.py / simple_nav_planner.py):
  * goals are PoseStamped on /goal_pose in the 'map' frame
  * arrival is judged from TF map -> base_link (works whether map->odom is static
    in sim or published by AMCL on the real robot)

The room-resolution helpers (load_rooms / resolve_room) are pure Python and can be
imported/tested WITHOUT ros2:
    from aloha.aloha_tasks import load_rooms, resolve_room

CLI:
    python3 aloha_tasks.py rooms --query conf
    python3 aloha_tasks.py goto-room "jaws"
    python3 aloha_tasks.py goto-xy -3.85 -21.73 --yaw 0
    python3 aloha_tasks.py pickup cup --ckpt-dir /home/aloha/models-long
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Optional

# Rooms DB shipped with the repo (the readme's *_nav_rooms.json does not exist;
# this corrected file does). Override with --rooms-json.
DEFAULT_ROOMS_JSON = (
    "/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_rooms_corrected.json"
)
RUN_POLICY = "/home/aloha/interbotix_ws/src/aloha/scripts/run_policy.py"

# Friendly words -> room 'category' in the JSON, so go_to_room("elevator") works.
CATEGORY_ALIASES = {
    "elevator": "vertical_transport",
    "elevators": "vertical_transport",
    "stair": "vertical_transport",
    "stairs": "vertical_transport",
    "restroom": "restroom",
    "bathroom": "restroom",
    "toilet": "restroom",
    "conference": "conference_room",
    "conf": "conference_room",
    "corridor": "corridor",
    "hallway": "corridor",
}


# ---------------------------------------------------------------------------
# Pure-Python room resolution (no ros2 required)
# ---------------------------------------------------------------------------
def load_rooms(path: str = DEFAULT_ROOMS_JSON) -> list[dict]:
    with open(path) as f:
        rooms = json.load(f)
    if not isinstance(rooms, list):
        raise ValueError(f"{path}: expected a JSON list of rooms")
    return rooms


def resolve_room(rooms: list[dict], query: str,
                 category: Optional[str] = None) -> list[dict]:
    """Return rooms matching `query`, best matches first.

    Matching order: exact label (case-insensitive) > label startswith >
    substring in label. `category` (or a CATEGORY_ALIAS in the query) filters.
    """
    q = query.strip().lower()
    cat = category or CATEGORY_ALIASES.get(q)
    pool = [r for r in rooms if (cat is None or r.get("category") == cat)]

    def label(r):
        return str(r.get("label", "")).lower()

    exact = [r for r in pool if label(r) == q]
    starts = [r for r in pool if label(r).startswith(q) and r not in exact]
    contains = [r for r in pool
                if q in label(r) and r not in exact and r not in starts]
    # If the query was purely a category word, return the whole (sorted) pool.
    if not (exact or starts or contains) and cat is not None:
        return pool
    return exact + starts + contains


# ---------------------------------------------------------------------------
# ROS 2 task node (imported lazily so the helpers above work without rclpy)
# ---------------------------------------------------------------------------
class AlohaTasks:
    def __init__(self, rooms_json: str = DEFAULT_ROOMS_JSON,
                 goal_topic: str = "/goal_pose",
                 map_frame: str = "map", base_frame: str = "base_link",
                 goal_tolerance: float = 0.3):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        import tf2_ros

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node = Node("aloha_tasks")
        self.map_frame = map_frame
        self.base_frame = base_frame
        self.goal_tolerance = goal_tolerance
        self.rooms = load_rooms(rooms_json)

        self._PoseStamped = PoseStamped
        self._goal_pub = self.node.create_publisher(PoseStamped, goal_topic, 10)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

    # ---- navigation ------------------------------------------------------
    def _robot_xy(self):
        """Current (x, y) of base in map frame via TF, or None."""
        import tf2_ros
        from rclpy.time import Time
        try:
            tf = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
            return (tf.transform.translation.x, tf.transform.translation.y)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    def go_to_location(self, x: float, y: float, yaw: float = 0.0,
                       block: bool = True, timeout: float = 120.0) -> bool:
        """Publish a nav goal at (x, y, yaw) in map frame. If block, wait for arrival."""
        goal = self._PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        # Publish a few times — the planner may not have subscribed yet.
        for _ in range(5):
            self._goal_pub.publish(goal)
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
        self.node.get_logger().info(f"goal sent: ({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.0f}deg)")

        if not block:
            return True

        start = self.node.get_clock().now()
        while self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
            pos = self._robot_xy()
            if pos is not None:
                d = math.hypot(pos[0] - x, pos[1] - y)
                if d < self.goal_tolerance:
                    self.node.get_logger().info(f"arrived (dist={d:.2f}m)")
                    return True
            if (self.node.get_clock().now() - start).nanoseconds / 1e9 > timeout:
                self.node.get_logger().warn(f"go_to_location timed out after {timeout:.0f}s")
                return False
        return False

    def go_to_room(self, name: str, block: bool = True, timeout: float = 180.0) -> bool:
        matches = resolve_room(self.rooms, name)
        if not matches:
            self.node.get_logger().error(f"no room matches '{name}'")
            return False
        r = matches[0]
        if len(matches) > 1:
            others = ", ".join(str(m["label"]) for m in matches[1:6])
            self.node.get_logger().info(
                f"'{name}' -> '{r['label']}' ({r['category']}); other matches: {others}")
        return self.go_to_location(r["map_x"], r["map_y"], 0.0, block=block, timeout=timeout)

    # ---- manipulation ----------------------------------------------------
    def pick_up(self, obj: str, ckpt_dir: Optional[str] = None,
                policy_class: str = "ACT", task_name: Optional[str] = None) -> bool:
        """Run a trained ACT policy to pick up `obj`.

        Requires a trained checkpoint (see Task #3 / act_training_evaluation).
        Delegates to scripts/run_policy.py as a subprocess.
        """
        if ckpt_dir is None:
            self.node.get_logger().error(
                "pick_up needs --ckpt_dir pointing at a trained policy "
                "(train one via act_training_evaluation, or pass ckpt_dir).")
            return False
        if not os.path.isdir(ckpt_dir):
            self.node.get_logger().error(f"ckpt_dir not found: {ckpt_dir}")
            return False
        cmd = [sys.executable, RUN_POLICY, "--ckpt_dir", ckpt_dir,
               "--policy_class", policy_class, "--temporal_agg"]
        if task_name:
            cmd += ["--task_name", task_name]
        self.node.get_logger().info(f"pick_up('{obj}') -> {' '.join(cmd)}")
        return subprocess.call(cmd) == 0

    def shutdown(self):
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Mobile ALOHA high-level task API")
    p.add_argument("--rooms-json", default=DEFAULT_ROOMS_JSON)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("rooms", help="list/search rooms (no robot needed)")
    pr.add_argument("--query", default="")
    pr.add_argument("--category", default=None)

    pg = sub.add_parser("goto-room")
    pg.add_argument("name")
    pg.add_argument("--no-block", action="store_true")

    px = sub.add_parser("goto-xy")
    px.add_argument("x", type=float)
    px.add_argument("y", type=float)
    px.add_argument("--yaw", type=float, default=0.0)
    px.add_argument("--no-block", action="store_true")

    pp = sub.add_parser("pickup")
    pp.add_argument("obj")
    pp.add_argument("--ckpt-dir", default=None)
    pp.add_argument("--task-name", default=None)

    args = p.parse_args(argv)

    # 'rooms' is offline — no ros2 needed.
    if args.cmd == "rooms":
        rooms = load_rooms(args.rooms_json)
        matches = resolve_room(rooms, args.query, args.category) if (args.query or args.category) else rooms
        print(f"{len(matches)} room(s):")
        for r in matches[:60]:
            print(f"  {str(r['label']):28s} {r['category']:18s} x={r['map_x']:8.2f} y={r['map_y']:8.2f}")
        return 0

    tasks = AlohaTasks(rooms_json=args.rooms_json)
    try:
        if args.cmd == "goto-room":
            ok = tasks.go_to_room(args.name, block=not args.no_block)
        elif args.cmd == "goto-xy":
            ok = tasks.go_to_location(args.x, args.y, args.yaw, block=not args.no_block)
        elif args.cmd == "pickup":
            ok = tasks.pick_up(args.obj, ckpt_dir=args.ckpt_dir, task_name=args.task_name)
        else:
            ok = False
        return 0 if ok else 1
    finally:
        tasks.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
