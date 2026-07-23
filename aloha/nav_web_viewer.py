#!/usr/bin/env python3
"""
Nav Web Viewer

Local web page (Flask) showing the robot's live RTAB-Map occupancy grid
overlaid on the building floorplan, with its current localized pose. A
click stages a candidate goal (in map-frame meters) -- nothing is sent
to the robot until a second, explicit "Confirm & Go" step. A STOP button
is always visible and publishes /nav_cancel, which simple_nav_planner.py
treats as an immediate kill switch regardless of what it's doing.

Run with the nav stack already up (e.g. navigate_mission.launch.py):
    ros2 run aloha nav_web_viewer
Then open http://<this-machine's-LAN-ip>:8080/ in a browser.
"""

import json
import math
import subprocess
import threading
import time
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, Response, jsonify, request

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,
)
import tf2_ros

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String
from std_srvs.srv import Empty as EmptySrv

ROOMS_JSON = (
    '/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_rooms_corrected.json'
)
FLOORPLAN_SVG = '/home/aloha/interbotix_ws/src/aloha/maps/floorplan.svg'
FALLBACK_PGM = '/home/aloha/maps/building16_with_parking_edited.pgm'
FALLBACK_YAML = '/home/aloha/maps/building16_with_parking_edited.yaml'

# The robot navigates in whatever frame its OWN live SLAM session establishes
# (frame "B") -- a fresh map-free session each run, so frame B's origin is
# different every time and nothing about it can be persisted across runs.
# That frame is tied to the floorplan/rooms-JSON frame ("A") via a single
# "click where the robot is + which way it's facing" anchor each session
# (see NavWebViewerNode.set_anchor / the /api/anchor route), not a prebuilt
# reference map or a multi-point fit.
#
# FLOORPLAN_SCALE is the one piece that IS safe to reuse run over run: it's a
# property of the floorplan itself (frame A's meters-per-unit), not of any
# particular SLAM session, so a single anchor point + this fixed scale fully
# determines the rotation+translation (Umeyama/least-squares needs >=2 points
# only because it doesn't know the scale in advance).
#
# 2026-07-22: recalibrated via direct measurement -- clicked two points on
# the live web map spanning a real, tape-measured 59in (1.4986m) desk run
# (frame-A distance 1.5417 units between the clicks, logged via the
# [CALIBRATION-CLICK] stage_goal log line), giving 1.5417/1.4986=1.028762.
# This replaces the old value (1.0849674771110611, inherited from a prior
# session's building-wide multi-point SLAM-vs-floorplan fit) -- 5.5% smaller,
# meaning that old value was making every commanded distance undershoot the
# real one by about 5%. A pixel-measurement of a door opening was tried
# first and gave a wildly different (~2x) result, most likely from
# misreading a double-door or icon detail -- discarded in favor of this
# direct, click-based measurement against a real tape-measured reference.
FLOORPLAN_SCALE = 1.028762

SVG_RENDER_WIDTH = 2400
SVG_RENDER_TIMEOUT_S = 60.0

HOST = '0.0.0.0'
PORT = 8080


def fit_svg_affine(rooms_json_path):
    """Per-axis least-squares fit: svg_x = a_x*map_x + b_x, svg_y = a_y*map_y + b_y.

    No rotation/shear term -- confirmed at plan time that the fit is a pure
    axis-aligned scale+offset to sub-pixel accuracy across all 556 rooms.
    """
    with open(rooms_json_path) as f:
        rooms = json.load(f)
    map_x = np.array([r['map_x'] for r in rooms], dtype=float)
    map_y = np.array([r['map_y'] for r in rooms], dtype=float)
    svg_x = np.array([r['svg_x'] for r in rooms], dtype=float)
    svg_y = np.array([r['svg_y'] for r in rooms], dtype=float)
    a_x, b_x = np.polyfit(map_x, svg_x, 1)
    a_y, b_y = np.polyfit(map_y, svg_y, 1)
    return {'a_x': float(a_x), 'b_x': float(b_x),
            'a_y': float(a_y), 'b_y': float(b_y)}


def read_svg_viewbox(svg_path):
    """Pull viewBox="minx miny w h" without parsing the whole 36MB file."""
    with open(svg_path, 'rb') as f:
        head = f.read(4096).decode('utf-8', errors='ignore')
    start = head.index('viewBox="') + len('viewBox="')
    end = head.index('"', start)
    minx, miny, w, h = (float(v) for v in head[start:end].split())
    return minx, miny, w, h


def render_floorplan_png(svg_path, output_width):
    """Render the SVG to PNG bytes once, off the hot path, with a hard timeout.

    Returns (png_bytes, png_width, png_height) or None on failure/timeout --
    caller falls back to the georeferenced PGM base layer.
    """
    result = {}

    def _render():
        try:
            import cairosvg
            result['png'] = cairosvg.svg2png(
                url=svg_path, output_width=output_width)
        except Exception as e:  # noqa: BLE001 - report via dict, not raise across threads
            result['error'] = e

    t = threading.Thread(target=_render, daemon=True)
    t.start()
    t.join(timeout=SVG_RENDER_TIMEOUT_S)
    if t.is_alive() or 'png' not in result:
        return None
    img = Image.open(BytesIO(result['png']))
    return result['png'], img.width, img.height


def render_pgm_fallback(pgm_path, yaml_path):
    import yaml
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img = Image.open(pgm_path).convert('RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    return {
        'png': buf.getvalue(),
        'width': img.width,
        'height': img.height,
        'resolution': meta['resolution'],
        'origin_x': meta['origin'][0],
        'origin_y': meta['origin'][1],
    }


def compute_anchor_transform(ax, ay, yaw_a, bx, by, yaw_b, scale=FLOORPLAN_SCALE):
    """Closed-form similarity transform (frame B meters -> frame A meters)
    from a SINGLE position+heading correspondence and a fixed scale.

    (ax, ay, yaw_a): the anchor as picked on the floorplan (frame A).
    (bx, by, yaw_b): the robot's live SLAM pose (frame B) at anchor time.

    With scale fixed, one correspondence fully determines rotation +
    translation -- no least-squares needed (that's only required when scale
    is also unknown, per point).
    """
    theta = yaw_a - yaw_b
    c, sn = math.cos(theta), math.sin(theta)
    tx = ax - scale * (c * bx - sn * by)
    ty = ay - scale * (sn * bx + c * by)
    return {'theta': theta, 'scale': scale, 'tx': tx, 'ty': ty}


def apply_similarity(calib, x, y):
    """Frame B (meters) -> frame A (meters)."""
    s, th, tx, ty = calib['scale'], calib['theta'], calib['tx'], calib['ty']
    c, sn = math.cos(th), math.sin(th)
    return s * (c * x - sn * y) + tx, s * (sn * x + c * y) + ty


def apply_similarity_inverse(calib, xa, ya):
    """Frame A (meters) -> frame B (meters)."""
    s, th, tx, ty = calib['scale'], calib['theta'], calib['tx'], calib['ty']
    c, sn = math.cos(th), math.sin(th)
    dx, dy = xa - tx, ya - ty
    return (c * dx + sn * dy) / s, (-sn * dx + c * dy) / s


def _mat_scale_translate(sx, sy, ox, oy):
    return np.array([[sx, 0, ox], [0, sy, oy], [0, 0, 1]], dtype=float)


def _mat_similarity(calib):
    s, th, tx, ty = calib['scale'], calib['theta'], calib['tx'], calib['ty']
    c, sn = math.cos(th), math.sin(th)
    return np.array([[s * c, -s * sn, tx], [s * sn, s * c, ty], [0, 0, 1]], dtype=float)


class NavWebViewerNode(Node):

    def __init__(self):
        super().__init__('nav_web_viewer')

        self.lock = threading.Lock()
        self.staged_goal = None  # {'ax','ay','bx','by','yaw'} or None
        self.live_grid = None    # {'array': np.ndarray, 'resolution', 'origin_x', 'origin_y', 'width', 'height'}
        self._live_map_png_cache = None  # (cache_key, png_bytes), see render_live_map_png
        self.pose = None         # {'x':.., 'y':.., 'yaw':..} in frame B (raw TF) or None
        self._displayed_pose_a = None  # rate-limited version of pose_a, see get_state

        # ---- frame B -> frame A anchor (see set_anchor / /api/anchor) ----
        # Set once per session by the user clicking "here" + "facing this
        # way" on the floorplan; None until then. A fresh SLAM session means
        # frame B's origin differs every run, so nothing here is persisted.
        self.calibration = None
        self.get_logger().warn(
            'No anchor set yet -- pose/live-map/goals will be WRONG until '
            'you click an anchor on the page. Refusing to confirm goals until then.')

        # ---- base layer: try the floorplan SVG, fall back to the PGM ----
        self.base_layer = None
        rendered = render_floorplan_png(FLOORPLAN_SVG, SVG_RENDER_WIDTH)
        if rendered is not None:
            png, w, h = rendered
            minx, miny, vb_w, vb_h = read_svg_viewbox(FLOORPLAN_SVG)
            affine = fit_svg_affine(ROOMS_JSON)
            # map_x -> svg_x -> png px (linear composition of both steps)
            sx = affine['a_x'] * w / vb_w
            ox = (affine['b_x'] - minx) * w / vb_w
            sy = affine['a_y'] * h / vb_h
            oy = (affine['b_y'] - miny) * h / vb_h
            self.base_layer = {
                'type': 'svg', 'png': png, 'width': w, 'height': h,
                'sx': sx, 'ox': ox, 'sy': sy, 'oy': oy,
            }
            self.get_logger().info(
                f'Floorplan SVG rendered OK ({w}x{h}px); using it as the base layer.')
        else:
            fb = render_pgm_fallback(FALLBACK_PGM, FALLBACK_YAML)
            res = fb['resolution']
            h = fb['height']
            self.base_layer = {
                'type': 'pgm', 'png': fb['png'], 'width': fb['width'], 'height': h,
                'sx': 1.0 / res, 'ox': -fb['origin_x'] / res,
                'sy': -1.0 / res, 'oy': h + fb['origin_y'] / res,
            }
            self.get_logger().warn(
                'Floorplan SVG render failed/timed out; falling back to '
                'the georeferenced PGM as the base layer.')

        # ---- static floorplan-derived obstacle grid (frame A pixel space) --
        # Walls/exterior boundary from the architectural floorplan itself,
        # not the live camera/lidar SLAM map -- avoids the whole class of
        # SLAM registration/drift problems for PLANNING specifically (the
        # live sensors still gate real-time collision avoidance separately,
        # in simple_nav_planner, unaffected by this). "Not white" -> wall:
        # checked visually against the actual rendered floorplan -- the
        # exterior boundary and interior partitions are all dark gray/black
        # against a white floor background, so this one threshold catches
        # both without needing separate exterior/interior handling.
        import cv2
        floor_arr = np.array(Image.open(BytesIO(self.base_layer['png'])).convert('L'))
        occ_bin = (floor_arr < 240).astype(np.uint8)
        # Light denoise only (open, no dilate/close) -- thin real wall lines
        # must survive; this just drops stray 1px noise. Deliberately RAW
        # (no extra inflation) here: simple_nav_planner already has its own
        # inflation_radius margin for A* clearance (see
        # autonomous_mapping.launch.py) -- inflating the source data too
        # double-stacked the two, closing off every corridor and making
        # every goal "unreachable" (confirmed live: this is what broke
        # Confirm & Go). Extra clearance belongs in inflation_radius, the
        # one place that's actually meant to own it, not here too.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        occ_bin = cv2.morphologyEx(occ_bin, cv2.MORPH_OPEN, kernel)
        self.floorplan_occ_px = occ_bin  # frame-A PIXEL space, 1=wall/boundary, 0=free
        self.get_logger().info(
            f'Floorplan obstacle grid built from base layer: '
            f'{100 * occ_bin.mean():.1f}% marked occupied.')

        # ---- named locations (rooms/desks) for "go to <label>" ----
        # Same ROOMS_JSON already used above for fit_svg_affine -- map_x/
        # map_y there are frame-A METERS (confirmed: fit_svg_affine fits
        # svg_x = a*map_x+b, and this file's base_layer sx/sy/ox/oy compose
        # that same fit with the render scale -- the exact chain the
        # frontend's pxToMap/mapToPx already use), so no extra conversion
        # is needed to feed them into stage_goal.
        self.locations = []
        try:
            with open(ROOMS_JSON) as f:
                self.locations = json.load(f)
            self.get_logger().info(
                f'Loaded {len(self.locations)} named locations from {ROOMS_JSON}')
        except Exception as e:
            self.get_logger().warn(
                f'Could not load room locations ({type(e).__name__}: {e}) '
                '-- "go to <label>" search will return no results.')
        self._locations_by_label = {loc['label'].upper(): loc for loc in self.locations}

        # ---- ROS I/O ----
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        # /floorplan_map, NOT /rtabmap/map: with use_odom_locked_map:=true,
        # RTAB-Map's own internal pose (which /rtabmap/map's cells are built
        # relative to) is completely decoupled from the locked "map" TF
        # frame this whole page otherwise operates in -- self.calibration
        # would warp it using the wrong transform entirely, and there's no
        # TF published anymore to derive the right one from (that's the
        # point of the lock). /floorplan_map is already published directly
        # in the locked frame (see _publish_floorplan_map), so it's the only
        # live_map source that's actually correctly aligned by construction.
        self.create_subscription(
            OccupancyGrid, '/floorplan_map', self._map_callback, map_qos)

        # Live camera/sensor-detected obstacles, shown as a SEPARATE overlay
        # from the floorplan grid -- projected via the trustworthy locked
        # map frame (map->base_link, now odometry-only) + the scan's own
        # static TF to base_link, mirroring simple_nav_planner's own
        # _live_scan_points_map_frame exactly. Deliberately NOT sourced from
        # RTAB-Map's /rtabmap/map (see the comment above) -- that's still in
        # RTAB-Map's own decoupled, unpublished internal frame.
        scan_qos = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.latest_scan = None
        # Accumulated camera-detected obstacle points, so the overlay builds
        # up a picture of everywhere the robot has looked -- not just the
        # latest single scan. Snapped to a coarse grid and stored as a set
        # (see _accumulate_scan_points) so revisiting the same spot doesn't
        # grow this unboundedly over a long session.
        self._scan_accum_res = 0.05  # meters/cell
        self.accumulated_scan_cells = set()
        self._anchor_settle_until = None  # see _SCAN_ACCUM_SETTLE_S
        self.create_subscription(LaserScan, '/scan', self._scan_callback, scan_qos)

        path_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.planned_path = None  # list of (x, y) in frame B, or None
        self.create_subscription(
            Path, '/smoothed_path', self._path_callback, path_qos)

        # Planner state visibility -- previously this page had NO way to
        # know if a goal's planning had failed (see simple_nav_planner.py's
        # _publish_status/plan_and_navigate for the bug this was added to
        # fix: a 2nd goal that couldn't be planned used to silently leave
        # the FIRST goal's path on screen forever, with zero indication
        # anything had gone wrong).
        self.nav_status = 'idle'
        self.create_subscription(
            String, '/nav_status', self._nav_status_callback, path_qos)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.create_timer(0.5, self._pose_tick)

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cancel_pub = self.create_publisher(Empty, '/nav_cancel', 10)
        self.floorplan_map_pub = self.create_publisher(
            OccupancyGrid, '/floorplan_map', map_qos)
        # Called at the start of every anchor confirm -- discards map data
        # recorded before the anchor moment, so what's on screen only ever
        # reflects "since you told us where the robot is," not whatever
        # accumulated (possibly during startup/positioning) beforehand.
        self._new_map_client = self.create_client(
            EmptySrv, '/rtabmap/rtabmap/trigger_new_map')

    def _map_callback(self, msg: OccupancyGrid):
        arr = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        with self.lock:
            self.live_grid = {
                'array': arr,
                'resolution': msg.info.resolution,
                'origin_x': msg.info.origin.position.x,
                'origin_y': msg.info.origin.position.y,
                'width': msg.info.width,
                'height': msg.info.height,
            }

    def _path_callback(self, msg: Path):
        with self.lock:
            self.planned_path = [
                (p.pose.position.x, p.pose.position.y) for p in msg.poses
            ]

    def _nav_status_callback(self, msg: String):
        with self.lock:
            self.nav_status = msg.data

    # Throttle accumulation instead of running it on every incoming scan
    # (camera publishes at ~15-30Hz) -- a full TF lookup + looping ~640
    # ranges + a set-union on EVERY message measurably loaded the CPU
    # (confirmed live: nav_web_viewer sustained ~48% CPU with this
    # unthrottled), competing for scheduling time with simple_nav_planner's
    # own 10Hz control loop on the same machine -- a very plausible real
    # cause of choppy motion having nothing to do with the planner/safety
    # logic itself. There's no benefit to accumulating faster than the
    # display itself refreshes anyway (500ms, see refreshLiveScan).
    _SCAN_ACCUM_MIN_INTERVAL_S = 0.5
    _last_scan_accum_time = 0.0

    def _scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        now = time.time()
        if now - self._last_scan_accum_time >= self._SCAN_ACCUM_MIN_INTERVAL_S:
            self._last_scan_accum_time = now
            self._accumulate_scan_points(msg)

    # Exactly the sensor's own trusted range (see depthimage_to_laserscan's
    # range_max) -- no margin beyond it, so nothing shown as "seen" is
    # farther than the camera can actually be trusted at.
    _SCAN_ACCUM_MAX_DIST_M = 3.0
    _SCAN_ACCUM_MAX_CELLS = 200000
    # Skip accumulating for a short settle window right after a new anchor
    # -- pose/TF right at that instant can still be catching up (matches
    # the same startup-transient pattern seen elsewhere in this project),
    # and unlike the old "latest scan only" rendering, bad early points now
    # get baked in permanently instead of just self-correcting next tick.
    _SCAN_ACCUM_SETTLE_S = 2.0

    def _accumulate_scan_points(self, scan: LaserScan):
        """Project this scan into the map frame (same trustworthy odometry-
        locked chain render_live_scan_png uses) and fold its points into the
        running accumulated set, so the overlay keeps everywhere the camera
        has ever seen, not just the latest instant. Uses THIS scan's own TF
        at receipt time, not whatever TF is current when later rendered --
        each scan's points need to be placed using the pose the robot
        actually had when it captured them.
        """
        if self._anchor_settle_until is not None and time.time() < self._anchor_settle_until:
            return
        with self.lock:
            pose = self.pose
        if pose is None:
            return
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                'map', scan.header.frame_id, rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return

        q = tf_stamped.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        tx = tf_stamped.transform.translation.x
        ty = tf_stamped.transform.translation.y
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        res = self._scan_accum_res

        new_cells = set()
        angle = scan.angle_min
        for rng in scan.ranges:
            if not (math.isnan(rng) or math.isinf(rng)) and scan.range_min <= rng <= scan.range_max:
                px_, py_ = rng * math.cos(angle), rng * math.sin(angle)
                map_x = cos_yaw * px_ - sin_yaw * py_ + tx
                map_y = sin_yaw * px_ + cos_yaw * py_ + ty
                if math.hypot(map_x - pose['x'], map_y - pose['y']) <= self._SCAN_ACCUM_MAX_DIST_M:
                    new_cells.add((round(map_x / res), round(map_y / res)))
            angle += scan.angle_increment

        with self.lock:
            self.accumulated_scan_cells |= new_cells
            if len(self.accumulated_scan_cells) > self._SCAN_ACCUM_MAX_CELLS:
                # Trim rather than grow forever -- exact set kept is
                # arbitrary once over the cap, just bounding memory/render
                # cost for an unusually long session.
                self.accumulated_scan_cells = set(
                    list(self.accumulated_scan_cells)[-self._SCAN_ACCUM_MAX_CELLS // 2:])

    def _pose_tick(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.pose = {'x': t.x, 'y': t.y, 'yaw': yaw}

    # Real motion can't exceed the planner's own velocity caps (0.3 m/s,
    # 1.0 rad/s) -- these per-call step limits are generous relative to
    # that (well above what a real ~0.5s poll interval implies), so real
    # movement always keeps up, but a SLAM registration snap (the pose
    # teleporting many degrees/meters in one tick, independent of actual
    # robot motion -- the root cause behind today's "red dot jumps") gets
    # confined to a bounded step per update instead of drawn instantly.
    # Display-only: does not touch simple_nav_planner's own pose/control.
    _POSE_STEP_MAX_M = 0.2
    _POSE_STEP_MAX_RAD = 0.3

    def _smooth_displayed_pose(self, pose_a):
        """Caller must already hold self.lock. Returns a step-limited copy
        of pose_a, and updates self._displayed_pose_a for next call."""
        prev = self._displayed_pose_a
        if prev is None:
            self._displayed_pose_a = dict(pose_a)
            return dict(pose_a)
        dx = pose_a['x'] - prev['x']
        dy = pose_a['y'] - prev['y']
        dist = math.hypot(dx, dy)
        if dist > self._POSE_STEP_MAX_M:
            frac = self._POSE_STEP_MAX_M / dist
            new_x = prev['x'] + dx * frac
            new_y = prev['y'] + dy * frac
        else:
            new_x, new_y = pose_a['x'], pose_a['y']
        dyaw = math.atan2(math.sin(pose_a['yaw'] - prev['yaw']),
                           math.cos(pose_a['yaw'] - prev['yaw']))
        if abs(dyaw) > self._POSE_STEP_MAX_RAD:
            new_yaw = prev['yaw'] + math.copysign(self._POSE_STEP_MAX_RAD, dyaw)
        else:
            new_yaw = pose_a['yaw']
        smoothed = {'x': new_x, 'y': new_y, 'yaw': new_yaw}
        self._displayed_pose_a = smoothed
        return smoothed

    # ---- called from Flask handlers (different thread) ----

    def get_state(self):
        with self.lock:
            live_map = None
            if self.live_grid is not None:
                g = self.live_grid
                live_map = {
                    'resolution': g['resolution'], 'origin_x': g['origin_x'],
                    'origin_y': g['origin_y'], 'width': g['width'], 'height': g['height'],
                }
            pose_a = None
            if self.pose is not None:
                if self.calibration:
                    xa, ya = apply_similarity(self.calibration, self.pose['x'], self.pose['y'])
                    # Position rotates through the anchor's rotation (theta) via
                    # apply_similarity -- heading must rotate by the same theta,
                    # or the drawn arrow shows the raw frame-B heading instead of
                    # the heading relative to the floorplan.
                    yaw_a = self.pose['yaw'] + self.calibration['theta']
                else:
                    xa, ya = self.pose['x'], self.pose['y']  # wrong, but better than nothing
                    yaw_a = self.pose['yaw']
                pose_a = {'x': xa, 'y': ya, 'yaw': yaw_a}
                pose_a = self._smooth_displayed_pose(pose_a)
            staged = None
            if self.staged_goal:
                sg = self.staged_goal
                staged = {'x': sg['ax'], 'y': sg['ay'], 'yaw': sg['yaw']}
            path_a = None
            if self.planned_path and self.calibration:
                path_a = [
                    {'x': xa, 'y': ya}
                    for xa, ya in (
                        apply_similarity(self.calibration, bx, by)
                        for bx, by in self.planned_path
                    )
                ]
            return {
                'pose': pose_a,
                'calibrated': self.calibration is not None,
                'base_layer': {
                    'type': self.base_layer['type'],
                    'width': self.base_layer['width'],
                    'height': self.base_layer['height'],
                    'sx': self.base_layer['sx'], 'ox': self.base_layer['ox'],
                    'sy': self.base_layer['sy'], 'oy': self.base_layer['oy'],
                },
                'live_map': live_map,
                'staged_goal': staged,
                'path': path_a,
                'nav_status': self.nav_status,
            }

    def render_live_map_png(self):
        """Warp the live occupancy grid directly into base-layer pixel space,
        composing grid-px->frame-B-meters, frame-B->frame-A (calibration),
        and frame-A-meters->base-layer-px into one affine (PIL resamples in
        one pass, so rotation from calibration comes along for free -- the
        frontend just draws the result full-canvas, no positioning math).

        Cached on (grid, calib) identity -- /floorplan_map is published ONCE
        per anchor and never changes until the next one, so redoing this
        affine warp on every poll (previously every 1.5s) was pure waste,
        found while tracking down unexpectedly high sustained CPU usage.
        """
        with self.lock:
            grid = self.live_grid
            calib = self.calibration
        if grid is None:
            return None
        cache_key = (id(grid), id(calib))
        if self._live_map_png_cache is not None and self._live_map_png_cache[0] == cache_key:
            return self._live_map_png_cache[1]
        arr = grid['array']
        rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)  # default transparent
        occupied = arr >= 50
        # Free space left fully transparent (no blue wash) -- the floorplan
        # grid covers the WHOLE building by construction (not just an
        # explored patch like the old live SLAM grid did), so tinting free
        # space would wash the entire visible floorplan in blue. Only mark
        # what actually matters: real obstacles.
        rgba[occupied] = (215, 90, 90, 130)  # softer/more translucent red
        # row index = y index (OccupancyGrid convention: data[row*width+col]),
        # baked directly into m1 below -- no image flip needed here.
        src_img = Image.fromarray(rgba, mode='RGBA')

        m1 = _mat_scale_translate(
            grid['resolution'], grid['resolution'], grid['origin_x'], grid['origin_y'])
        m2 = _mat_similarity(calib) if calib else np.eye(3)
        bl = self.base_layer
        m3 = _mat_scale_translate(bl['sx'], bl['sy'], bl['ox'], bl['oy'])
        m = m3 @ m2 @ m1
        m_inv = np.linalg.inv(m)
        coeffs = (m_inv[0, 0], m_inv[0, 1], m_inv[0, 2],
                  m_inv[1, 0], m_inv[1, 1], m_inv[1, 2])
        out = src_img.transform(
            (bl['width'], bl['height']), Image.AFFINE, coeffs, resample=Image.BILINEAR)
        buf = BytesIO()
        out.save(buf, format='PNG')
        png = buf.getvalue()
        self._live_map_png_cache = (cache_key, png)
        return png

    def render_live_scan_png(self):
        """Render the ACCUMULATED set of camera-detected points (see
        _accumulate_scan_points) -- everywhere the camera has looked this
        anchor session, not just the latest instant -- as a scatter,
        projected through the fixed frame-A<->map scale. A scatter, not an
        affine-warped grid, since it's a sparse point set, not a dense
        raster.

        Returns None (caller sends 204) if there's nothing accumulated yet
        or no anchor -- same "missing data must not look like a clear path"
        rule render_live_map_png follows.
        """
        with self.lock:
            calib = self.calibration
            cells = list(self.accumulated_scan_cells)
        if calib is None or not cells:
            return None

        bl = self.base_layer
        img = Image.new('RGBA', (bl['width'], bl['height']), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        r = 1.5  # dot radius, px -- small relative to the floorplan detail
        res = self._scan_accum_res

        for gx, gy in cells:
            map_x, map_y = gx * res, gy * res
            fa_x, fa_y = apply_similarity(calib, map_x, map_y)
            sx_, sy_ = fa_x * bl['sx'] + bl['ox'], fa_y * bl['sy'] + bl['oy']
            draw.ellipse((sx_ - r, sy_ - r, sx_ + r, sy_ + r), fill=(255, 170, 60, 170))

        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def stage_goal(self, x, y, yaw):
        """x, y are frame-A (floorplan) meters, from a click on the base layer."""
        with self.lock:
            calib = self.calibration
            if calib:
                bx, by = apply_similarity_inverse(calib, x, y)
            else:
                bx, by = None, None  # can't compute -- confirm_goal will refuse
            self.staged_goal = {'ax': x, 'ay': y, 'bx': bx, 'by': by, 'yaw': yaw}
            pose = self.pose
            pose_a = apply_similarity(calib, pose['x'], pose['y']) if (pose and calib) else \
                ((pose['x'], pose['y']) if pose else None)
        dist = math.hypot(x - pose_a[0], y - pose_a[1]) if pose_a else None
        # Logged for manual scale calibration -- clicking (staging) never
        # moves the robot, only Confirm & Go does, so this is a safe way to
        # read off exact frame-A coordinates for two physically-measured
        # reference points.
        self.get_logger().info(f'[CALIBRATION-CLICK] frame-A x={x:.4f} y={y:.4f}')
        return {'x': x, 'y': y, 'yaw': yaw, 'distance_m': dist, 'calibrated': calib is not None}

    def confirm_goal(self):
        with self.lock:
            goal = self.staged_goal
            self.staged_goal = None
        if goal is None:
            return False, 'no goal staged'
        if goal['bx'] is None:
            return False, 'not anchored -- click an anchor on the page before sending real goals'
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(goal['bx'])
        msg.pose.position.y = float(goal['by'])
        msg.pose.orientation.z = math.sin(goal['yaw'] / 2.0)
        msg.pose.orientation.w = math.cos(goal['yaw'] / 2.0)
        for _ in range(5):
            self.goal_pub.publish(msg)
            time.sleep(0.05)
        self.get_logger().info(
            f"Confirmed goal sent (frame B): x={goal['bx']:.2f} y={goal['by']:.2f} "
            f"(floorplan click was x={goal['ax']:.2f} y={goal['ay']:.2f}) "
            f"yaw={math.degrees(goal['yaw']):.0f}deg")
        return True, 'ok'

    def cancel(self):
        with self.lock:
            self.staged_goal = None
        self.cancel_pub.publish(Empty())
        self.get_logger().warn('STOP pressed: /nav_cancel published')
        # _stop_nav_teleop (called on anchor-confirm) retires manual teleop
        # for the rest of the session -- but STOP is exactly the moment a
        # user needs to manually drive the robot again (e.g. back to a
        # starting point after cancelling a bad drive), so bring it back
        # here. Safe to call even if it was never running.
        self._start_nav_teleop()

    def _start_nav_teleop(self):
        """Companion to _stop_nav_teleop: relaunch nav_joystick_teleop with
        the exact same parameters aloha_bringup.launch.py's nav_teleop_node
        uses (see that launch file for why these specific values -- must
        stay in sync), so STOP gives manual control back instead of leaving
        the robot strandable with no way to drive it except re-anchoring.

        No-ops if an instance is already running (checked by node name, not
        just "did we call this before" -- covers e.g. it having been left
        running because an anchor was never confirmed this session).
        """
        check = subprocess.run(
            ['pgrep', '-f', '__node:=nav_joystick_teleop'],
            capture_output=True, timeout=2.0)
        if check.returncode == 0:
            return  # already running
        try:
            subprocess.Popen(
                ['ros2', 'run', 'teleop_twist_joy', 'teleop_node',
                 '--ros-args',
                 '-r', '__node:=nav_joystick_teleop',
                 '-r', '__ns:=/mobile_base',
                 '-r', 'cmd_vel:=/nav_cmd_vel',
                 '-p', 'axis_linear.x:=1',
                 '-p', 'scale_linear.x:=0.35',
                 '-p', 'axis_angular.yaw:=3',
                 '-p', 'scale_angular.yaw:=0.3',
                 '-p', 'enable_button:=6'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            self.get_logger().info(
                'Restarted nav_joystick_teleop -- hold L2 and use the stick '
                'to drive manually (still gated by nav_deadman).')
        except Exception as e:
            self.get_logger().warn(
                f'_start_nav_teleop failed ({type(e).__name__}: {e}) -- '
                'manual teleop unavailable; restart demo_ops.sh bringup nav '
                'if you need it.')

    # ---- named-location lookup ("go to <label>") ----

    def search_locations(self, query, limit=20):
        """Case-insensitive match against room/desk labels (e.g. "314",
        "D3-12", "jaws"). Exact match first, then prefix, then substring --
        same ranking navigate_to_room.py's fuzzy_match_rooms left implicit
        by picking shortest-label-wins; made explicit here since this
        drives a live autocomplete dropdown, not a single CLI pick.
        """
        q = query.strip().upper()
        if not q:
            return []
        exact, prefix, contains = [], [], []
        for loc in self.locations:
            label = loc['label'].upper()
            if label == q:
                exact.append(loc)
            elif label.startswith(q):
                prefix.append(loc)
            elif q in label:
                contains.append(loc)
        results = (exact + prefix + contains)[:limit]
        return [{'label': l['label'], 'category': l['category']} for l in results]

    def goto_location(self, label):
        """Resolve a room/desk label to its own (map_x, map_y) and stage it
        as a goal via the normal stage_goal path (same as a click) -- so
        Confirm & Go, the distance readout, and everything else downstream
        is unchanged; this is purely an alternate way to PICK the point.

        Always the label's own center/icon location -- no separate
        "outside"/doorway-standoff mode. That was tried (ray-casting for a
        real doorway, even switching to a proper walls-only mask + a
        clearance check) but still wasn't landing reliably right in
        practice. simple_nav_planner's own A* already refuses to plan
        through occupied/inflated cells (is_point_collision_free_grid gates
        every neighbor in astar()) and already snaps an occupied/unreachable
        goal to find_nearest_free_cell -- so "never cross a wall to get
        there, stop at the nearest reachable point instead" is the
        planner's job, not something to re-solve here with floorplan pixel
        geometry.
        """
        loc = self._locations_by_label.get(label.strip().upper())
        if loc is None:
            return None, f"no location named '{label}'"
        result = self.stage_goal(loc['map_x'], loc['map_y'], 0.0)
        result['label'] = loc['label']
        result['category'] = loc['category']
        return result, 'ok'

    # ---- single-click anchor (see compute_anchor_transform) ----

    def _broadcast_map_odom_tf(self, ax, ay, yaw_a, odom_x, odom_y, odom_yaw):
        """Publish map->odom as a FIXED transform derived once from wheel+IMU
        odometry at anchor time -- not RTAB-Map's own SLAM pose. This is the
        actual fix for the pose jitter/jumps (and the real robot motion
        jerkiness they caused): once set, odom continues purely from wheel
        encoders + IMU gyro (robot_localization's EKF), never touched again
        by SLAM loop-closure/registration corrections, so map->base_link
        (TF composition of this fixed transform with the live
        odom->base_link) can only change as smoothly/boundedly as real
        robot motion allows.

        map frame is defined to share frame A's (the floorplan's) origin
        and global orientation exactly, differing only by the fixed
        floorplan<->real-meters scale -- so the anchor click (ax, ay,
        yaw_a), in frame-A units, is converted to real meters (divided by
        FLOORPLAN_SCALE) before being used as this transform's target.

        Requires rtabmap_mapping.launch.py's use_odom_locked_map:=true (the
        default) so rtabmap itself isn't ALSO trying to publish map->odom --
        two publishers on the same transform would silently conflict.
        """
        ax_m, ay_m = ax / FLOORPLAN_SCALE, ay / FLOORPLAN_SCALE
        calib = compute_anchor_transform(
            ax_m, ay_m, yaw_a, odom_x, odom_y, odom_yaw, scale=1.0)
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'odom'
        msg.transform.translation.x = calib['tx']
        msg.transform.translation.y = calib['ty']
        msg.transform.translation.z = 0.0
        msg.transform.rotation.z = math.sin(calib['theta'] / 2.0)
        msg.transform.rotation.w = math.cos(calib['theta'] / 2.0)
        self._tf_static_broadcaster.sendTransform(msg)
        self.get_logger().info(
            f"Broadcast map->odom: theta={math.degrees(calib['theta']):.1f}deg "
            f"tx={calib['tx']:.2f} ty={calib['ty']:.2f} (real meters)")

    def set_anchor(self, ax, ay, hx, hy):
        """(ax, ay): where the user clicked "the robot is here" on the
        floorplan. (hx, hy): where they clicked "the robot is facing this
        way" -- together these give a heading (yaw_a). Combined with the
        robot's live frame-B pose right now, this fully determines the
        floorplan<->SLAM transform for this session (see
        compute_anchor_transform).

        Starts with rtabmap's trigger_new_map service so any map data
        recorded before this moment (startup, positioning, driving around
        before anchoring) is discarded from what's shown/planned against
        going forward -- an anchor means "the map starts here," not just
        "here's a coordinate translation for whatever's already recorded."
        """
        if self._new_map_client.service_is_ready():
            future = self._new_map_client.call_async(EmptySrv.Request())
            start = time.time()
            while not future.done() and time.time() - start < 3.0:
                time.sleep(0.02)
            if not future.done():
                self.get_logger().warn(
                    'trigger_new_map did not respond in time -- anchoring '
                    'against the map as-is (may still include older data).')
        else:
            self.get_logger().warn(
                'rtabmap trigger_new_map service not available -- anchoring '
                'against the map as-is (may still include older data).')

        # Read ODOMETRY (odom->base_link), not RTAB-Map's own map->base_link --
        # the whole point of this anchor is to define "map" as a fixed
        # transform from wheel+IMU odometry alone, never touched again by
        # SLAM loop-closure/registration corrections (see
        # _broadcast_map_odom_tf's docstring for the full rationale: this is
        # what actually fixed the pose jitter/jumps, not just their display).
        try:
            odom_tf = self._tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            return None, f'no live odom->base_link TF yet ({type(e).__name__}) -- is the base driver running?'
        ot = odom_tf.transform.translation
        oq = odom_tf.transform.rotation
        odom_yaw = math.atan2(2 * (oq.w * oq.z + oq.x * oq.y), 1 - 2 * (oq.y * oq.y + oq.z * oq.z))

        yaw_a = math.atan2(hy - ay, hx - ax)
        self._broadcast_map_odom_tf(ax, ay, yaw_a, ot.x, ot.y, odom_yaw)

        with self.lock:
            # map frame is now DEFINED (by the broadcast above) to share
            # frame A's origin and orientation globally, differing only by
            # the fixed floorplan<->real-meters scale -- so unlike the old
            # per-anchor rotation+translation fit, this is now always the
            # same simple pure-scale transform regardless of where/how the
            # robot was anchored.
            transform = {'theta': 0.0, 'scale': FLOORPLAN_SCALE, 'tx': 0.0, 'ty': 0.0}
            self.calibration = transform
            # A path published under a PREVIOUS anchor is meaningless once
            # the anchor changes (different frame-B origin) -- ROS's
            # TRANSIENT_LOCAL durability means a fresh subscriber otherwise
            # immediately redelivers whatever /smoothed_path last published,
            # even from a session/anchor ago. Cleared here; a real path
            # reappears once an actual goal is planned under this anchor.
            self.planned_path = None
            self._displayed_pose_a = None  # also reset the pose smoother below
            # Same reasoning as planned_path above: accumulated scan points
            # are stored in map-frame meters, which this anchor just
            # redefined -- old points would be silently mis-registered
            # (a "wrong place" glitch, not just a boring stale display) if
            # carried across the anchor boundary. Builds back up fresh from
            # here for this session.
            self.accumulated_scan_cells = set()
            self._anchor_settle_until = time.time() + self._SCAN_ACCUM_SETTLE_S
        self.get_logger().info(
            f"Anchor set: floorplan=({ax:.2f},{ay:.2f}) heading={math.degrees(yaw_a):.0f}deg "
            f"<-> robot odom pose=({ot.x:.2f},{ot.y:.2f},"
            f"{math.degrees(odom_yaw):.0f}deg), scale={transform['scale']:.3f}")
        self._publish_floorplan_map()
        self._stop_nav_teleop()
        return transform, 'ok'

    def _stop_nav_teleop(self):
        """Kill nav_joystick_teleop (aloha_bringup.launch.py's
        teleop_twist_joy instance, remapped to /nav_cmd_vel for manual
        repositioning before an anchor is set) the moment an anchor is
        confirmed.

        Found 2026-07-22: it shares nav_deadman's enable button (L2) with
        the autonomous planner. Holding L2 for the deadman gate while NOT
        touching the drive stick makes teleop_twist_joy publish an all-zero
        Twist on every joystick message -- at the joystick's native poll
        rate, much faster than the planner's 10Hz -- straight onto the same
        /nav_cmd_vel topic the planner publishes real commands to. With no
        arbitration between the two publishers, nav_deadman mostly saw
        teleop's zeros (confirmed via [deadman-tick]/[control-tick] log
        cross-reference: 60-75% of ticks read target_lin=0.000 despite the
        planner's own log showing a clean continuous ~0.3 the whole time),
        which is what "smooth for a second, stutter, smooth again" actually
        was. An anchor means positioning is done and autonomous driving is
        about to start, so this is the one clean point to retire teleop for
        the rest of the session -- no need to relaunch it until the next
        fresh bringup.

        Matches by the node's ROS-args name (unique to this one process,
        set by aloha_bringup.launch.py's `name='nav_joystick_teleop'`), not
        a generic 'teleop' pattern, so it can't catch anything else.
        Best-effort: joy_node and nav_deadman are untouched either way, so
        even if this no-ops (already dead, or bringup used
        use_nav_teleop:=false), L2 still gates all motion as normal.
        """
        try:
            result = subprocess.run(
                ['pkill', '-f', '__node:=nav_joystick_teleop'],
                capture_output=True, timeout=2.0)
            if result.returncode == 0:
                self.get_logger().info(
                    'Anchor confirmed -- stopped nav_joystick_teleop so it '
                    "can't fight the planner on /nav_cmd_vel for the rest "
                    'of this session (joy_node + nav_deadman untouched, L2 '
                    'still gates all motion).')
        except Exception as e:
            self.get_logger().warn(
                f'_stop_nav_teleop: pkill failed ({type(e).__name__}: {e}) '
                '-- if motion is choppy, check for a stray nav_joystick_teleop '
                'process manually.')

    def reset_anchor(self):
        with self.lock:
            self.calibration = None
        self.get_logger().warn('Anchor cleared -- goals refused until re-anchored.')

    def _publish_floorplan_map(self):
        """Warp the static floorplan-derived obstacle grid (frame A pixels)
        into frame B (the robot's live SLAM/odom frame) via the current
        anchor, and publish it on /floorplan_map -- the map simple_nav_planner
        actually plans A* against (see set_anchor's docstring for why:
        floorplan walls, not noisy live SLAM/camera data). Live sensors still
        gate real-time collision avoidance separately, unaffected by this.
        """
        with self.lock:
            calib = self.calibration
        if calib is None:
            return
        bl = self.base_layer
        occ = self.floorplan_occ_px  # [row=y_px, col=x_px], 1=wall/boundary

        # frame-A pixel -> frame-A meters (inverse of the forward map used
        # for on-screen display: meters -> px via sx/sy/ox/oy).
        m_px_to_a = np.linalg.inv(_mat_scale_translate(
            bl['sx'], bl['sy'], bl['ox'], bl['oy']))
        # frame-A meters -> frame-B meters (inverse of apply_similarity).
        m_a_to_b = np.linalg.inv(_mat_similarity(calib))
        m_px_to_b = m_a_to_b @ m_px_to_a

        # Output resolution matched to what the rest of the stack already
        # uses (Grid/CellSize 0.05). Extent: bounding box of the source
        # image's 4 corners warped into frame B.
        resolution = 0.05
        h, w = occ.shape
        corners_px = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
        corners_b = m_px_to_b @ corners_px
        min_x, min_y = float(corners_b[0].min()), float(corners_b[1].min())
        max_x, max_y = float(corners_b[0].max()), float(corners_b[1].max())
        out_w = min(int(np.ceil((max_x - min_x) / resolution)) + 1, 6000)
        out_h = min(int(np.ceil((max_y - min_y) / resolution)) + 1, 6000)

        # frame-B meters -> output-grid pixels. Row0 = min-y, matching
        # OccupancyGrid/world_to_grid convention (NOT image row0=max-y).
        m_b_to_outpx = _mat_scale_translate(
            1.0 / resolution, 1.0 / resolution,
            -min_x / resolution, -min_y / resolution)
        m_px_to_outpx = m_b_to_outpx @ m_px_to_b
        m_outpx_to_px = np.linalg.inv(m_px_to_outpx)
        coeffs = (m_outpx_to_px[0, 0], m_outpx_to_px[0, 1], m_outpx_to_px[0, 2],
                  m_outpx_to_px[1, 0], m_outpx_to_px[1, 1], m_outpx_to_px[1, 2])
        src_img = Image.fromarray((occ * 255).astype(np.uint8), mode='L')
        out_img = src_img.transform(
            (out_w, out_h), Image.AFFINE, coeffs, resample=Image.NEAREST)
        out_arr = np.array(out_img)

        grid = np.zeros((out_h, out_w), dtype=np.int8)
        grid[out_arr > 127] = 100
        # Everything else is 0 (free), deliberately not -1/unknown -- the
        # floorplan is ground truth over its whole known extent, not a
        # partially-explored live map.

        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = resolution
        msg.info.width = out_w
        msg.info.height = out_h
        msg.info.origin.position.x = min_x
        msg.info.origin.position.y = min_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.floorplan_map_pub.publish(msg)
        self.get_logger().info(
            f'Published floorplan-derived obstacle map: {out_w}x{out_h} cells '
            f'@ {resolution}m, origin=({min_x:.2f},{min_y:.2f})'
        )


def create_app(node: NavWebViewerNode):
    app = Flask(__name__)

    @app.get('/')
    def index():
        return INDEX_HTML

    @app.get('/api/floorplan.png')
    def floorplan_png():
        return Response(node.base_layer['png'], mimetype='image/png')

    @app.get('/api/live_map.png')
    def live_map_png():
        png = node.render_live_map_png()
        if png is None:
            return Response(status=204)
        return Response(png, mimetype='image/png')

    @app.get('/api/live_scan.png')
    def live_scan_png():
        png = node.render_live_scan_png()
        if png is None:
            return Response(status=204)
        return Response(png, mimetype='image/png')

    @app.get('/api/state')
    def state():
        return jsonify(node.get_state())

    @app.post('/api/stage_goal')
    def stage_goal():
        body = request.get_json(force=True)
        x, y = float(body['x']), float(body['y'])
        yaw = float(body.get('yaw', 0.0))
        return jsonify(node.stage_goal(x, y, yaw))

    @app.post('/api/confirm_goal')
    def confirm_goal():
        ok, msg = node.confirm_goal()
        return jsonify({'ok': ok, 'message': msg})

    @app.post('/api/cancel')
    def cancel():
        node.cancel()
        return jsonify({'ok': True})

    @app.post('/api/anchor')
    def set_anchor():
        body = request.get_json(force=True)
        transform, msg = node.set_anchor(
            float(body['ax']), float(body['ay']), float(body['hx']), float(body['hy']))
        return jsonify({'ok': transform is not None, 'transform': transform, 'message': msg})

    @app.post('/api/anchor/reset')
    def reset_anchor():
        node.reset_anchor()
        return jsonify({'ok': True})

    @app.get('/api/locations')
    def search_locations():
        q = request.args.get('q', '')
        return jsonify(node.search_locations(q))

    @app.post('/api/goto_location')
    def goto_location():
        body = request.get_json(force=True)
        label = str(body.get('label', ''))
        result, msg = node.goto_location(label)
        if result is None:
            return jsonify({'ok': False, 'message': msg})
        result['ok'] = True
        return jsonify(result)

    return app


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Mobile ALOHA - Nav Viewer</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #1b1b1f; color: #eee; }
  #wrap { display: flex; flex-direction: column; align-items: center; padding: 12px; }
  #stage { position: relative; border: 1px solid #444; max-width: 95vw; max-height: 80vh; overflow: auto; }
  canvas { display: block; cursor: crosshair; }
  #zoomHud { display: flex; gap: 6px; align-items: center; }
  #zoomHud button { padding: 4px 10px; border-radius: 4px; border: none; cursor: pointer; background: #333; color: #eee; }
  #hud { margin: 8px 0; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  #stop {
    background: #c62828; color: white; border: none; border-radius: 6px;
    font-size: 20px; font-weight: bold; padding: 10px 28px; cursor: pointer;
  }
  #stop:hover { background: #e53935; }
  #confirmBox {
    display: none; background: #2a2a30; border: 1px solid #555; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0;
  }
  #confirmBox button { margin-right: 8px; padding: 6px 14px; border-radius: 4px; border: none; cursor: pointer; }
  #go { background: #2e7d32; color: white; }
  #cancelStage { background: #555; color: white; }
  .badge { padding: 2px 8px; border-radius: 4px; background: #333; font-size: 13px; }
  .badgebtn { padding: 2px 8px; border-radius: 4px; background: #333; font-size: 13px; color: #eee; border: none; cursor: pointer; }
  #anchorBanner {
    background: #6d4c00; color: #ffe082; padding: 8px 16px;
    border-radius: 6px; margin-bottom: 8px; font-size: 14px;
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  }
  #anchorBanner button { padding: 6px 14px; border-radius: 4px; border: none; cursor: pointer; }
  #anchorConfirm { background: #2e7d32; color: white; }
  #anchorRedo { background: #555; color: white; }
  #locationSearch { position: relative; width: 100%; max-width: 420px; margin-bottom: 8px; }
  #locInput {
    width: 100%; box-sizing: border-box; padding: 8px 10px; border-radius: 6px;
    border: 1px solid #555; background: #2a2a30; color: #eee; font-size: 14px;
  }
  #locResults {
    display: none; position: absolute; top: 100%; left: 0; right: 0; z-index: 5;
    background: #2a2a30; border: 1px solid #555; border-radius: 0 0 6px 6px;
    max-height: 260px; overflow-y: auto;
  }
  .locRow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 10px; border-top: 1px solid #3a3a40; font-size: 13px;
  }
  .locRow:first-child { border-top: none; }
  .locLabel { font-weight: bold; }
  .locCategory { color: #999; font-size: 11px; margin-left: 6px; }
  .locBtns button {
    margin-left: 6px; padding: 3px 9px; border-radius: 4px; border: none;
    cursor: pointer; font-size: 12px; background: #444; color: #eee;
  }
  .locBtns button:hover { background: #555; }
</style>
</head>
<body>
<div id="wrap">
  <div id="anchorBanner">
    <span id="anchorMsg">Click where the robot is right now.</span>
    <button id="anchorConfirm" style="display:none">Confirm Anchor</button>
    <button id="anchorRedo" style="display:none">Redo</button>
  </div>
  <div id="hud">
    <span class="badge" id="poseBadge">pose: --</span>
    <span class="badge" id="layerBadge">layer: --</span>
    <span class="badge" id="navStatusBadge">nav: --</span>
    <button class="badgebtn" id="reanchor">re-anchor</button>
    <div id="zoomHud">
      <button id="zoomOut">&minus;</button>
      <span class="badge" id="zoomLabel">100%</span>
      <button id="zoomIn">+</button>
      <button id="zoomReset">fit</button>
    </div>
    <button id="stop">STOP</button>
  </div>
  <div id="locationSearch">
    <input id="locInput" type="text" placeholder="Go to a room/desk, e.g. 314 or D3-12" autocomplete="off">
    <div id="locResults"></div>
  </div>
  <div id="confirmBox">
    Send robot to (<span id="cx"></span>, <span id="cy"></span>),
    ~<span id="cd"></span> m away<span id="cLabel"></span>?
    <button id="go">Confirm &amp; Go</button>
    <button id="cancelStage">Cancel</button>
  </div>
  <div id="stage">
    <canvas id="canvas"></canvas>
  </div>
</div>
<script>
let layer = null;   // base_layer info from /api/state
let pending = null; // {x, y} in map meters, staged locally before confirm

// Anchor flow -- two clicks, entirely local until "Confirm Anchor":
// 1st click = "the robot is here", 2nd click = "facing this way".
let anchorPos = null;
let anchorHead = null;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const floorplanImg = new Image();
let liveMapImg = new Image();
let liveScanImg = new Image();
let floorplanReady = false;

floorplanImg.onload = () => { floorplanReady = true; draw(); };
floorplanImg.src = '/api/floorplan.png';

function mapToPx(x, y) {
  return [x * layer.sx + layer.ox, y * layer.sy + layer.oy];
}
function pxToMap(px, py) {
  return [(px - layer.ox) / layer.sx, (py - layer.oy) / layer.sy];
}

// ---- named-location search ("go to <label>") ----
let locSearchTimer = null;
const locInput = document.getElementById('locInput');
const locResults = document.getElementById('locResults');

locInput.addEventListener('input', () => {
  clearTimeout(locSearchTimer);
  const q = locInput.value;
  if (!q.trim()) { locResults.style.display = 'none'; locResults.innerHTML = ''; return; }
  locSearchTimer = setTimeout(async () => {
    const r = await fetch('/api/locations?q=' + encodeURIComponent(q));
    const matches = await r.json();
    renderLocResults(matches);
  }, 200);
});

function renderLocResults(matches) {
  if (!matches.length) {
    locResults.innerHTML = '<div class="locRow">no matches</div>';
    locResults.style.display = 'block';
    return;
  }
  locResults.innerHTML = matches.map(m => `
    <div class="locRow">
      <span><span class="locLabel">${m.label}</span><span class="locCategory">${m.category}</span></span>
      <span class="locBtns">
        <button data-label="${m.label}">Go To</button>
      </span>
    </div>
  `).join('');
  locResults.style.display = 'block';
  locResults.querySelectorAll('button').forEach(btn => {
    btn.onclick = () => goToLocation(btn.dataset.label);
  });
}

async function goToLocation(label) {
  if (!(latestState && latestState.calibrated)) {
    alert('Not anchored yet -- click an anchor on the page first.');
    return;
  }
  const r = await fetch('/api/goto_location', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({label}),
  });
  const j = await r.json();
  if (!j.ok) { alert('Could not resolve "' + label + '": ' + j.message); return; }
  pending = { x: j.x, y: j.y };
  document.getElementById('cx').textContent = j.x.toFixed(2);
  document.getElementById('cy').textContent = j.y.toFixed(2);
  document.getElementById('cd').textContent = (j.distance_m != null ? j.distance_m.toFixed(1) : '?');
  document.getElementById('cLabel').textContent = ' (' + j.label + ')';
  document.getElementById('confirmBox').style.display = 'block';
  locResults.style.display = 'none';
  draw();
}

document.addEventListener('click', (ev) => {
  if (!locResults.contains(ev.target) && ev.target !== locInput) {
    locResults.style.display = 'none';
  }
});

let latestState = null;

function draw() {
  if (!floorplanReady || !layer) return;
  canvas.width = layer.width;
  canvas.height = layer.height;
  ctx.drawImage(floorplanImg, 0, 0, layer.width, layer.height);
  if (liveMapImg.complete && liveMapImg.naturalWidth > 0 && latestState && latestState.live_map) {
    // render_live_map_png already warps grid-px -> frame-B -> frame-A
    // (rotation included) -> base-layer px server-side, producing an image
    // the same size as the base layer -- draw it full-canvas as-is. (Fixed
    // 2026-07-22: this used to re-derive a bounding box client-side from
    // RAW frame-B origin numbers run through the frame-A-only pixel scale,
    // skipping the anchor's rotation entirely -- looked fine at theta~0 but
    // put the already-correctly-rotated image in the wrong place/shape for
    // any real anchor rotation, which is every anchor set in practice.)
    ctx.drawImage(liveMapImg, 0, 0, layer.width, layer.height);
  }
  if (liveScanImg.complete && liveScanImg.naturalWidth > 0) {
    // Live camera/sensor-detected obstacles (orange dots) -- separate from
    // the floorplan's static walls (red), rendered full-canvas the same way.
    ctx.drawImage(liveScanImg, 0, 0, layer.width, layer.height);
  }
  if (latestState && latestState.pose) {
    const [px, py] = mapToPx(latestState.pose.x, latestState.pose.y);
    ctx.fillStyle = '#ff3d3d';
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, 2 * Math.PI);
    ctx.fill();
    const hx = px + 20 * Math.cos(-latestState.pose.yaw);
    const hy = py + 20 * Math.sin(-latestState.pose.yaw);
    ctx.strokeStyle = '#ff3d3d';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(hx, hy);
    ctx.stroke();
  }
  if (anchorPos) {
    const [px, py] = mapToPx(anchorPos.x, anchorPos.y);
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(px, py, 10, 0, 2 * Math.PI);
    ctx.stroke();
    if (anchorHead) {
      const [hxp, hyp] = mapToPx(anchorHead.x, anchorHead.y);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(hxp, hyp);
      ctx.stroke();
    }
  }
  if (latestState && latestState.path && latestState.path.length > 1) {
    ctx.strokeStyle = '#ff9100';
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    latestState.path.forEach((pt, i) => {
      const [px, py] = mapToPx(pt.x, pt.y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (pending) {
    const [px, py] = mapToPx(pending.x, pending.y);
    ctx.strokeStyle = '#ffd600';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(px, py, 10, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

function updateAnchorUI() {
  const msg = document.getElementById('anchorMsg');
  const confirmBtn = document.getElementById('anchorConfirm');
  const redoBtn = document.getElementById('anchorRedo');
  if (!anchorPos) {
    msg.textContent = 'Click where the robot is right now.';
    confirmBtn.style.display = 'none';
    redoBtn.style.display = 'none';
  } else if (!anchorHead) {
    msg.textContent = 'Now click a point the robot is facing toward.';
    confirmBtn.style.display = 'none';
    redoBtn.style.display = 'inline-block';
  } else {
    msg.textContent = 'Anchor ready -- confirm, or redo if the click was off.';
    confirmBtn.style.display = 'inline-block';
    redoBtn.style.display = 'inline-block';
  }
}

async function refreshState() {
  const r = await fetch('/api/state');
  const s = await r.json();
  latestState = s;
  if (!layer) {
    layer = s.base_layer;
    document.getElementById('layerBadge').textContent = 'layer: ' + layer.type;
    applyZoom();
  }
  document.getElementById('poseBadge').textContent = s.pose
    ? `pose: x=${s.pose.x.toFixed(2)} y=${s.pose.y.toFixed(2)} yaw=${(s.pose.yaw*180/Math.PI).toFixed(0)}deg`
    : 'pose: unknown';
  const navStatusEl = document.getElementById('navStatusBadge');
  const [statusWord, statusReason] = (s.nav_status || 'idle').split(':');
  navStatusEl.textContent = 'nav: ' + statusWord + (statusReason ? ' (' + statusReason + ')' : '');
  navStatusEl.style.background = (statusWord === 'failed') ? '#c62828' : '#333';
  document.getElementById('anchorBanner').style.display = s.calibrated ? 'none' : 'flex';
  draw();
}

function refreshLiveMap() {
  const img = new Image();
  img.onload = () => { liveMapImg = img; draw(); };
  img.src = '/api/live_map.png?t=' + Date.now();
}

function refreshLiveScan() {
  const img = new Image();
  img.onload = () => { liveScanImg = img; draw(); };
  img.src = '/api/live_scan.png?t=' + Date.now();
}

// ---- Pan (drag) + zoom -- purely a CSS display-size change on top of the
// canvas's own full-resolution internal pixel grid, so all existing click
// coordinate math (getBoundingClientRect + canvas.width/rect.width) keeps
// working unmodified regardless of zoom level. Panning uses #stage's own
// native scroll (scrollLeft/scrollTop), not a custom transform.
const stageEl = document.getElementById('stage');
let viewScale = 1;
const ZOOM_MIN = 0.2, ZOOM_MAX = 4;

function applyZoom() {
  if (!layer) return;
  canvas.style.width = (layer.width * viewScale) + 'px';
  canvas.style.height = (layer.height * viewScale) + 'px';
  document.getElementById('zoomLabel').textContent = Math.round(viewScale * 100) + '%';
}
function setZoom(newScale, anchorClientX, anchorClientY) {
  newScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newScale));
  if (!layer) { viewScale = newScale; return; }
  // Keep the point under the cursor (if given) visually stationary.
  const rect = stageEl.getBoundingClientRect();
  const cx = anchorClientX != null ? anchorClientX - rect.left : rect.width / 2;
  const cy = anchorClientY != null ? anchorClientY - rect.top : rect.height / 2;
  const worldX = (stageEl.scrollLeft + cx) / viewScale;
  const worldY = (stageEl.scrollTop + cy) / viewScale;
  viewScale = newScale;
  applyZoom();
  stageEl.scrollLeft = worldX * viewScale - cx;
  stageEl.scrollTop = worldY * viewScale - cy;
}
document.getElementById('zoomIn').onclick = () => setZoom(viewScale * 1.25);
document.getElementById('zoomOut').onclick = () => setZoom(viewScale / 1.25);
document.getElementById('zoomReset').onclick = () => {
  if (!layer) return;
  const rect = stageEl.getBoundingClientRect();
  setZoom(Math.min(rect.width / layer.width, rect.height / layer.height, 1));
};
stageEl.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  setZoom(viewScale * (ev.deltaY < 0 ? 1.1 : 1 / 1.1), ev.clientX, ev.clientY);
}, { passive: false });

// Drag-to-pan vs. click-to-place: only treat it as a tap (anchor/goal
// click) if the pointer barely moved -- otherwise it was a pan, and must
// NOT stage a goal/anchor point.
let dragState = null; // {startX, startY, startScrollLeft, startScrollTop, moved}
const DRAG_THRESHOLD_PX = 6;

stageEl.addEventListener('mousedown', (ev) => {
  dragState = {
    startX: ev.clientX, startY: ev.clientY,
    startScrollLeft: stageEl.scrollLeft, startScrollTop: stageEl.scrollTop,
    moved: false,
  };
});
window.addEventListener('mousemove', (ev) => {
  if (!dragState) return;
  const dx = ev.clientX - dragState.startX;
  const dy = ev.clientY - dragState.startY;
  if (Math.hypot(dx, dy) > DRAG_THRESHOLD_PX) dragState.moved = true;
  if (dragState.moved) {
    stageEl.scrollLeft = dragState.startScrollLeft - dx;
    stageEl.scrollTop = dragState.startScrollTop - dy;
  }
});
window.addEventListener('mouseup', (ev) => {
  if (!dragState) return;
  if (!dragState.moved) handleTap(ev);
  dragState = null;
});

function handleTap(ev) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (ev.clientX - rect.left) * scaleX;
  const py = (ev.clientY - rect.top) * scaleY;
  if (px < 0 || py < 0 || px > canvas.width || py > canvas.height) return; // outside canvas
  const [mx, my] = pxToMap(px, py);

  if (!(latestState && latestState.calibrated)) {
    if (!anchorPos) {
      anchorPos = { x: mx, y: my };
    } else {
      anchorHead = { x: mx, y: my };
    }
    updateAnchorUI();
    draw();
    return;
  }

  pending = { x: mx, y: my };
  const dist = latestState && latestState.pose
    ? Math.hypot(mx - latestState.pose.x, my - latestState.pose.y).toFixed(1)
    : '?';
  document.getElementById('cx').textContent = mx.toFixed(2);
  document.getElementById('cy').textContent = my.toFixed(2);
  document.getElementById('cd').textContent = dist;
  document.getElementById('cLabel').textContent = '';
  document.getElementById('confirmBox').style.display = 'block';
  fetch('/api/stage_goal', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x: mx, y: my, yaw: 0.0}),
  });
  draw();
}

document.getElementById('anchorConfirm').onclick = async () => {
  const r = await fetch('/api/anchor', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ax: anchorPos.x, ay: anchorPos.y, hx: anchorHead.x, hy: anchorHead.y}),
  });
  const j = await r.json();
  if (!j.ok) { alert('Anchor failed: ' + j.message); return; }
  anchorPos = null; anchorHead = null;
  await refreshState();
};
document.getElementById('anchorRedo').onclick = () => {
  anchorPos = null; anchorHead = null;
  updateAnchorUI();
  draw();
};
document.getElementById('reanchor').onclick = async () => {
  await fetch('/api/anchor/reset', { method: 'POST' });
  anchorPos = null; anchorHead = null;
  pending = null;
  document.getElementById('confirmBox').style.display = 'none';
  updateAnchorUI();
  await refreshState();
};

document.getElementById('go').onclick = async () => {
  const r = await fetch('/api/confirm_goal', { method: 'POST' });
  const j = await r.json();
  if (!j.ok) { alert('Not sent: ' + j.message); return; }
  document.getElementById('confirmBox').style.display = 'none';
  pending = null;
  draw();
};
document.getElementById('cancelStage').onclick = async () => {
  await fetch('/api/cancel', { method: 'POST' });
  document.getElementById('confirmBox').style.display = 'none';
  pending = null;
  draw();
};
document.getElementById('stop').onclick = async () => {
  await fetch('/api/cancel', { method: 'POST' });
  document.getElementById('confirmBox').style.display = 'none';
  pending = null;
  draw();
};

updateAnchorUI();
refreshState();
// 2026-07-22: slowed down from 500/500/1500ms -- Flask's dev server spawns
// a fresh OS thread per request (not built for sustained polling), and at
// the old rate this measurably loaded the CPU (confirmed live: ~45-48%
// sustained) enough to plausibly steal scheduling time from
// simple_nav_planner's own 10Hz control loop on the same machine -- a real
// candidate for choppy motion having nothing to do with the planner logic.
setInterval(refreshState, 800);
setInterval(refreshLiveMap, 2000);
setInterval(refreshLiveScan, 1000);
</script>
</body>
</html>
"""


def main():
    rclpy.init()
    node = NavWebViewerNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = create_app(node)
    try:
        app.run(host=HOST, port=PORT, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
