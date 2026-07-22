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
import threading
import time
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, Response, jsonify, request

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,
)
import tf2_ros

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Empty
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
# only because it doesn't know the scale in advance). Value carried over from
# the old building-wide multi-point calibration this file used to compute.
FLOORPLAN_SCALE = 1.0849674771110611

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
        self.pose = None         # {'x':.., 'y':.., 'yaw':..} in frame B (raw TF) or None

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
        # must survive; this just drops stray 1px noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        occ_bin = cv2.morphologyEx(occ_bin, cv2.MORPH_OPEN, kernel)
        self.floorplan_occ_px = occ_bin  # frame-A PIXEL space, 1=wall/boundary, 0=free
        self.get_logger().info(
            f'Floorplan obstacle grid built from base layer: '
            f'{100 * occ_bin.mean():.1f}% marked occupied.')

        # ---- ROS I/O ----
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            OccupancyGrid, '/rtabmap/map', self._map_callback, map_qos)

        path_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.planned_path = None  # list of (x, y) in frame B, or None
        self.create_subscription(
            Path, '/smoothed_path', self._path_callback, path_qos)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
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
            }

    def render_live_map_png(self):
        """Warp the live occupancy grid directly into base-layer pixel space,
        composing grid-px->frame-B-meters, frame-B->frame-A (calibration),
        and frame-A-meters->base-layer-px into one affine (PIL resamples in
        one pass, so rotation from calibration comes along for free -- the
        frontend just draws the result full-canvas, no positioning math).
        """
        with self.lock:
            grid = self.live_grid
            calib = self.calibration
        if grid is None:
            return None
        arr = grid['array']
        rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
        unknown = arr < 0
        occupied = arr >= 50
        free = (~unknown) & (~occupied)
        rgba[free] = (60, 140, 255, 60)
        rgba[occupied] = (220, 30, 30, 190)
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

    # ---- single-click anchor (see compute_anchor_transform) ----

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

        with self.lock:
            pose = self.pose
        if pose is None:
            return None, 'no live robot pose yet -- is the mapping/nav stack running?'
        yaw_a = math.atan2(hy - ay, hx - ax)
        transform = compute_anchor_transform(
            ax, ay, yaw_a, pose['x'], pose['y'], pose['yaw'])
        with self.lock:
            self.calibration = transform
        self.get_logger().info(
            f"Anchor set: floorplan=({ax:.2f},{ay:.2f}) heading={math.degrees(yaw_a):.0f}deg "
            f"<-> robot frame-B pose=({pose['x']:.2f},{pose['y']:.2f},"
            f"{math.degrees(pose['yaw']):.0f}deg), scale={transform['scale']:.3f}")
        self._publish_floorplan_map()
        return transform, 'ok'

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

    return app


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Mobile ALOHA - Nav Viewer</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #1b1b1f; color: #eee; }
  #wrap { display: flex; flex-direction: column; align-items: center; padding: 12px; }
  #stage { position: relative; border: 1px solid #444; max-width: 95vw; overflow: auto; }
  canvas { display: block; cursor: crosshair; }
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
    <button class="badgebtn" id="reanchor">re-anchor</button>
    <button id="stop">STOP</button>
  </div>
  <div id="confirmBox">
    Send robot to (<span id="cx"></span>, <span id="cy"></span>),
    ~<span id="cd"></span> m away?
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
let floorplanReady = false;

floorplanImg.onload = () => { floorplanReady = true; draw(); };
floorplanImg.src = '/api/floorplan.png';

function mapToPx(x, y) {
  return [x * layer.sx + layer.ox, y * layer.sy + layer.oy];
}
function pxToMap(px, py) {
  return [(px - layer.ox) / layer.sx, (py - layer.oy) / layer.sy];
}

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
  }
  document.getElementById('poseBadge').textContent = s.pose
    ? `pose: x=${s.pose.x.toFixed(2)} y=${s.pose.y.toFixed(2)} yaw=${(s.pose.yaw*180/Math.PI).toFixed(0)}deg`
    : 'pose: unknown';
  document.getElementById('anchorBanner').style.display = s.calibrated ? 'none' : 'flex';
  draw();
}

function refreshLiveMap() {
  const img = new Image();
  img.onload = () => { liveMapImg = img; draw(); };
  img.src = '/api/live_map.png?t=' + Date.now();
}

canvas.addEventListener('click', (ev) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (ev.clientX - rect.left) * scaleX;
  const py = (ev.clientY - rect.top) * scaleY;
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
  document.getElementById('confirmBox').style.display = 'block';
  fetch('/api/stage_goal', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x: mx, y: my, yaw: 0.0}),
  });
  draw();
});

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
setInterval(refreshState, 500);
setInterval(refreshLiveMap, 1500);
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
