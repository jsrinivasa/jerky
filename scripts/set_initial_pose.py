#!/usr/bin/env python3
"""
Interactive tool to set AMCL initial pose by clicking on the floor plan map.

Usage:
    # While AMCL is running:
    python3 set_initial_pose.py [--map-yaml PATH_TO_YAML]

    1. Click on the map where the robot is.
    2. Drag in the direction the robot is facing.
    3. The pose is published to /initialpose.
    4. Press 'r' to redo, 'q' to quit.
"""

import argparse
import math
import sys
import cv2
import numpy as np
import yaml

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseWithCovarianceStamped
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


def load_map(yaml_path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    from pathlib import Path
    pgm_path = Path(yaml_path).parent / meta['image']
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    res = float(meta['resolution'])
    ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
    return img, res, ox, oy


def px_to_map(px, py, img_h, res, ox, oy):
    """Convert pixel coords to map coords."""
    mx = px * res + ox
    my = (img_h - 1 - py) * res + oy
    return mx, my


def map_to_px(mx, my, img_h, res, ox, oy):
    """Convert map coords to pixel coords."""
    px = int((mx - ox) / res)
    py = int(img_h - 1 - (my - oy) / res)
    return px, py


class PosePublisher:
    def __init__(self):
        if HAS_ROS:
            rclpy.init()
            self.node = rclpy.create_node('set_initial_pose')
            self.pub = self.node.create_publisher(
                PoseWithCovarianceStamped, '/initialpose', 10
            )
        else:
            self.node = None
            self.pub = None

    def publish(self, x, y, yaw):
        if not HAS_ROS or self.pub is None:
            print(f"\n[NO ROS] Would publish: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
            print(f"  Launch args: initial_pose_x:={x:.3f} initial_pose_y:={y:.3f} "
                  f"initial_pose_yaw:={yaw:.3f}")
            return

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.covariance[0] = 0.25   # x variance
        msg.pose.covariance[7] = 0.25   # y variance
        msg.pose.covariance[35] = 0.07  # yaw variance
        self.pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.1)
        print(f"  Published to /initialpose")

    def destroy(self):
        if self.node:
            self.node.destroy_node()
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Set AMCL initial pose by clicking on the map.')
    parser.add_argument('--map-yaml', default='/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_walls.yaml')
    args = parser.parse_args()

    img, res, ox, oy = load_map(args.map_yaml)
    img_h, img_w = img.shape

    # Scale for display
    max_disp = 1200
    scale = min(max_disp / img_w, max_disp / img_h, 1.0)
    disp_w = int(img_w * scale)
    disp_h = int(img_h * scale)

    publisher = PosePublisher()

    click_pt = None   # (px, py) in original image coords
    drag_pt = None
    pose_set = False

    def draw():
        nonlocal pose_set
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw instructions
        cv2.putText(disp, "Click = position, Drag = heading. 'r'=redo 'q'=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6 / scale, (0, 0, 200), 2)

        if click_pt:
            cv2.circle(disp, click_pt, int(8 / scale), (0, 0, 255), -1)
            mx, my = px_to_map(click_pt[0], click_pt[1], img_h, res, ox, oy)
            cv2.putText(disp, f"({mx:.2f}, {my:.2f})",
                        (click_pt[0] + int(10/scale), click_pt[1] - int(10/scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 / scale, (0, 0, 255), 1)

        if click_pt and drag_pt:
            cv2.arrowedLine(disp, click_pt, drag_pt, (0, 200, 0), int(3 / scale), tipLength=0.3)
            dx = drag_pt[0] - click_pt[0]
            dy = drag_pt[1] - click_pt[1]
            # Image y is flipped vs map y
            yaw = math.atan2(-dy, dx)
            cv2.putText(disp, f"yaw={yaw:.2f} rad ({math.degrees(yaw):.0f} deg)",
                        (click_pt[0] + int(10/scale), click_pt[1] + int(20/scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 / scale, (0, 200, 0), 1)

        small = cv2.resize(disp, (disp_w, disp_h))
        cv2.imshow('Set Initial Pose', small)

    def mouse_cb(event, x, y, flags, param):
        nonlocal click_pt, drag_pt, pose_set
        # Convert display coords to original image coords
        ox_img = int(x / scale)
        oy_img = int(y / scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt = (ox_img, oy_img)
            drag_pt = None
            pose_set = False
            draw()
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            drag_pt = (ox_img, oy_img)
            draw()
        elif event == cv2.EVENT_LBUTTONUP:
            if click_pt and drag_pt:
                dx = drag_pt[0] - click_pt[0]
                dy = drag_pt[1] - click_pt[1]
                if math.sqrt(dx*dx + dy*dy) > 5:
                    mx, my = px_to_map(click_pt[0], click_pt[1], img_h, res, ox, oy)
                    yaw = math.atan2(-dy, dx)
                    print(f"\n=== Initial Pose ===")
                    print(f"  Map: x={mx:.3f}, y={my:.3f}, yaw={yaw:.3f} ({math.degrees(yaw):.1f} deg)")
                    publisher.publish(mx, my, yaw)
                    pose_set = True
                    draw()

    cv2.namedWindow('Set Initial Pose', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Set Initial Pose', mouse_cb)
    draw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            click_pt = None
            drag_pt = None
            pose_set = False
            draw()

    cv2.destroyAllWindows()
    publisher.destroy()


if __name__ == '__main__':
    main()
