#!/usr/bin/env python3
"""
SVG-derived Floor Plan → ROS Occupancy Grid Map Converter
==========================================================
Optimized for the clean, vector-rendered PNG exported from TRIRIGA SVG.
"""

import cv2
import numpy as np
import yaml
import os

INPUT = "/mnt/user-data/uploads/floorplan_hires.png"
OUT = "/home/claude/occ_map_svg"
os.makedirs(OUT, exist_ok=True)

# ─── RESOLUTION ─────────────────────────────────────────────
# CALIBRATE: measure a known distance on the real floor (meters)
# and count the corresponding pixels in the cropped image.
# resolution = real_meters / pixel_count
RESOLUTION = 0.05  # meters/pixel — PLACEHOLDER, CALIBRATE!
INFLATION_PX = 5   # inflation buffer in pixels

print("=" * 65)
print("SVG Floor Plan → ROS Occupancy Grid (High-Res)")
print("=" * 65)

# ─── 1. LOAD ────────────────────────────────────────────────
print("\n[1] Loading...")
img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
print(f"  Size: {img.shape[1]} x {img.shape[0]} px")

# ─── 2. CROP to content ────────────────────────────────────
print("\n[2] Cropping to floor plan content...")
_, bw = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(bw)
x, y, w, h = cv2.boundingRect(coords)
pad = 30
y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
gray = img[y1:y2, x1:x2]
print(f"  Cropped: {gray.shape[1]} x {gray.shape[0]} px")
cv2.imwrite(os.path.join(OUT, "00_cropped.png"), gray)

# ─── 3. BINARY THRESHOLD ───────────────────────────────────
print("\n[3] Thresholding...")
# The SVG render has very clean black lines on white background
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# ─── 4. REMOVE TEXT LABELS ──────────────────────────────────
print("\n[4] Removing text labels...")
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

# Analyze component size distribution
areas = stats[1:, cv2.CC_STAT_AREA]
print(f"  Total components: {num_labels - 1}")
print(f"  Area range: {areas.min()} - {areas.max()}")
print(f"  Median area: {np.median(areas):.0f}")

# Strategy: text characters are small. Walls are large connected structures.
# Also: text chars tend to be compact (low aspect ratio), walls are elongated.
walls_only = np.zeros_like(binary)
furniture_and_walls = np.zeros_like(binary)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    w_comp = stats[i, cv2.CC_STAT_WIDTH]
    h_comp = stats[i, cv2.CC_STAT_HEIGHT]
    aspect = max(w_comp, h_comp) / max(min(w_comp, h_comp), 1)

    # ── Walls: large area OR very elongated (thin lines) ──
    # Wall segments are typically long thin structures
    is_wall = False
    if area > 500 and aspect > 6:     # long thin wall segment
        is_wall = True
    elif area > 2000:                   # large structure (thick wall, room boundary)
        is_wall = True

    # ── Furniture/obstacles: medium components that aren't text ──
    is_obstacle = False
    if area > 150:                      # bigger than text
        is_obstacle = True

    # ── Text: small, compact ──
    is_text = (area < 150) or (area < 400 and aspect < 3 and w_comp < 40 and h_comp < 40)

    if is_wall:
        walls_only[labels == i] = 255
    if is_obstacle and not is_text:
        furniture_and_walls[labels == i] = 255

# Additional wall extraction using morphological line detection
print("\n[5] Extracting structural lines...")
for length in [50, 35, 25]:
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    walls_only = cv2.bitwise_or(walls_only, h_lines)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    walls_only = cv2.bitwise_or(walls_only, v_lines)

# Close small gaps in walls (connect corners)
walls_only = cv2.morphologyEx(walls_only, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# Also apply line detection to the furniture+walls map
for length in [50, 35, 25]:
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    furniture_and_walls = cv2.bitwise_or(furniture_and_walls, h_lines)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    furniture_and_walls = cv2.bitwise_or(furniture_and_walls, v_lines)

furniture_and_walls = cv2.morphologyEx(furniture_and_walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

cv2.imwrite(os.path.join(OUT, "01_walls_binary.png"), walls_only)
cv2.imwrite(os.path.join(OUT, "02_full_binary.png"), furniture_and_walls)

# ─── 6. GENERATE OCCUPANCY GRIDS ───────────────────────────
print("\n[6] Generating occupancy grids...")

def to_occ(binary_obs):
    grid = np.full(binary_obs.shape, 254, dtype=np.uint8)
    grid[binary_obs > 0] = 0
    return grid

def inflate(binary_obs, radius):
    if radius <= 0: return binary_obs
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(binary_obs, k)

def save_map(grid, name, res):
    flipped = cv2.flip(grid, 0)  # ROS origin = bottom-left
    cv2.imwrite(os.path.join(OUT, f"{name}.pgm"), flipped)
    cv2.imwrite(os.path.join(OUT, f"{name}.png"), flipped)

    # Preview (not flipped, for human viewing)
    preview = np.full((*grid.shape, 3), 255, dtype=np.uint8)
    preview[grid == 0] = [0, 0, 0]
    cv2.imwrite(os.path.join(OUT, f"{name}_preview.png"), preview)

    yaml_cfg = {
        'image': f'{name}.pgm',
        'resolution': float(res),
        'origin': [0.0, 0.0, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }
    with open(os.path.join(OUT, f"{name}.yaml"), 'w') as f:
        yaml.dump(yaml_cfg, f, default_flow_style=False)

    occ = np.sum(grid == 0)
    total = grid.size
    print(f"  {name}: {grid.shape[1]}x{grid.shape[0]} px, "
          f"occupied={100*occ/total:.1f}%, "
          f"~{grid.shape[1]*res:.1f}x{grid.shape[0]*res:.1f}m")

# A: Walls only
save_map(to_occ(walls_only), "map_walls", RESOLUTION)

# B: Walls + furniture
save_map(to_occ(furniture_and_walls), "map_full", RESOLUTION)

# C: Walls inflated
save_map(to_occ(inflate(walls_only, INFLATION_PX)), "map_walls_inflated", RESOLUTION)

# D: Full inflated
save_map(to_occ(inflate(furniture_and_walls, INFLATION_PX)), "map_full_inflated", RESOLUTION)

# ─── 7. STATS & SUMMARY ────────────────────────────────────
print(f"""
{'='*65}
RESOLUTION: {RESOLUTION} m/px  (*** CALIBRATE THIS! ***)
  Current estimate: ~{gray.shape[1]*RESOLUTION:.0f}m x ~{gray.shape[0]*RESOLUTION:.0f}m
  
TO CALIBRATE:
  1. Measure a known real-world distance (hallway, room width)
  2. Open 00_cropped.png, measure same distance in pixels
  3. resolution = real_meters / pixels
  4. Update YAML files: change 'resolution: X.XX'

MAP FILES (each has .pgm + .yaml + .png + _preview.png):
  map_walls           — Structural walls only
  map_full            — Walls + furniture (conservative)
  map_walls_inflated  — Walls + {INFLATION_PX}px safety buffer
  map_full_inflated   — Full + {INFLATION_PX}px safety buffer

ROS2/Nav2:
  ros2 run nav2_map_server map_server \\
    --ros-args -p yaml_filename:=/path/to/map_walls.yaml
{'='*65}
""")

# List output files
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f:<40} {sz:>12,} bytes")
