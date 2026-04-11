#!/usr/bin/env python3
"""
SVG Floor Plan → ROS Occupancy Grid Map Converter
==================================================
Converts an SVG (or PNG) building floor plan into a ROS-compatible
occupancy grid (PGM + YAML) ready for nav2_map_server and the
simple_nav_planner.

The pipeline:
  1. Render SVG to high-res grayscale image  (cairosvg)
  2. Auto-crop to content
  3. Threshold to binary
  4. Remove text labels via connected-component analysis
  5. Extract structural wall lines via morphological filtering
  6. Generate occupancy grids (walls-only, walls+furniture, inflated variants)
  7. Write PGM + YAML files with correct ROS origin convention

Usage:
    # Basic: generate all map variants from SVG
    python3 svg_to_map.py floorplan.svg -r 0.05 -o maps/building16

    # Preview before committing (requires display)
    python3 svg_to_map.py floorplan.svg --preview

    # Only walls, auto-centered origin
    python3 svg_to_map.py floorplan.svg -r 0.05 -o maps/building16 --mode walls

    # Interactive resolution measurement
    python3 svg_to_map.py floorplan.svg --measure

    # PNG input (skip SVG rendering)
    python3 svg_to_map.py floorplan.png -r 0.05 -o maps/building16
"""

import argparse
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────
# Module 1: Image Loading
# ─────────────────────────────────────────────────────────────────────

def strip_svg_text(svg_path: str) -> str:
    """Remove all <text> elements from an SVG and return modified XML string.

    TRIRIGA SVG floor plans embed room labels, area annotations, and
    directional markers (UP/DN) as <text> elements.  These render as
    dark pixels that the wall-detection pipeline mistakes for obstacles.
    Stripping them before rasterisation produces a much cleaner image
    with only structural lines.

    Args:
        svg_path: Path to the original SVG file.

    Returns:
        SVG content as a UTF-8 string with <text> elements removed.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    removed = 0
    for parent in root.iter():
        children_to_remove = []
        for child in parent:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'text':
                children_to_remove.append(child)
        for child in children_to_remove:
            parent.remove(child)
            removed += 1
    print(f"  Stripped {removed} text elements from SVG")
    return ET.tostring(root, encoding='unicode')


def load_image(input_path: str, dpi: int = 300,
               strip_text: bool = True) -> np.ndarray:
    """Load an SVG or raster image and return a grayscale numpy array.

    For SVG files, renders via cairosvg at the requested DPI.
    For raster files (PNG, PGM, BMP, …), loads directly with OpenCV.

    Args:
        input_path: Path to SVG or image file.
        dpi: Render resolution for SVG (ignored for raster).
        strip_text: If True and input is SVG, remove <text> elements
                    before rendering so labels don't appear as obstacles.

    Returns:
        Grayscale uint8 image (H×W).
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext == '.svg':
        try:
            import cairosvg
        except ImportError:
            print("Error: cairosvg is required for SVG input.")
            print("Install with:  pip install cairosvg")
            sys.exit(1)

        print(f"Rendering SVG at {dpi} DPI...")
        if strip_text:
            svg_string = strip_svg_text(input_path)
            png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'),
                                        dpi=dpi)
        else:
            png_data = cairosvg.svg2png(url=input_path, dpi=dpi)
        arr = np.frombuffer(png_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: cairosvg rendered the SVG but OpenCV could not decode it.")
            sys.exit(1)
    else:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: could not load image: {input_path}")
            sys.exit(1)

    print(f"  Image size: {img.shape[1]} x {img.shape[0]} pixels")
    return img


# ─────────────────────────────────────────────────────────────────────
# Module 2: Preprocessing (crop + threshold)
# ─────────────────────────────────────────────────────────────────────

def crop_to_content(gray: np.ndarray, padding: int = 30) -> np.ndarray:
    """Auto-crop to non-white content with padding.

    Finds the bounding box of all pixels darker than 240 and crops
    the image to that region plus a padding border.

    Args:
        gray: Grayscale input image.
        padding: Pixels of white border to keep around content.

    Returns:
        Cropped grayscale image.
    """
    _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(bw)
    if coords is None:
        print("Warning: image appears to be entirely white; skipping crop.")
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    y1 = max(0, y - padding)
    y2 = min(gray.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(gray.shape[1], x + w + padding)
    cropped = gray[y1:y2, x1:x2]
    print(f"  Cropped: {cropped.shape[1]} x {cropped.shape[0]} pixels")
    return cropped


def threshold_binary(gray: np.ndarray, thresh: int = 200) -> np.ndarray:
    """Threshold grayscale image to binary (dark pixels → 255).

    SVG renders have clean black lines on white; a threshold of ~200
    captures walls while ignoring anti-aliased edges.

    Args:
        gray: Cropped grayscale image.
        thresh: Brightness cutoff (0–255). Pixels darker than this
                become 255 (foreground) in the output.

    Returns:
        Binary uint8 image (0 or 255).
    """
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary


# ─────────────────────────────────────────────────────────────────────
# Module 3: Text Removal & Wall Extraction
# ─────────────────────────────────────────────────────────────────────

def classify_components(binary: np.ndarray,
                        wall_area_min: int = 500,
                        wall_aspect_min: float = 6.0,
                        wall_area_large: int = 2000,
                        text_area_max: int = 150,
                        text_area_compact: int = 400,
                        text_aspect_max: float = 3.0,
                        text_dim_max: int = 40,
                        obstacle_area_min: int = 150):
    """Classify connected components into walls, furniture, and text.

    Uses area and aspect-ratio heuristics:
      - Walls:     large area OR elongated (long thin segments)
      - Furniture:  medium-sized components that aren't text
      - Text:      small, compact blobs (room labels, numbers)

    Args:
        binary: Binary image (255 = foreground).
        wall_area_min:    Min area for elongated wall segments.
        wall_aspect_min:  Min aspect ratio to consider elongated.
        wall_area_large:  Area above which any component is a wall.
        text_area_max:    Components smaller than this are always text.
        text_area_compact: Upper area bound for compact-text check.
        text_aspect_max:  Max aspect ratio for compact text.
        text_dim_max:     Max width/height for compact text.
        obstacle_area_min: Min area to be a non-text obstacle.

    Returns:
        walls_only:          Binary mask of wall-classified pixels.
        furniture_and_walls: Binary mask of walls + furniture pixels.
        stats_summary:       Dict with component statistics.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    stats_summary = {
        'total_components': num_labels - 1,
        'area_min': int(areas.min()) if len(areas) else 0,
        'area_max': int(areas.max()) if len(areas) else 0,
        'area_median': float(np.median(areas)) if len(areas) else 0,
    }

    walls_only = np.zeros_like(binary)
    furniture_and_walls = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_comp = stats[i, cv2.CC_STAT_WIDTH]
        h_comp = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w_comp, h_comp) / max(min(w_comp, h_comp), 1)

        is_wall = (area > wall_area_min and aspect > wall_aspect_min) or \
                  (area > wall_area_large)

        is_text = (area < text_area_max) or \
                  (area < text_area_compact and aspect < text_aspect_max
                   and w_comp < text_dim_max and h_comp < text_dim_max)

        is_obstacle = area > obstacle_area_min

        if is_wall:
            walls_only[labels == i] = 255
        if is_obstacle and not is_text:
            furniture_and_walls[labels == i] = 255

    return walls_only, furniture_and_walls, stats_summary


def extract_structural_lines(binary: np.ndarray,
                             walls_mask: np.ndarray,
                             line_lengths=(50, 35, 25),
                             close_kernel_size: int = 5) -> np.ndarray:
    """Reinforce wall detection with morphological line extraction.

    Uses horizontal and vertical opening kernels of decreasing length
    to extract straight structural lines from the binary image, then
    OR-merges them into the wall mask.  A closing step reconnects
    small gaps at wall intersections/corners.

    Args:
        binary:            Full binary image (before text removal).
        walls_mask:        Current wall mask to augment (modified in place).
        line_lengths:      Sequence of kernel lengths to try.
        close_kernel_size: Size of the closing kernel for gap filling.

    Returns:
        Updated wall mask with structural lines added.
    """
    result = walls_mask.copy()
    for length in line_lengths:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        result = cv2.bitwise_or(result, h_lines)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        result = cv2.bitwise_or(result, v_lines)

    if close_kernel_size > 0:
        k = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k)

    return result


# ─────────────────────────────────────────────────────────────────────
# Module 4: Occupancy Grid Generation
# ─────────────────────────────────────────────────────────────────────

def binary_to_occupancy(binary_obs: np.ndarray) -> np.ndarray:
    """Convert a binary obstacle mask to a ROS occupancy image.

    In the ROS convention for map images:
      254 = free space
      0   = occupied / wall

    Args:
        binary_obs: Binary mask (255 = obstacle).

    Returns:
        Occupancy grid image (uint8).
    """
    grid = np.full(binary_obs.shape, 254, dtype=np.uint8)
    grid[binary_obs > 0] = 0
    return grid


def inflate_obstacles(binary_obs: np.ndarray, radius_px: int) -> np.ndarray:
    """Dilate obstacles by a pixel radius to create a safety buffer.

    Uses an elliptical structuring element so the inflation is
    roughly circular (matching the round robot footprint).

    Args:
        binary_obs: Binary obstacle mask (255 = obstacle).
        radius_px:  Inflation radius in pixels.

    Returns:
        Inflated binary mask.
    """
    if radius_px <= 0:
        return binary_obs
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
    )
    return cv2.dilate(binary_obs, k)


# ─────────────────────────────────────────────────────────────────────
# Module 5: Map File I/O
# ─────────────────────────────────────────────────────────────────────

def save_map(grid: np.ndarray, output_prefix: str, resolution: float,
             origin_x: float = None, origin_y: float = None,
             save_preview: bool = True):
    """Write PGM, YAML, and optional preview PNG for a ROS map.

    The grid is flipped vertically because ROS map_server expects
    origin at the bottom-left, but image row 0 is the top.

    Args:
        grid:          Occupancy image (254=free, 0=occupied).
        output_prefix: File path prefix (e.g. 'maps/building16_walls').
        resolution:    Meters per pixel.
        origin_x:      Map origin X in meters (None = auto-center).
        origin_y:      Map origin Y in meters (None = auto-center).
        save_preview:  Write a human-readable preview PNG (not flipped).
    """
    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)

    pgm_path = f"{output_prefix}.pgm"
    yaml_path = f"{output_prefix}.yaml"

    # ROS origin = bottom-left → flip vertically
    flipped = cv2.flip(grid, 0)
    cv2.imwrite(pgm_path, flipped)

    # Also save a PNG copy (some map_server builds prefer it)
    cv2.imwrite(f"{output_prefix}.png", flipped)

    # Auto-center the origin if not specified
    if origin_x is None:
        origin_x = -(grid.shape[1] * resolution) / 2.0
    if origin_y is None:
        origin_y = -(grid.shape[0] * resolution) / 2.0

    yaml_cfg = {
        'image': os.path.basename(pgm_path),
        'resolution': float(resolution),
        'origin': [float(origin_x), float(origin_y), 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_cfg, f, default_flow_style=False)

    if save_preview:
        preview = np.full((*grid.shape, 3), 255, dtype=np.uint8)
        preview[grid == 0] = [0, 0, 0]
        cv2.imwrite(f"{output_prefix}_preview.png", preview)

    occ_pct = 100 * np.sum(grid == 0) / grid.size
    real_w = grid.shape[1] * resolution
    real_h = grid.shape[0] * resolution
    print(f"  {os.path.basename(output_prefix)}: "
          f"{grid.shape[1]}x{grid.shape[0]} px, "
          f"occupied={occ_pct:.1f}%, "
          f"~{real_w:.1f}x{real_h:.1f}m")

    return yaml_path


# ─────────────────────────────────────────────────────────────────────
# Module 6: Interactive Tools (measure / preview)
# ─────────────────────────────────────────────────────────────────────

def measure_mode(gray: np.ndarray):
    """Interactive mode: click two points to compute resolution.

    Opens a window showing the cropped floor plan.  The user clicks
    two points on a feature with a known real-world length, then
    enters the distance in meters.  The tool prints the computed
    resolution (m/pixel).
    """
    points = []
    max_display = 900
    h, w = gray.shape[:2]
    scale = min(max_display / w, max_display / h, 1.0)
    display_size = (int(w * scale), int(h * scale))
    display_img = cv2.cvtColor(cv2.resize(gray, display_size), cv2.COLOR_GRAY2BGR)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(display_img, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Click two points", display_img)

    print("\nMeasure mode: click TWO points on a known real-world distance.")
    cv2.imshow("Click two points", display_img)
    cv2.setMouseCallback("Click two points", mouse_cb)

    while len(points) < 2:
        key = cv2.waitKey(100)
        if key == 27:
            cv2.destroyAllWindows()
            print("Cancelled.")
            return
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    px_dist = np.sqrt((points[1][0] - points[0][0]) ** 2 +
                      (points[1][1] - points[0][1]) ** 2) / scale
    print(f"\nPixel distance (original image): {px_dist:.1f} px")

    try:
        real_dist = float(input("Enter the real-world distance in meters: "))
    except (ValueError, EOFError):
        print("Invalid input.")
        return

    resolution = real_dist / px_dist
    print(f"\n{'=' * 50}")
    print(f"  Calculated resolution: {resolution:.6f} m/pixel")
    print(f"  Use:  --resolution {resolution:.4f}")
    print(f"{'=' * 50}")


def preview_mode(gray: np.ndarray, walls: np.ndarray, full: np.ndarray):
    """Show original, walls-only, and full side by side."""
    max_display = 500
    h, w = gray.shape[:2]
    scale = min(max_display / w, max_display / h, 1.0)
    sz = (int(w * scale), int(h * scale))

    panels = []
    for img, label in [(gray, "Original"),
                       (binary_to_occupancy(walls), "Walls Only"),
                       (binary_to_occupancy(full), "Walls+Furniture")]:
        resized = cv2.resize(img, sz, interpolation=cv2.INTER_NEAREST)
        color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(color, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        panels.append(color)

    combined = np.hstack(panels)
    cv2.imshow("SVG Map Preview (press any key to close)", combined)
    print("\nPreview window opened. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────
# Module 7: SVG Room / Landmark Extraction
# ─────────────────────────────────────────────────────────────────────

def extract_rooms_from_svg(svg_path: str, resolution: float,
                           image_shape: tuple,
                           crop_offset: tuple = (0, 0),
                           origin_x: float = None,
                           origin_y: float = None) -> list:
    """Parse an SVG floor plan and extract room labels with map coordinates.

    Reads every <text> element from the SVG, converts its SVG coordinate
    to the ROS map frame (meters), and returns a list of room dicts.

    The coordinate mapping is:
      1. SVG uses a coordinate system where Y is negative (up).
         cairosvg renders with the SVG viewBox mapped to the output
         image.  We compute the pixel position of each label in the
         rendered (pre-crop) image.
      2. Crop offset is subtracted to get pixel coords in the cropped
         image.
      3. Pixel coords are converted to meters using resolution.
      4. The ROS map is vertically flipped (origin at bottom-left),
         so map_y = (image_height - pixel_y) * resolution + origin_y.

    Args:
        svg_path:     Path to the SVG file.
        resolution:   Meters per pixel of the output map.
        image_shape:  (height, width) of the cropped image in pixels.
        crop_offset:  (x_offset, y_offset) pixels removed by cropping.
        origin_x:     Map origin X in meters (None = auto-center).
        origin_y:     Map origin Y in meters (None = auto-center).

    Returns:
        List of dicts with keys:
          label, map_x, map_y, svg_x, svg_y, tri_id, category
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # ── Parse SVG viewBox to map SVG coords → pixel coords ──
    viewbox = root.get('viewBox', '')
    if not viewbox:
        print("Warning: SVG has no viewBox; room coordinates may be inaccurate.")
        return []

    vb_parts = viewbox.split()
    vb_x, vb_y, vb_w, vb_h = (float(v) for v in vb_parts)

    # The rendered image size (before crop) comes from cairosvg.
    # We need to know it to map SVG coords → pixel coords.
    # Re-render at same DPI to get the uncropped dimensions.
    try:
        import cairosvg
        png_data = cairosvg.svg2png(url=svg_path, dpi=300)
        arr = np.frombuffer(png_data, dtype=np.uint8)
        full_img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        full_h, full_w = full_img.shape[:2]
    except Exception:
        # Fallback: assume image_shape is the full image
        full_h, full_w = image_shape
        crop_offset = (0, 0)

    # SVG coord → pixel coord
    sx = full_w / vb_w   # pixels per SVG unit (horizontal)
    sy = full_h / vb_h   # pixels per SVG unit (vertical)

    # Map origin defaults
    img_h, img_w = image_shape
    if origin_x is None:
        origin_x = -(img_w * resolution) / 2.0
    if origin_y is None:
        origin_y = -(img_h * resolution) / 2.0

    # ── Extract text elements ──
    rooms = []
    for el in root.iter():
        tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
        if tag != 'text':
            continue

        label = (el.text or el.get('text', '') or '').strip()
        if not label:
            continue

        svg_x = float(el.get('x', '0'))
        svg_y = float(el.get('y', '0'))
        tri_id = el.get('tri-element-name', '')

        # SVG coord → pixel in full (uncropped) image
        px = (svg_x - vb_x) * sx
        py = (svg_y - vb_y) * sy

        # Adjust for crop
        px -= crop_offset[0]
        py -= crop_offset[1]

        # Pixel → ROS map meters
        # ROS map is flipped vertically: map_y increases upward
        map_x = px * resolution + origin_x
        map_y = (img_h - py) * resolution + origin_y

        # Categorize
        label_upper = label.upper()
        if label.endswith('sqm'):
            category = 'area_annotation'
        elif any(kw in label_upper for kw in ['STAIR', 'ELEV', 'UP', 'DN']):
            category = 'vertical_transport'
        elif 'CONF' in label_upper:
            category = 'conference_room'
        elif any(kw in label_upper for kw in ['MEN', 'WOMEN', 'RESTROOM']):
            category = 'restroom'
        elif any(kw in label_upper for kw in ['STOR', 'JAN', 'IDF']):
            category = 'utility'
        elif label_upper.startswith('SH-'):
            category = 'shared_space'
        elif label_upper.startswith('C-'):
            category = 'corridor'
        elif label.replace('-', '').replace(' ', '').replace('A', '').replace('B', '').replace('C', '').isdigit():
            category = 'numbered_room'
        else:
            category = 'named_space'

        rooms.append({
            'label': label,
            'map_x': round(float(map_x), 3),
            'map_y': round(float(map_y), 3),
            'svg_x': round(svg_x, 1),
            'svg_y': round(svg_y, 1),
            'tri_id': tri_id,
            'category': category,
        })

    # De-duplicate: keep only one entry per label (prefer the one with tri_id)
    seen = {}
    for r in rooms:
        key = r['label']
        if key not in seen or (r['tri_id'] and not seen[key]['tri_id']):
            seen[key] = r
    rooms = sorted(seen.values(), key=lambda r: r['label'])

    return rooms


def save_room_locations(rooms: list, output_path: str):
    """Save extracted room locations to a JSON file.

    The JSON can be loaded by the navigation mission planner to
    look up goal poses by room name.

    Args:
        rooms:       List of room dicts from extract_rooms_from_svg().
        output_path: Path to write the JSON file.
    """
    # Filter out area annotations (sqm) — not useful as nav targets
    nav_rooms = [r for r in rooms if r['category'] != 'area_annotation']

    with open(output_path, 'w') as f:
        json.dump(nav_rooms, f, indent=2)

    # Summary by category
    from collections import Counter
    cats = Counter(r['category'] for r in nav_rooms)
    print(f"  Saved {len(nav_rooms)} room locations to {output_path}")
    for cat, count in sorted(cats.items()):
        print(f"    {cat:25s}: {count}")

    return nav_rooms


def lookup_rooms(rooms_json_path: str, category: str = None,
                 name: str = None) -> list:
    """Look up rooms from a saved rooms JSON file.

    Filter by category, partial name match, or both.  Useful for
    querying navigation targets programmatically.

    Args:
        rooms_json_path: Path to the rooms JSON file.
        category:        Filter by category (e.g. 'conference_room',
                         'restroom', 'corridor', 'numbered_room',
                         'shared_space', 'vertical_transport', 'utility',
                         'named_space').  None returns all.
        name:            Substring match on label (case-insensitive).
                         None skips name filtering.

    Returns:
        List of matching room dicts with keys:
          label, map_x, map_y, svg_x, svg_y, tri_id, category
    """
    with open(rooms_json_path, 'r') as f:
        rooms = json.load(f)

    if category:
        rooms = [r for r in rooms if r['category'] == category]
    if name:
        name_lower = name.lower()
        rooms = [r for r in rooms if name_lower in r['label'].lower()]

    return rooms


# ─────────────────────────────────────────────────────────────────────
# Module 8: Full Pipeline
# ─────────────────────────────────────────────────────────────────────

def convert(input_path: str, output_prefix: str, resolution: float,
            dpi: int = 300, mode: str = 'all',
            inflation_px: int = 5, wall_threshold: int = 200,
            origin_x: float = None, origin_y: float = None,
            crop_padding: int = 30, extract_rooms: bool = False):
    """Run the full SVG/PNG → ROS map conversion pipeline.

    This is the main entry point for programmatic use.

    Args:
        input_path:     Path to SVG or PNG file.
        output_prefix:  Output path prefix (e.g. 'maps/building16').
        resolution:     Meters per pixel.
        dpi:            Render DPI for SVG input.
        mode:           'walls', 'full', or 'all' (generates both + inflated).
        inflation_px:   Pixel radius for obstacle inflation.
        wall_threshold: Binary threshold for wall detection (0–255).
        origin_x:       Map origin X (None = auto-center).
        origin_y:       Map origin Y (None = auto-center).
        crop_padding:   Padding pixels around content after crop.
        extract_rooms:  If True and input is SVG, extract room labels to JSON.

    Returns:
        List of generated YAML file paths.
    """
    print("=" * 65)
    print("SVG/PNG Floor Plan → ROS Occupancy Grid")
    print("=" * 65)

    # Step 1: Load
    print("\n[1] Loading image...")
    gray = load_image(input_path, dpi=dpi)

    # Step 2: Crop
    print("\n[2] Cropping to content...")
    cropped = crop_to_content(gray, padding=crop_padding)

    # Step 3: Threshold
    print("\n[3] Binary thresholding...")
    binary = threshold_binary(cropped, thresh=wall_threshold)

    # Step 4: Classify components (remove text)
    print("\n[4] Removing text labels...")
    walls_only, furniture_and_walls, comp_stats = classify_components(binary)
    print(f"  Components: {comp_stats['total_components']}, "
          f"area range: {comp_stats['area_min']}–{comp_stats['area_max']}, "
          f"median: {comp_stats['area_median']:.0f}")

    # Step 5: Structural line extraction
    print("\n[5] Extracting structural lines...")
    walls_only = extract_structural_lines(binary, walls_only)
    furniture_and_walls = extract_structural_lines(
        binary, furniture_and_walls, close_kernel_size=3
    )

    # Step 6: Generate and save maps
    print("\n[6] Generating occupancy grids...")
    yaml_files = []

    if mode in ('walls', 'all'):
        yf = save_map(binary_to_occupancy(walls_only),
                       f"{output_prefix}_walls", resolution,
                       origin_x, origin_y)
        yaml_files.append(yf)

    if mode in ('full', 'all'):
        yf = save_map(binary_to_occupancy(furniture_and_walls),
                       f"{output_prefix}_full", resolution,
                       origin_x, origin_y)
        yaml_files.append(yf)

    if mode == 'all' and inflation_px > 0:
        yf = save_map(
            binary_to_occupancy(inflate_obstacles(walls_only, inflation_px)),
            f"{output_prefix}_walls_inflated", resolution,
            origin_x, origin_y)
        yaml_files.append(yf)

        yf = save_map(
            binary_to_occupancy(inflate_obstacles(furniture_and_walls, inflation_px)),
            f"{output_prefix}_full_inflated", resolution,
            origin_x, origin_y)
        yaml_files.append(yf)

    # Step 7: Save debug images
    debug_dir = os.path.dirname(output_prefix) or '.'
    cv2.imwrite(os.path.join(debug_dir,
                os.path.basename(output_prefix) + "_cropped.png"), cropped)

    # Step 7b: Extract room locations from SVG
    rooms_file = None
    if extract_rooms and os.path.splitext(input_path)[1].lower() == '.svg':
        print("\n[7] Extracting room locations from SVG...")
        # Compute crop offset (how many pixels were trimmed from top-left)
        # Re-render full image to find the crop bounding box
        full_gray = load_image(input_path, dpi=dpi, strip_text=False)
        _, bw = cv2.threshold(full_gray, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(bw)
        if coords is not None:
            x_off, y_off, _, _ = cv2.boundingRect(coords)
            x_off = max(0, x_off - crop_padding)
            y_off = max(0, y_off - crop_padding)
        else:
            x_off, y_off = 0, 0

        # Use same origin as the saved maps
        eff_origin_x = origin_x if origin_x is not None else -(cropped.shape[1] * resolution) / 2.0
        eff_origin_y = origin_y if origin_y is not None else -(cropped.shape[0] * resolution) / 2.0

        rooms = extract_rooms_from_svg(
            svg_path=input_path,
            resolution=resolution,
            image_shape=(cropped.shape[0], cropped.shape[1]),
            crop_offset=(x_off, y_off),
            origin_x=eff_origin_x,
            origin_y=eff_origin_y,
        )
        rooms_file = f"{output_prefix}_rooms.json"
        save_room_locations(rooms, rooms_file)

    # Summary
    print(f"\n{'=' * 65}")
    print(f"Resolution: {resolution} m/pixel")
    print(f"Real-world size: ~{cropped.shape[1]*resolution:.1f} x "
          f"{cropped.shape[0]*resolution:.1f} m")
    print(f"\nGenerated maps:")
    for yf in yaml_files:
        print(f"  {yf}")
    if rooms_file:
        print(f"\nRoom locations: {rooms_file}")
        # Show some navigable rooms
        nav_rooms = [r for r in rooms if r['category'] not in ('area_annotation',)]
        print(f"  {len(nav_rooms)} navigable locations extracted")
        print(f"  Examples:")
        for r in nav_rooms[:8]:
            print(f"    {r['label']:20s} → ({r['map_x']:7.2f}, {r['map_y']:7.2f}) m  [{r['category']}]")
        if len(nav_rooms) > 8:
            print(f"    ... and {len(nav_rooms) - 8} more")
    print(f"\nTo launch navigation:")
    print(f"  ros2 launch aloha navigate_mission.launch.py "
          f"map_file:={os.path.abspath(yaml_files[0])}")
    print(f"{'=' * 65}")

    return yaml_files


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert an SVG or PNG floor plan to ROS occupancy grid maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to SVG or PNG floor plan")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path prefix (default: same dir/name as input)")
    parser.add_argument("-r", "--resolution", type=float, default=None,
                        help="Map resolution in meters/pixel (required for output)")
    parser.add_argument("-m", "--mode", choices=['walls', 'full', 'all'],
                        default='all',
                        help="'walls' = structural walls only, "
                             "'full' = walls + furniture, "
                             "'all' = both + inflated variants (default: all)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="SVG render DPI (default: 300)")
    parser.add_argument("--wall-threshold", type=int, default=200,
                        help="Binary threshold for wall detection 0-255 (default: 200)")
    parser.add_argument("--inflation", type=int, default=5,
                        help="Inflation radius in pixels (default: 5)")
    parser.add_argument("--crop-padding", type=int, default=30,
                        help="Padding around content after crop in pixels (default: 30)")
    parser.add_argument("--origin-x", type=float, default=None,
                        help="Map origin X in meters (default: auto-center)")
    parser.add_argument("--origin-y", type=float, default=None,
                        help="Map origin Y in meters (default: auto-center)")
    parser.add_argument("--preview", action="store_true",
                        help="Show preview of the conversion (requires display)")
    parser.add_argument("--measure", action="store_true",
                        help="Interactive mode to measure resolution (requires display)")
    parser.add_argument("--extract-rooms", action="store_true",
                        help="Extract room labels and coordinates from SVG to JSON")
    parser.add_argument("--no-strip-text", action="store_true",
                        help="Do NOT strip text from SVG before rendering "
                             "(keeps labels visible but may pollute wall detection)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    # Compute default output prefix from input filename
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base

    # For measure/preview, we only need to load and crop
    if args.measure or args.preview:
        gray = load_image(args.input, dpi=args.dpi)
        cropped = crop_to_content(gray, padding=args.crop_padding)

        if args.measure:
            measure_mode(cropped)
            return

        if args.preview:
            binary = threshold_binary(cropped, thresh=args.wall_threshold)
            walls, full, _ = classify_components(binary)
            walls = extract_structural_lines(binary, walls)
            full = extract_structural_lines(binary, full, close_kernel_size=3)
            preview_mode(cropped, walls, full)
            return

    # Full conversion requires resolution
    if args.resolution is None:
        print("Error: --resolution is required to generate map files.")
        print("Use --measure to interactively determine resolution.")
        print("Example: python3 svg_to_map.py floor.svg -r 0.05 -o maps/building")
        sys.exit(1)

    convert(
        input_path=args.input,
        output_prefix=args.output,
        resolution=args.resolution,
        dpi=args.dpi,
        mode=args.mode,
        inflation_px=args.inflation,
        wall_threshold=args.wall_threshold,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
        crop_padding=args.crop_padding,
        extract_rooms=args.extract_rooms,
    )


if __name__ == "__main__":
    main()
