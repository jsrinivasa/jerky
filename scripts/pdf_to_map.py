#!/usr/bin/env python3
"""
Convert a PDF floor plan into a ROS occupancy grid map (PGM + YAML).

Usage:
    # Generate a map preserving full visual detail (text, labels visible in RViz)
    python3 pdf_to_map.py floor_plan.pdf --resolution 0.05 --output my_map

    # Generate a clean walls-only map (no text, just walls and free space)
    python3 pdf_to_map.py floor_plan.pdf --resolution 0.05 --output my_map --mode clean

    # Preview the conversion (requires display)
    python3 pdf_to_map.py floor_plan.pdf --preview

    # Interactively measure a known distance to calculate resolution
    python3 pdf_to_map.py floor_plan.pdf --measure

Examples:
    python3 pdf_to_map.py building16.pdf --resolution 0.05 --output building16
    python3 pdf_to_map.py building16.pdf --resolution 0.05 --output building16 --mode clean
    python3 pdf_to_map.py building16.pdf --measure
"""

import argparse
import subprocess
import sys
import os
import tempfile

import cv2
import numpy as np


def pdf_to_image(pdf_path: str, dpi: int = 300) -> np.ndarray:
    """Convert first page of PDF to an image using pdftoppm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = os.path.join(tmpdir, "page")
        result = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf_path, out_prefix],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Error converting PDF: {result.stderr}")
            sys.exit(1)

        img_path = out_prefix + ".png"
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: could not read converted image at {img_path}")
            sys.exit(1)
        return img


def image_to_full_map(img: np.ndarray, border_pixels: int = 2) -> np.ndarray:
    """
    Convert a floor plan image to a grayscale map that preserves full visual
    detail. Text, labels, annotations, and shading remain visible in RViz.

    Uses 'scale' mode in the YAML so the map server preserves the grayscale
    range. Dark pixels become walls, white pixels become free space, and
    gray pixels get intermediate occupancy values.

    Args:
        img: Input BGR image
        border_pixels: Add a border of walls around the edges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add thin border
    if border_pixels > 0:
        gray[:border_pixels, :] = 0
        gray[-border_pixels:, :] = 0
        gray[:, :border_pixels] = 0
        gray[:, -border_pixels:] = 0

    return gray


def image_to_clean_map(img: np.ndarray, wall_threshold: int = 100,
                       border_pixels: int = 2) -> np.ndarray:
    """
    Convert a floor plan image to a clean occupancy grid with only walls
    and free space. Text and annotations are removed using morphological
    filtering (erode to strip thin features, dilate to restore wall width).

    - White (254) = free space
    - Black (0) = occupied / wall

    Args:
        img: Input BGR image
        wall_threshold: Pixels darker than this are potential walls (0-255)
        border_pixels: Add a border of walls around the edges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: only truly dark pixels are potential walls
    _, wall_mask = cv2.threshold(gray, wall_threshold, 255, cv2.THRESH_BINARY_INV)

    # Erode to remove thin features (text, dimension lines)
    kernel = np.ones((2, 2), np.uint8)
    walls_only = cv2.erode(wall_mask, kernel, iterations=1)

    # Dilate to restore wall thickness
    walls_restored = cv2.dilate(walls_only, kernel, iterations=2)

    # Create occupancy grid
    occupancy = np.full(gray.shape, 254, dtype=np.uint8)  # default: free
    occupancy[walls_restored > 0] = 0  # walls

    # Add border
    if border_pixels > 0:
        occupancy[:border_pixels, :] = 0
        occupancy[-border_pixels:, :] = 0
        occupancy[:, :border_pixels] = 0
        occupancy[:, -border_pixels:] = 0

    return occupancy


def write_map_files(map_image: np.ndarray, output_name: str, resolution: float,
                     mode: str = 'full',
                     origin_x: float = 0.0, origin_y: float = 0.0):
    """Write PGM and YAML map files.

    Args:
        map_image: Grayscale map image (from image_to_full_map or image_to_clean_map)
        output_name: Output filename prefix (without extension)
        resolution: Meters per pixel
        mode: 'full' uses scale mode (preserves grayscale detail),
              'clean' uses trinary mode (walls/free/unknown only)
        origin_x: Map origin X in meters (default: auto-center)
        origin_y: Map origin Y in meters (default: auto-center)
    """
    pgm_path = f"{output_name}.pgm"
    yaml_path = f"{output_name}.yaml"

    # Write PGM
    cv2.imwrite(pgm_path, map_image)

    # Center the origin so (0,0) is roughly in the middle of the map
    if origin_x == 0.0 and origin_y == 0.0:
        origin_x = -(map_image.shape[1] * resolution) / 2.0
        origin_y = -(map_image.shape[0] * resolution) / 2.0

    # Write YAML
    pgm_filename = os.path.basename(pgm_path)
    if mode == 'full':
        yaml_mode = 'scale'
        occupied_thresh = 0.65
        free_thresh = 0.50
    else:
        yaml_mode = 'trinary'
        occupied_thresh = 0.65
        free_thresh = 0.25

    with open(yaml_path, "w") as f:
        f.write(f"image: {pgm_filename}\n")
        f.write(f"mode: {yaml_mode}\n")
        f.write(f"resolution: {resolution}\n")
        f.write(f"origin: [{origin_x:.2f}, {origin_y:.2f}, 0.0]\n")
        f.write(f"negate: 0\n")
        f.write(f"occupied_thresh: {occupied_thresh}\n")
        f.write(f"free_thresh: {free_thresh}\n")

    print(f"\nMap files written:")
    print(f"  PGM: {os.path.abspath(pgm_path)}")
    print(f"  YAML: {os.path.abspath(yaml_path)}")
    print(f"\nMap info:")
    print(f"  Image size: {map_image.shape[1]} x {map_image.shape[0]} pixels")
    print(f"  Resolution: {resolution} m/pixel")
    print(f"  Real-world size: {map_image.shape[1] * resolution:.1f} x {map_image.shape[0] * resolution:.1f} meters")
    print(f"  Origin: ({origin_x:.2f}, {origin_y:.2f})")
    print(f"  Mode: {yaml_mode} ({'preserves visual detail' if mode == 'full' else 'clean walls only'})")
    print(f"\nTo use with navigation:")
    print(f"  ros2 launch aloha sim_navigation.launch.py map_file:={os.path.abspath(yaml_path)}")


def preview_mode(img: np.ndarray, map_image: np.ndarray, mode: str = 'full'):
    """Show the original image and map side by side."""
    max_display = 800
    h, w = img.shape[:2]
    scale = min(max_display / w, max_display / h, 1.0)
    display_size = (int(w * scale), int(h * scale))

    img_resized = cv2.resize(img, display_size)
    map_resized = cv2.resize(map_image, display_size, interpolation=cv2.INTER_NEAREST)
    map_color = cv2.cvtColor(map_resized, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([img_resized, map_color])

    print("\nPreview window opened.")
    print("  LEFT = original floor plan")
    print(f"  RIGHT = map output ({mode} mode)")
    print(f"\n  Image size: {w} x {h} pixels")
    print(f"\nTo calculate resolution:")
    print(f"  1. Measure a known distance in pixels (e.g., a hallway)")
    print(f"  2. resolution = real_meters / pixel_count")
    print(f"  Example: a 20m hallway is 400 pixels -> resolution = 20/400 = 0.05 m/pixel")
    print(f"\nPress any key to close the preview.")

    cv2.imshow("Floor Plan -> Map", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def measure_mode(img: np.ndarray):
    """Interactive mode to measure pixel distances for scale calculation."""
    points = []
    display_img = img.copy()

    max_display = 900
    h, w = img.shape[:2]
    scale = min(max_display / w, max_display / h, 1.0)
    display_size = (int(w * scale), int(h * scale))
    display_img = cv2.resize(display_img, display_size)

    print("\nMeasure mode:")
    print("  Click TWO points on a feature whose real-world length you know")
    print("  (e.g., a hallway, a room width, a door)")
    print("  Then enter the real-world distance in meters.")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(display_img, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Click two points", display_img)

    cv2.imshow("Click two points", display_img)
    cv2.setMouseCallback("Click two points", mouse_callback)

    print("\nClick the first point...")
    while len(points) < 2:
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            print("Cancelled.")
            return
        if len(points) == 1:
            print("Click the second point...")

    cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Calculate pixel distance (accounting for display scaling)
    px_dist = np.sqrt((points[1][0] - points[0][0])**2 +
                      (points[1][1] - points[0][1])**2)
    # Scale back to original image pixels
    px_dist_original = px_dist / scale

    print(f"\nPixel distance (in original image): {px_dist_original:.1f} pixels")

    try:
        real_dist = float(input("Enter the real-world distance in meters: "))
    except (ValueError, EOFError):
        print("Invalid input.")
        return

    resolution = real_dist / px_dist_original
    print(f"\n{'='*50}")
    print(f"  Calculated resolution: {resolution:.6f} m/pixel")
    print(f"  (Use --resolution {resolution:.4f} when generating the map)")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF floor plan to a ROS occupancy grid map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf", help="Path to the PDF floor plan")
    parser.add_argument("--output", "-o", default="map",
                       help="Output filename prefix (default: 'map')")
    parser.add_argument("--resolution", "-r", type=float, default=None,
                       help="Map resolution in meters/pixel. Required for final output.")
    parser.add_argument("--mode", "-m", choices=['full', 'clean'], default='full',
                       help="'full' preserves visual detail (text visible in RViz), "
                            "'clean' removes text and keeps only walls (default: full)")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for PDF rendering (default: 300)")
    parser.add_argument("--wall_threshold", type=int, default=100,
                       help="Pixel brightness threshold for walls in clean mode (0-255, default: 100)")
    parser.add_argument("--border", type=int, default=2,
                       help="Border wall thickness in pixels (default: 2)")
    parser.add_argument("--preview", action="store_true",
                       help="Show preview of the conversion (requires display)")
    parser.add_argument("--measure", action="store_true",
                       help="Interactive measurement mode to calculate resolution (requires display)")
    parser.add_argument("--origin_x", type=float, default=0.0,
                       help="Map origin X (default: auto-center)")
    parser.add_argument("--origin_y", type=float, default=0.0,
                       help="Map origin Y (default: auto-center)")

    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)

    print(f"Converting {args.pdf} at {args.dpi} DPI...")
    img = pdf_to_image(args.pdf, dpi=args.dpi)
    print(f"Image size: {img.shape[1]} x {img.shape[0]} pixels")

    if args.measure:
        measure_mode(img)
        return

    if args.mode == 'full':
        map_image = image_to_full_map(img, border_pixels=args.border)
    else:
        map_image = image_to_clean_map(img, wall_threshold=args.wall_threshold,
                                        border_pixels=args.border)

    if args.preview:
        preview_mode(img, map_image, mode=args.mode)
        return

    if args.resolution is None:
        print("\nError: --resolution is required to generate map files.")
        print("Use --preview to view the conversion, or --measure to calculate resolution.")
        print("\nExample: python3 pdf_to_map.py floor_plan.pdf --resolution 0.05 --output my_map")
        sys.exit(1)

    write_map_files(map_image, args.output, args.resolution,
                    mode=args.mode,
                    origin_x=args.origin_x, origin_y=args.origin_y)


if __name__ == "__main__":
    main()
