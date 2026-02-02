#!/usr/bin/env python3

import argparse
import os
import time

import cv2
import h5py
import numpy as np


def _get_camera_names(f: h5py.File):
    if 'observations' not in f or 'images' not in f['observations']:
        raise KeyError("Missing group 'observations/images' in HDF5 file")
    return list(f['observations/images'].keys())


def _num_frames(f: h5py.File, cam_name: str):
    return int(f['observations/images'][cam_name].shape[0])


def _decode_frame(f: h5py.File, cam_name: str, cam_idx: int, t: int):
    compress = bool(f.attrs.get('compress', False))
    if not compress:
        img = f['observations/images'][cam_name][t]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Unexpected image shape for {cam_name}: {img.shape}")
        return img

    padded = f['observations/images'][cam_name][t]
    if 'compress_len' not in f:
        raise KeyError("File is marked compress=True but missing dataset 'compress_len'")
    true_len = int(f['compress_len'][cam_idx, t])
    jpg_bytes = padded[:true_len].tobytes()
    img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to decode JPEG for cam={cam_name} frame={t} (len={true_len})")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to episode_*.hdf5')
    parser.add_argument('--camera', type=str, default=None, help='Camera name (e.g. cam_high)')
    parser.add_argument('--list', action='store_true', help='List cameras and exit')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--end', type=int, default=None, help='End frame index (exclusive)')
    parser.add_argument('--fps', type=float, default=30.0, help='Playback fps')
    parser.add_argument('--step', action='store_true', help='Step frame-by-frame (keys: a/d, q)')
    parser.add_argument('--save_dir', type=str, default=None, help='If set, saves frames as PNG to this dir')
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(args.path)

    with h5py.File(args.path, 'r') as f:
        cam_names = _get_camera_names(f)
        if args.list:
            print('Cameras:')
            for name in cam_names:
                ds = f['observations/images'][name]
                print(f"- {name}: shape={ds.shape} dtype={ds.dtype}")
            print(f"compress={bool(f.attrs.get('compress', False))}")
            return

        cam_name = args.camera or (cam_names[0] if cam_names else None)
        if cam_name is None:
            raise ValueError('No cameras found in file')
        if cam_name not in cam_names:
            raise ValueError(f"Camera '{cam_name}' not found. Available: {cam_names}")
        cam_idx = cam_names.index(cam_name)

        n = _num_frames(f, cam_name)
        start = max(0, int(args.start))
        end = n if args.end is None else min(n, int(args.end))
        if start >= end:
            raise ValueError(f'Invalid range start={start} end={end} (n={n})')

        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)

        win = f"{os.path.basename(args.path)} :: {cam_name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        t = start
        dt = 1.0 / max(1e-6, float(args.fps))
        paused = bool(args.step)

        while True:
            img_rgb = _decode_frame(f, cam_name, cam_idx, t)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            overlay = img_bgr.copy()
            cv2.putText(
                overlay,
                f"t={t}/{n-1}  compress={bool(f.attrs.get('compress', False))}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, overlay)

            if args.save_dir is not None:
                out_path = os.path.join(args.save_dir, f"{cam_name}_{t:06d}.png")
                cv2.imwrite(out_path, img_bgr)

            if paused:
                key = cv2.waitKey(0) & 0xFF
            else:
                t0 = time.time()
                key = cv2.waitKey(1) & 0xFF
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused
            if key == ord('a'):
                t = max(start, t - 1)
                paused = True
                continue
            if key == ord('d'):
                t = min(end - 1, t + 1)
                paused = True
                continue

            if not paused:
                t += 1
                if t >= end:
                    break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
