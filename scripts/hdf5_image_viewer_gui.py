#!/usr/bin/env python3

import argparse
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import h5py
import numpy as np
from PIL import Image, ImageTk


class Hdf5EpisodeViewer(tk.Tk):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.title('HDF5 Episode Image Viewer')
        self.geometry('1100x800')

        self.file_path: str | None = None
        self.h5: h5py.File | None = None
        self.cam_names: list[str] = []
        self.cam_idx: int = 0
        self.num_frames: int = 0
        self.compress: bool = False

        self._photo = None

        self._build_ui()

        if initial_path:
            self.load_file(initial_path)

        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.open_btn = ttk.Button(top, text='Open HDF5...', command=self.open_dialog)
        self.open_btn.pack(side=tk.LEFT)

        self.path_var = tk.StringVar(value='')
        self.path_entry = ttk.Entry(top, textvariable=self.path_var, width=80)
        self.path_entry.pack(side=tk.LEFT, padx=8)

        self.reload_btn = ttk.Button(top, text='Reload', command=self.reload)
        self.reload_btn.pack(side=tk.LEFT)

        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        ttk.Label(controls, text='Camera:').pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value='')
        self.camera_combo = ttk.Combobox(controls, textvariable=self.camera_var, state='readonly', width=30)
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_changed)
        self.camera_combo.pack(side=tk.LEFT, padx=6)

        self.info_var = tk.StringVar(value='')
        self.info_label = ttk.Label(controls, textvariable=self.info_var)
        self.info_label.pack(side=tk.LEFT, padx=12)

        slider_frame = ttk.Frame(self)
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.frame_label_var = tk.StringVar(value='t=0')
        self.frame_label = ttk.Label(slider_frame, textvariable=self.frame_label_var, width=12)
        self.frame_label.pack(side=tk.LEFT, padx=8)

        btns = ttk.Frame(self)
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.prev_btn = ttk.Button(btns, text='Prev', command=self.prev_frame)
        self.prev_btn.pack(side=tk.LEFT)

        self.next_btn = ttk.Button(btns, text='Next', command=self.next_frame)
        self.next_btn.pack(side=tk.LEFT, padx=6)

        self.playing = False
        self.play_btn = ttk.Button(btns, text='Play', command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=6)

        ttk.Label(btns, text='FPS:').pack(side=tk.LEFT, padx=(16, 4))
        self.fps_var = tk.DoubleVar(value=30.0)
        self.fps_entry = ttk.Entry(btns, textvariable=self.fps_var, width=6)
        self.fps_entry.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, bg='black')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

    def open_dialog(self):
        path = filedialog.askopenfilename(
            title='Select HDF5 episode file',
            filetypes=[('HDF5 files', '*.hdf5'), ('All files', '*.*')],
        )
        if path:
            self.load_file(path)

    def reload(self):
        if self.file_path:
            self.load_file(self.file_path)

    def load_file(self, path: str):
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

            self._close_h5()
            self.h5 = h5py.File(path, 'r')
            self.file_path = path
            self.path_var.set(path)

            if 'observations' not in self.h5 or 'images' not in self.h5['observations']:
                raise KeyError("Missing group 'observations/images'")

            self.cam_names = list(self.h5['observations/images'].keys())
            if not self.cam_names:
                raise ValueError('No cameras found in observations/images')

            self.compress = bool(self.h5.attrs.get('compress', False))
            self.camera_combo['values'] = self.cam_names
            self.camera_combo.current(0)
            self.camera_var.set(self.cam_names[0])
            self.cam_idx = 0

            self.num_frames = int(self.h5['observations/images'][self.cam_names[0]].shape[0])
            self.slider.configure(from_=0, to=max(0, self.num_frames - 1))
            self.slider.set(0)
            self.frame_var.set(0)

            self.info_var.set(f"frames={self.num_frames}  compress={self.compress}")
            self.show_frame(0)

        except Exception as e:
            messagebox.showerror('Failed to load HDF5', str(e))
            self._close_h5()

    def on_camera_changed(self, _evt=None):
        if not self.h5:
            return
        cam = self.camera_var.get()
        if cam not in self.cam_names:
            return
        self.cam_idx = self.cam_names.index(cam)
        self.num_frames = int(self.h5['observations/images'][cam].shape[0])
        self.slider.configure(from_=0, to=max(0, self.num_frames - 1))
        t = min(int(self.frame_var.get()), max(0, self.num_frames - 1))
        self.slider.set(t)
        self.show_frame(t)

    def on_slider(self, value):
        t = int(float(value))
        if t != int(self.frame_var.get()):
            self.frame_var.set(t)
            self.show_frame(t)

    def prev_frame(self):
        if not self.h5:
            return
        t = max(0, int(self.frame_var.get()) - 1)
        self.slider.set(t)
        self.frame_var.set(t)
        self.show_frame(t)

    def next_frame(self):
        if not self.h5:
            return
        t = min(max(0, self.num_frames - 1), int(self.frame_var.get()) + 1)
        self.slider.set(t)
        self.frame_var.set(t)
        self.show_frame(t)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text='Pause' if self.playing else 'Play')
        if self.playing:
            self.after(0, self._play_loop)

    def _play_loop(self):
        if not self.playing or not self.h5:
            return
        try:
            fps = float(self.fps_var.get())
        except Exception:
            fps = 30.0
        fps = max(1e-3, fps)
        delay_ms = int(1000.0 / fps)

        t = int(self.frame_var.get()) + 1
        if t >= self.num_frames:
            self.playing = False
            self.play_btn.configure(text='Play')
            return

        self.slider.set(t)
        self.frame_var.set(t)
        self.show_frame(t)
        self.after(delay_ms, self._play_loop)

    def _decode_frame_rgb(self, cam_name: str, cam_idx: int, t: int) -> np.ndarray:
        if not self.h5:
            raise RuntimeError('No file loaded')

        if not self.compress:
            img = self.h5['observations/images'][cam_name][t]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Unexpected image shape for {cam_name}: {img.shape}")
            return img

        if 'compress_len' not in self.h5:
            raise KeyError("File is marked compress=True but missing dataset 'compress_len'")

        padded = self.h5['observations/images'][cam_name][t]
        true_len = int(self.h5['compress_len'][cam_idx, t])
        jpg_bytes = padded[:true_len].tobytes()
        img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to decode JPEG for cam={cam_name} frame={t} (len={true_len})")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def show_frame(self, t: int):
        if not self.h5:
            return
        cam = self.camera_var.get()
        if not cam:
            return
        t = int(t)
        t = max(0, min(t, self.num_frames - 1))

        try:
            img_rgb = self._decode_frame_rgb(cam, self.cam_idx, t)
        except Exception as e:
            self.info_var.set(f"decode error: {e}")
            return

        pil = Image.fromarray(img_rgb)

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        pil.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)

        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.delete('all')
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._photo, anchor=tk.CENTER)
        self.frame_label_var.set(f"t={t}")

    def _close_h5(self):
        if self.h5 is not None:
            try:
                self.h5.close()
            except Exception:
                pass
        self.h5 = None

    def on_close(self):
        self._close_h5()
        self.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='Optional path to episode_*.hdf5')
    args = parser.parse_args()

    app = Hdf5EpisodeViewer(initial_path=args.path)
    app.mainloop()


if __name__ == '__main__':
    main()
