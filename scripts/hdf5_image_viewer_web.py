#!/usr/bin/env python3

import argparse
import io
import os
from urllib.parse import unquote

import cv2
import h5py
import numpy as np
from flask import Flask, Response, jsonify, request


def _safe_path(path: str, directory: str) -> str:
    if not path:
        raise ValueError('Missing path')
    path = unquote(path)
    path = os.path.abspath(os.path.expanduser(path))
    if not path.endswith('.hdf5'):
        raise ValueError('Path must end with .hdf5')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if directory:
        if os.path.commonpath([path, directory]) != directory:
            raise ValueError(f'Path must be under: {directory}')
    return path


def _get_cam_names(f: h5py.File) -> list[str]:
    if 'observations' not in f or 'images' not in f['observations']:
        raise KeyError("Missing group 'observations/images'")
    return list(f['observations/images'].keys())


def _decode_frame_rgb(f: h5py.File, cam_names: list[str], cam_name: str, t: int) -> np.ndarray:
    compress = bool(f.attrs.get('compress', False))

    if not compress:
        img = f['observations/images'][cam_name][t]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Unexpected image shape for {cam_name}: {img.shape}")
        return img

    if 'compress_len' not in f:
        raise KeyError("File is marked compress=True but missing dataset 'compress_len'")

    cam_idx = cam_names.index(cam_name)
    padded = f['observations/images'][cam_name][t]
    true_len = int(f['compress_len'][cam_idx, t])
    jpg_bytes = padded[:true_len].tobytes()
    img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to decode JPEG for cam={cam_name} frame={t} (len={true_len})")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _safe_browse_path(path: str, directory: str) -> str:
    if not path:
        return directory
    path = unquote(path)
    path = os.path.abspath(os.path.expanduser(path))
    if directory:
        if os.path.commonpath([path, directory]) != directory:
            raise ValueError(f'Path must be under: {directory}')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def create_app(directory: str):
    app = Flask(__name__)

    @app.get('/')
    def index():
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HDF5 Episode Viewer</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; }}
    .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
    select, button, input[type=number] {{ padding: 6px 10px; }}
    #img {{ max-width: 95vw; max-height: 60vh; border: 1px solid #ccc; background: #111; display: block; }}
    .muted {{ color: #666; font-size: 13px; }}
    #browser {{ border: 1px solid #ccc; padding: 10px; margin-bottom: 14px; max-width: 550px; background: #f9f9f9; }}
    #browser select {{ width: 100%; padding: 6px; margin: 4px 0; }}
    #pathDisplay {{ font-family: monospace; font-size: 12px; color: #333; margin-bottom: 6px; word-break: break-all; }}
  </style>
</head>
<body>
  <h2>HDF5 Episode Viewer</h2>

  <div id="browser">
    <div class="muted">Select folder then file:</div>
    <div id="pathDisplay">{directory}</div>
    <select id="folderSelect" size="4"></select>
    <select id="fileSelect" size="4"></select>
    <button id="loadBtn" style="margin-top:6px;">Load Selected File</button>
    <button id="downloadBtn" style="margin-top:6px; margin-left:6px;">Download File</button>
  </div>

  <div class="row">
    <label>Camera:</label>
    <select id="camera"></select>
    <span id="info" class="muted"></span>
  </div>

  <div class="row">
    <button id="prev">&lt; Prev</button>
    <input id="frame" type="range" min="0" max="0" value="0" style="width:400px;" />
    <button id="next">Next &gt;</button>
    <span id="frameLabel" style="min-width:80px;">t=0</span>
    <input id="fps" type="number" min="1" max="60" value="20" style="width:60px;" />
    <span class="muted">FPS</span>
    <button id="play">Play</button>
  </div>

  <img id="img" alt="frame" />

<script>
const state = {{ currentDir: '{directory}', loadedPath: '', n: 0, playing: false, tid: null }};

const $ = id => document.getElementById(id);

async function loadDir(dir) {{
  $('folderSelect').innerHTML = '<option disabled>Loading...</option>';
  $('fileSelect').innerHTML = '';
  try {{
    const r = await fetch('/api/browse?path=' + encodeURIComponent(dir));
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    state.currentDir = d.path;
    $('pathDisplay').textContent = d.path;

    let folderHtml = '';
    if (d.parent) folderHtml += '<option value="' + d.parent + '">..</option>';
    for (const f of d.folders) folderHtml += '<option value="' + d.path + '/' + f + '">' + f + '/</option>';
    $('folderSelect').innerHTML = folderHtml || '<option disabled>(no subfolders)</option>';

    let fileHtml = '';
    for (const f of d.files) fileHtml += '<option value="' + d.path + '/' + f + '">' + f + '</option>';
    $('fileSelect').innerHTML = fileHtml || '<option disabled>(no .hdf5 files)</option>';
  }} catch (e) {{
    alert(e);
  }}
}}

$('folderSelect').ondblclick = () => {{
  const v = $('folderSelect').value;
  if (v) loadDir(v);
}};

$('loadBtn').onclick = async () => {{
  const fp = $('fileSelect').value;
  if (!fp || fp.startsWith('(')) {{ alert('Select an .hdf5 file first'); return; }}
  try {{
    const r = await fetch('/api/info?path=' + encodeURIComponent(fp));
    if (!r.ok) throw new Error(await r.text());
    const info = await r.json();
    state.loadedPath = fp;
    state.n = info.num_frames;

    $('camera').innerHTML = info.cameras.map(c => '<option>' + c + '</option>').join('');
    $('frame').max = Math.max(0, state.n - 1);
    $('frame').value = 0;
    $('info').textContent = fp.split('/').pop() + ' | ' + state.n + ' frames | compress=' + info.compress;
    showFrame(0);
  }} catch (e) {{
    alert(e);
  }}
}};

$('downloadBtn').onclick = () => {{
  const fp = $('fileSelect').value;
  if (!fp || fp.startsWith('(')) {{ alert('Select an .hdf5 file first'); return; }}
  window.location.href = '/api/download?path=' + encodeURIComponent(fp);
}};

function showFrame(t, onDone) {{
  if (!state.loadedPath) return;
  $('frameLabel').textContent = 't=' + t + '/' + (state.n - 1);
  const img = $('img');
  const url = '/api/frame.png?path=' + encodeURIComponent(state.loadedPath) +
              '&cam=' + encodeURIComponent($('camera').value) +
              '&t=' + t + '&_=' + Date.now();
  if (onDone) {{
    img.onload = onDone;
    img.onerror = onDone;
  }}
  img.src = url;
}}

$('frame').oninput = () => showFrame(parseInt($('frame').value));
$('camera').onchange = () => showFrame(parseInt($('frame').value));
$('prev').onclick = () => {{ const v = Math.max(0, parseInt($('frame').value) - 1); $('frame').value = v; showFrame(v); }};
$('next').onclick = () => {{ const v = Math.min(state.n - 1, parseInt($('frame').value) + 1); $('frame').value = v; showFrame(v); }};

function playNext() {{
  if (!state.playing) return;
  let t = parseInt($('frame').value) + 1;
  if (t >= state.n) {{
    state.playing = false;
    $('play').textContent = 'Play';
    return;
  }}
  $('frame').value = t;
  const fps = Math.max(1, parseInt($('fps').value) || 20);
  const delay = 1000 / fps;
  const start = Date.now();
  showFrame(t, () => {{
    const elapsed = Date.now() - start;
    const wait = Math.max(0, delay - elapsed);
    setTimeout(playNext, wait);
  }});
}}

$('play').onclick = () => {{
  if (state.playing) {{
    state.playing = false;
    $('play').textContent = 'Play';
  }} else {{
    state.playing = true;
    $('play').textContent = 'Pause';
    playNext();
  }}
}};

loadDir('{directory}');
</script>
</body>
</html>"""

    @app.get('/api/browse')
    def api_browse():
        try:
            path = _safe_browse_path(request.args.get('path', ''), directory)
            if not os.path.isdir(path):
                raise ValueError(f'Not a directory: {path}')

            folders = []
            files = []
            for entry in sorted(os.listdir(path)):
                full = os.path.join(path, entry)
                if os.path.isdir(full):
                    folders.append(entry)
                elif entry.endswith('.hdf5'):
                    files.append(entry)

            parent = None
            if directory and path != directory:
                parent_path = os.path.dirname(path)
                if os.path.commonpath([parent_path, directory]) == directory:
                    parent = parent_path

            return jsonify({
                'path': path,
                'parent': parent,
                'folders': folders,
                'files': files,
            })
        except Exception as e:
            return Response(str(e), status=400)

    @app.get('/api/info')
    def api_info():
        try:
            path = _safe_path(request.args.get('path', ''), directory)
            with h5py.File(path, 'r') as f:
                cams = _get_cam_names(f)
                cam0 = cams[0]
                n = int(f['observations/images'][cam0].shape[0])
                return jsonify({
                    'path': path,
                    'compress': bool(f.attrs.get('compress', False)),
                    'cameras': cams,
                    'num_frames': n,
                })
        except Exception as e:
            return Response(str(e), status=400)

    @app.get('/api/frame.png')
    def api_frame_png():
        try:
            path = _safe_path(request.args.get('path', ''), directory)
            cam = request.args.get('cam', '')
            t = int(request.args.get('t', '0'))

            with h5py.File(path, 'r') as f:
                cams = _get_cam_names(f)
                if cam not in cams:
                    raise ValueError(f"Camera '{cam}' not found. Available: {cams}")
                n = int(f['observations/images'][cam].shape[0])
                if t < 0 or t >= n:
                    raise ValueError(f'Frame index out of range t={t}, n={n}')

                rgb = _decode_frame_rgb(f, cams, cam, t)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, png = cv2.imencode('.png', bgr)
                if not ok:
                    raise ValueError('Failed to encode PNG')

            return Response(png.tobytes(), mimetype='image/png')
        except Exception as e:
            return Response(str(e), status=400)

    @app.get('/api/download')
    def api_download():
        try:
            path = _safe_path(request.args.get('path', ''), directory)
            filename = os.path.basename(path)
            return Response(
                open(path, 'rb').read(),
                mimetype='application/x-hdf5',
                headers={'Content-Disposition': f'attachment; filename="{filename}"'}
            )
        except Exception as e:
            return Response(str(e), status=400)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='/home/aloha/aloha_data',
        help='Directory containing .hdf5 episode files. Only files under this directory can be viewed.',
    )
    args = parser.parse_args()

    directory = os.path.abspath(os.path.expanduser(args.directory)) if args.directory else ''
    app = create_app(directory)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
