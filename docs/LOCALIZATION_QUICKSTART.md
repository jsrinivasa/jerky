# Localization + 2D Nav Quickstart

Commands to localize on a previously-built RTAB-Map and drive with 2D goals.
See `NAV_RUNBOOK.md` Phase 3 for the canonical version of this flow --
this file adds the gotchas found on 2026-07-20 that aren't there yet.

Every terminal: `cd ~/interbotix_ws && source install/setup.bash` first.

## The commands

```bash
./src/aloha/scripts/demo_ops.sh stop        # always start clean -- see gotcha below
./src/aloha/scripts/demo_ops.sh bringup nav # base + joy_node ONLY, no direct teleop
./src/aloha/scripts/demo_ops.sh maps        # list saved maps, pick one
./src/aloha/scripts/demo_ops.sh nav <map_name>   # e.g. building16_imu_v2
./src/aloha/scripts/demo_ops.sh localize    # hold L2; rotates until RTAB-Map
                                             # confirms 2 visual matches -> LOCALIZED
./src/aloha/scripts/demo_ops.sh viewer      # open the printed http://<ip>:8080/ URL
```

In the viewer: click a point, **Confirm & Go**, then **hold L2** to drive
there. **STOP** cancels immediately. Robot moves only while L2 is held.

When done: `./src/aloha/scripts/demo_ops.sh stop`.

`nav` flags: `--no-ekf` (raw wheel odom instead of EKF fusion), `--continuous-mapping`
(keep extending the map live instead of treating it as frozen/read-only).

## Checking it's actually working (don't just trust the launch logs)

Startup logs looking clean does NOT mean data is flowing. After `nav`,
give the camera ~15-20s to warm up, then check:

```bash
grep -iE "MIPI|Hardware Error|Did not receive data|RealSense Node Is Up" \
    ~/.aloha_demo/logs/nav.log
```

- `RealSense Node Is Up!` with no MIPI/Hardware Error after it, and the
  "Did not receive data" warnings stop appearing after a few seconds =
  camera is actually streaming.
- Warnings that keep repeating past ~15s = rtabmap isn't getting real
  camera frames, even though everything "started". Don't try to localize/
  drive in this state -- see Known Issues below.

## Known issues (as of 2026-07-20, still open)

**Camera (D435I "cam_high") intermittently fails to stream color/depth.**
Symptom: `realsense2_camera_node` logs "RealSense Node Is Up!" cleanly, but
`rgbd_sync`/`rtabmap` never receive a single frame, and the kernel log shows:
```
uvcvideo <bus>:1.1: Non-zero status (-71) in video completion handler.
uvcvideo <bus>:1.4: Non-zero status (-71) in video completion handler.
```
(check with `journalctl -k --since "2 minutes ago" | grep -i uvcvideo`)

Ruled out this session, all reproduced the identical failure: 2 different
physical D435I units, 3 different USB cables, 3 different USB ports across
2 different host controllers, lower resolution (424x240 vs 640x480), and
CPU/system load (see leak fix below -- load dropped from 5.2 to 3.0 and the
error still happened identically). Leading remaining theory: outdated
camera firmware (was `5.12.6`, Intel recommends `5.17.0.9` --
check with `rs-enumerate-devices --compact`) vs the installed
librealsense/uvcvideo driver. Not yet attempted: `rs-fw-update` firmware
update.

**If the camera isn't streaming, don't try to navigate.** Without real
visual loop closures, RTAB-Map's map->odom TF is just its unconfirmed
starting guess. `simple_nav_planner` will compute paths against that wrong
belief while `check_collision_ahead()` sees real (mismatched) obstacles via
the live `/scan`, so every replan fails and the robot cycles
wait -> rotate -> replan (see `simple_nav_planner.py:1094-1168`) --
this is what "it just spins in a circle when I hold L2" was.

**Two process leaks in `demo_ops.sh stop`, fixed 2026-07-20 (already
patched in this repo, just documenting so it's understood):**
`interbotix_gravity_compensation` (leader arms) and `robot_pose_marker`
were both missing from the `stop` kill patterns, so every `bringup`/`nav`
left a pair/single process running forever -- 16+ `robot_pose_marker`
copies had accumulated over one evening, each at 4-9% CPU. If `stop`
followed by `pgrep -af "gravity_compensation|robot_pose_marker"` ever shows
matches again, something regressed -- patterns live in
`scripts/demo_ops.sh` `cmd_stop()`.

**Arm errors during `bringup`/`bringup nav` are expected and harmless for
nav-only sessions** (arms aren't needed to drive the base):
```
Failed to open port at '/dev/ttyDXL_leader_left'
Failed to open port at '/dev/ttyDXL_leader_right'
Failed to find all motors  (follower_left)
```
These mean the arms are powered off / not connected -- fine if you only
care about navigation. Ignore them.

**CAM_HIGH_SERIAL in `rtabmap_mapping.launch.py`** is hardcoded to the
physical camera's serial number (line ~81). If you ever swap the camera
unit, update it (confirm the new serial with
`rs-enumerate-devices --compact`) and `colcon build --packages-select aloha`
before relaunching, or the camera node will fail to find the configured
serial.

**Known-good map:** `building16_imu_v2` (saved 2026-07-16) is confirmed to
have real loop closures from a prior session. Other maps recorded more
recently this same night (`building16_corrected`) were found to have zero
real loop-closure inliers -- prefer `building16_imu_v2` until that's
re-investigated.
