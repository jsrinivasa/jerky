# RPLIDAR S2 — remaining on-robot steps

All the software side is done (in this dev copy: `nav_dev_ws/src/aloha`, not
the original workspace) — driver node, TF, and sensor fusion into `/scan` are
wired into `launch/navigate_mission.launch.py`, gated behind `use_rplidar`
(default `true`) so nothing changes if you launch without the lidar attached.
What's left needs the physical hardware:

## 1. Install the driver (one-time, needs sudo)
```bash
sudo apt-get install -y ros-humble-rplidar-ros
```
Confirms it supports S2: `ros-humble-rplidar-ros` covers A1/A2/A3/S1/S2/S3/T1.

## 2. Mount + plug in
Front, near the bottom, as planned. Two things to keep in mind:
- **Self-occlusion**: mounted at the front (not centered), the chassis will
  block part of the S2's own 360° sweep (roughly the rear wedge, behind the
  mount). This is expected and fine for driving-direction obstacle avoidance
  (`simple_nav_planner` only checks a forward cone) — it just means the
  merged `/scan`'s rear coverage relies on nothing being back there, or on
  future rear sensing. Don't expect true 360° coverage from this mount
  position.
- Keep the sweep plane clear of the arms' resting pose and any mast/cable
  clutter directly at lidar height.

## 3. Stable device path (recommended, one-time)
Without this, the port is whatever `/dev/ttyUSBx` the kernel happens to
assign, which can shift if other USB-serial devices are plugged in a
different order.
```bash
lsusb | grep -i "10c4:ea60"      # confirm exactly one match
sudo cp config/99-rplidar.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -la /dev/rplidar               # should appear
sudo usermod -aG dialout $USER    # if not already in this group; re-login after
```
If you skip this, pass the real port instead: `rplidar_port:=/dev/ttyUSB0`
(check with `ls /dev/ttyUSB*` after plugging in).

## 4. Measure the mount offset (replaces placeholders in the launch file)
Same tape-measure approach already used for `cam_high` in
`rtabmap_mapping.launch.py`. From `base_link` origin to the lidar's rotation
center:
- `rplidar_x` — forward offset (m), placeholder `0.25`
- `rplidar_y` — left/right offset (m), placeholder `0.0`
- `rplidar_z` — height off the ground (m), placeholder `0.05`
- `rplidar_yaw` — rotation (rad) of the lidar's zero-angle mark vs.
  `base_link`'s forward (+x) axis, placeholder `0.0` — check the housing for
  its printed zero-mark, don't assume it's forward-facing.

Pass overrides on launch, e.g.:
```bash
ros2 launch aloha navigate_mission.launch.py map_name:=<map> \
    rplidar_x:=0.30 rplidar_y:=0.0 rplidar_z:=0.08 rplidar_yaw:=0.0
```
Once confirmed correct, edit the defaults directly in
`launch/navigate_mission.launch.py` (the `rplidar_x_arg` etc. declarations)
so you don't have to pass them every time.

## 5. Test
```bash
ros2 launch aloha navigate_mission.launch.py map_name:=<existing map>
ros2 topic hz /scan_rplidar     # confirm the driver is publishing
ros2 topic hz /scan             # confirm the merged topic is publishing
```
In RViz, add `/scan` (or `/scan_rplidar` alone first, to sanity check the
raw driver output before trusting the fused topic) and confirm the points
line up with real walls/objects, not offset or rotated — a wrong
`rplidar_yaw` shows up immediately as a scan that's rotated relative to the
map.

## 6. Tune self-occlusion (only if needed)
If a sector of `/scan` shows a constant, unmoving close-range obstacle that
isn't real, it's almost certainly the robot's own chassis. Find its angular
range in `base_link` (0 = forward, +/- pi = behind) and set it in
`launch/navigate_mission.launch.py`'s `laser_scan_merger_node` parameters,
e.g.:
```python
'self_occlusion_ranges': [2.6, 3.14, -3.14, -2.6],  # rear wedge example
```
