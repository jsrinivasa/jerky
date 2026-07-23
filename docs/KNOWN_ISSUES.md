# Known Issues, Hardware Gotchas, and Lessons From Past Sessions

*Compiled 2026-07-22 from prior debugging sessions on this machine
(2026-07-07 through 2026-07-22). This is a distillation, not a full log --
kept here so nobody has to re-derive any of it from scratch. Dates below
are when each issue was found, not necessarily when it started.*

## This laptop has two distinct hard-crash modes -- both hardware, not nav bugs

Found 2026-07-16, same day, back to back. If the machine hard-freezes
during a nav/mapping session, check `journalctl -b -1 -k` for which
signature it is before assuming it's a repeat of either:

1. **USB4/Thunderbolt instability.** Signature: `usb 4-2.4.2: Non-zero
   status (-71) in video completion handler` / `Failed to query (SET_CUR)
   UVC control` on a RealSense camera, then a USB disconnect/re-enumerate
   storm, then hard freeze. The camera chain runs through 2+ chained USB
   hubs on the USB4/Thunderbolt controller (Bus 04) -- tunneled USB3-over-
   USB4 through chained hubs is less robust for high-bandwidth isochronous
   video than a native controller. A udev autosuspend-disable rule was
   tried and **confirmed NOT sufficient** (errors persisted, just changed
   pattern, device stopped actually disconnecting). This looks
   physical-layer (marginal cable/connector or hub power/signal), not
   software -- next steps if it recurs are physical: reseat the cable, try
   a different USB-C cable, try a native USB3 port instead of tunneled
   USB4, or plug directly into the laptop instead of through the hub
   chain.
2. **Intel i915 GPU GuC/forcewake hang.** Signature: `gt: timed out
   waiting for forcewake ack request`, `GUC: TLB invalidation response
   timed out`, cascading into IOMMU/VT-d timeouts, then `watchdog:
   Watchdog detected hard LOCKUP`. Triggered by a long RTAB-Map ICP
   registration spin (30+ seconds failing to converge) combined with
   RViz's GPU load (live camera texture upload every frame) -- this is a
   known bug class on early Meteor Lake iGPU support. Mitigated (not
   fully fixed) by lowering RViz's `rtabmap_localization.rviz` camera feed
   to Best-Effort/depth-1 QoS and capping its frame rate to 15. The real
   fix (`i915.enable_guc=0` kernel boot param) needs a reboot and was
   deliberately deferred rather than applied casually -- only worth doing
   if this signature recurs.

**General mitigation that applies to both:** avoid long unattended
launch/kill cycling when the system has been running many hours straight
-- several sessions found unrelated subsystems (camera, then separately
the base driver) intermittently failing to come up cleanly after many
hours of continuous bringup/teardown, which reads as hardware/driver state
fatigue rather than a specific bug. A clean reboot before a demo-critical
session is cheap insurance.

## Recurring bug pattern: two publishers fighting for the same cmd_vel topic

This has now happened **twice**, on two different topics, months apart --
worth recognizing as a pattern, not two unrelated incidents, if it
happens a third time:

- **2026-07-16**: `teleop_twist_joy` (`base_joystick_teleop`, direct stick
  → `/mobile_base/cmd_vel`) and `nav_deadman` both gated on the same L2
  button, both able to reach `/mobile_base/cmd_vel`. Robot hit a wall and
  kept moving after L2 release -- not a localization bug, `teleop_twist_joy`
  had zero knowledge of `nav_deadman` and kept independently publishing.
  Fixed via a `use_direct_teleop` launch arg so mapping (`bringup`, direct
  teleop) and nav (`bringup nav`, no direct teleop) are mutually exclusive
  bringup modes.
- **2026-07-22**: `nav_joystick_teleop` (the "safe" replacement -- moved to
  publish `/nav_cmd_vel` instead, so it goes through `nav_deadman`) and
  `simple_nav_planner` both publish to `/nav_cmd_vel` during an
  autonomous drive, same L2 button, no arbitration. Symptom this time was
  "smooth for a second, stutter, smooth again" rather than an outright
  wall hit -- `nav_deadman` was receiving mostly zeros from teleop, with
  the planner's real command winning only occasionally. See
  `docs/MAPFREE_NAV.md`'s "Why manual teleop turns off after anchoring"
  for the full fix.

**Lesson**: any time a joystick-driven teleop node and an autonomous
controller can both reach the same downstream cmd_vel topic while sharing
an enable button, they will fight, even if one of them is "supposed to be
gated." If unexplained jerky/uncontrolled motion shows up again, `ps aux |
grep teleop` before assuming it's a planning or localization bug.

## The saved-map flow has a real, still-not-fully-resolved loop-closure history

This is why `docs/MAPFREE_NAV.md` (fresh SLAM every session, no
loop-closure dependency) was built and is the recommended flow as of
2026-07-22, rather than continuing to chase this.

Across 2026-07-20 and 07-21, extensive live debugging found RTAB-Map
producing **exactly 0 inliers on every single loop-closure attempt**, on
freshly-recorded maps, regardless of viewpoint or how many raw feature
matches were found (16-45, healthy) -- a signature that points at
something structurally broken in geometric verification, not "the office
is visually repetitive" (which would occasionally get lucky). Real,
confirmed, independent bugs were found and fixed along the way (all
already in the current code, not open anymore):

- `depth_module.profile`/`rgb_camera.profile` are **not valid**
  `realsense2_camera` parameter names for the installed version -- the
  real ones are `depth_module.depth_profile`/`rgb_camera.color_profile`.
  Every resolution value ever configured via the wrong names was silently
  ignored (no error) for this whole project's history until found and
  fixed 2026-07-20. If a camera setting doesn't seem to take effect,
  check `ros2 param list /cam_high/camera` for the actual valid names
  before assuming the code is wrong.
- RTAB-Map package version (0.22.1) was older than what had created an
  existing `.db` file (0.23.3) -- the older binary silently got 0 inliers
  reading vocabulary it couldn't parse correctly, while `map->base_link`
  still resolved as a near-identity transform, LOOKING like successful
  localization while being meaningless. Upgraded to 0.23.7. **Lesson: a
  resolving TF is not proof of correct localization** -- cross-check
  against something external (a human confirming actual position) before
  trusting it.
- Both cameras defaulted to identical internal TF frame names
  (`camera_name` defaults to `'camera'` regardless of ROS namespace) --
  the rear camera's data was silently registered through the front
  camera's TF chain. Fixed by explicitly setting `camera_name`.
- EKF was fusing the wrong raw IMU axis for yaw -- the IMU's
  `_optical_frame` convention (Z=forward) doesn't map onto body-frame yaw
  the way a naive array-index assumption expects; confirmed and fixed via
  `robot_localization`'s own `debug:true` dump, not guessed.

**What's still open**: even after all of the above, a freshly-recorded map
still produced 0 loop closures, while an older map (`building16_imu_v2.db`,
predates several of these fixes) is confirmed via direct SQLite query to
have 77 real closure links -- so the hardware/environment is not
fundamentally incapable, something changed between when that map was made
and later sessions. The comparison retest that would isolate "config
change" vs. "current hardware/calibration state" was never completed
(blocked by hardware flakiness late in the 07-20 session). If picking the
saved-map flow back up, that comparison test is the right first move, not
more remapping attempts -- confirmed more driving alone doesn't fix a
registration-level problem.

## Smaller gotchas worth knowing

- **`--symlink-install` matters.** If a source edit doesn't seem to take
  effect after a relaunch, check whether `install/aloha/...` is actually
  symlinked back to `src/aloha/...` (`stat -c '%i' src/... install-or-build/...`
  -- matching inode numbers = symlinked, fixed automatically; different =
  a stale copy, needs `colcon build --packages-select aloha`). This has
  been lost and rediscovered more than once across sessions. Launch-file
  changes specifically have needed a rebuild even when this workspace was
  otherwise symlink-installed correctly -- Python node (`.py`) edits take
  effect immediately, launch-file (`.launch.py`) edits have not always.
- **`delete_db_on_start` will silently wipe a map.** Re-launching mapping
  against the SAME map name to inspect/republish it wipes the existing
  recording unless you explicitly pass `delete_db_on_start:=false` or
  `localization:=true`. Take a defensive `cp` backup of a `.db` file
  before re-touching it if it represents a physical drive that can't be
  trivially redone.
- **`save` must run before `stop`, not after.** `map_saver_cli` reads the
  live `/rtabmap/map` topic, not a persisted file -- once `stop` kills
  rtabmap there's no publisher left to read from. (`demo_ops.sh`'s help
  text and `NAV_RUNBOOK.md` already reflect the correct order.)
- **Never use a `__`-prefixed name for a diagnostic marker in a launch
  file.** ROS 2 reserves double-underscore-prefixed names (`__node:=`,
  `__ns:=`, etc.) for CLI directives -- a stray `__test_marker__` remap
  left in a shared launch file silently corrupted argument parsing for
  every subsequent launch of that node until removed. Remove one-off
  diagnostic edits to a shared launch file the moment the test is done.
- **DDS shared-memory exhaustion** (`RTPS_TRANSPORT_SHM Error: Failed
  init_port ... open_and_lock_file failed`) can recur after enough rapid
  restart cycles in one session -- stale `/dev/shm/fastrtps_*` segments
  pile up and cause bizarre-looking node/topic failures easy to
  misdiagnose as something else. `demo_ops.sh stop` already clears this
  automatically every time (`find /dev/shm -maxdepth 1 -name
  "fastrtps_*" -delete`) -- if diagnosing on a machine/script that
  doesn't do this, check for it directly.
- **Physical measurement corrections already baked into the current
  code** (don't re-measure/re-guess these): `base_link` pivot sits ~4.7in
  (0.119m) toward the back of the 28in x 22in footprint, not centered
  (see `docs/MAPFREE_NAV.md`'s anchor-point section); `cam_high` height is
  1.143m (45in); RPLIDAR S2 x-offset is 0.1397m (5.5in, a measured-range
  midpoint estimate, not exact); the RPLIDAR is mounted physically
  rotated 90°, so its TF yaw is `+π/2`, not 0.
- **`~/maps/` accumulates old iterations across sessions with nothing
  cleaning it up** -- as of 2026-07-22 it has 20+ `session*.db` files from
  map-free testing plus many older `building16*`/`floorplan*` variants
  from tuning `svg_to_map.py`. Harmless to leave, but don't assume
  everything in there is current/meaningful.
