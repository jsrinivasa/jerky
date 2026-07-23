# Mobile ALOHA Navigation Runbook

*Dependency status and links last verified: 2026-07-22.*

Step-by-step instructions to go from an empty map to the robot driving itself
from point A to point B.

**No prebuilt map? Start with [`docs/MAPFREE_NAV.md`](docs/MAPFREE_NAV.md)
instead of this file.** As of 2026-07-22 that's the most-tested, working
click-to-go flow -- floorplan-anchored, no saved RTAB-Map database required,
fresh SLAM every session. This runbook's Phases 2-3 below are for the OLDER
flow: build and save a real map once, then localize against it on future
runs. Both are real, working, currently-maintained flows -- pick map-free for
a quick one-off session in a space you don't need a persistent map of, pick
this runbook's saved-map flow if you want the map to persist and be reused
run over run.

**Current approach (as of 2026-07-20): RTAB-Map visual SLAM + a custom
`simple_nav_planner` (A* + Pure Pursuit), driven end-to-end by
`scripts/demo_ops.sh`.** This replaces an earlier PDF-floorplan + AMCL + Nav2
approach that this runbook used to describe (Phases 2/3 below, pre-2026-07-20)
-- that code (`pdf_to_map.py`, `simple_navigation.launch.py`, AMCL) still
exists in the repo but isn't part of the current recommended flow. If you're
picking this up cold, start here, not with old commit history.

---

## Phase 1: Learn the Planner (Simulation, No Hardware)

Exercises the real A*/Pure-Pursuit planner with fake odometry and no robot --
best first step for understanding the code before touching hardware.

```bash
cd /home/aloha/interbotix_ws && source install/setup.bash
ros2 launch aloha sim_navigation.launch.py     # RViz opens
```

In RViz: click **"2D Goal Pose"**, click-and-drag on the map to set a
destination + heading, release. You should see a planned path and the robot
marker driving along it.

| Problem | Fix |
|---------|-----|
| RViz opens but no map visible | Add `/map` topic manually (Add -> By topic -> /map -> Map) |
| Goal pose does nothing | Check the goal is on free (white) space, not a wall |
| "Could not find path" | Goal may be unreachable or too close to an obstacle |

---

## Phase 2: Build a Map (RTAB-Map SLAM, real hardware)

All of this is wrapped by `scripts/demo_ops.sh` -- run it with no arguments
for the full command list. Every command below assumes the robot is docked
to this laptop and powered on.

```bash
# Terminal 1 (or just use demo_ops.sh, which backgrounds everything):
./scripts/demo_ops.sh bringup              # base + joystick teleop, no cameras
# Drive the robot with the controller to wherever you want mapping to start.

./scripts/demo_ops.sh map building16_v3    # starts RTAB-Map mapping + IMU fusion
./scripts/demo_ops.sh preflight            # sanity check: IMU, TFs, odom, /scan alive
# Now drive around with the controller to build the map. Watch rtabmap_viz
# (opened automatically) for loop closures / drift.

./scripts/demo_ops.sh save building16_v3   # writes ~/maps/building16_v3.{yaml,pgm} --
                                            # MUST run before 'stop': it reads the
                                            # live /rtabmap/map topic, not a file.
./scripts/demo_ops.sh stop                 # tear down once saved
```

`demo_ops.sh maps` lists everything already saved in `~/maps` if you forget
what you named a session.

**If mapping looks unhealthy** (robot doesn't move, map doesn't grow, IMU
warnings in the rtabmap terminal): re-run `./scripts/demo_ops.sh preflight` --
it checks the specific things that have broken before (IMU orientation not
initializing, `/mobile_base/odom` dying when the camera comes up, missing
TFs) and tells you exactly which one, PASS/FAIL/WARN per check.

---

## Phase 3: Navigate on a Saved Map

```bash
./scripts/demo_ops.sh bringup nav          # base + joy_node ONLY (no direct teleop --
                                            # teleop_twist_joy fights nav_deadman for L2
                                            # if both run, see nav_deadman.py)
./scripts/demo_ops.sh maps                 # pick a map name
./scripts/demo_ops.sh nav building16_v3    # starts localization + planner + L2 deadman gate
./scripts/demo_ops.sh localize             # hold L2; rotates until RTAB-Map confirms
                                            # 2 visual matches, then reports LOCALIZED
./scripts/demo_ops.sh viewer               # open the printed URL
```

In the viewer: click a point on the map, **Confirm & Go**, then **hold L2**
to let the robot drive there. **STOP** cancels immediately. Click again to
send it somewhere else -- repeat as many times as you want, no restart needed.
The robot moves *only* while L2 is held, full stop the instant you release it
or hit STOP.

`nav` accepts two flags for testing (2026-07-20 additions, see
`nav-drift-fixes-2026-07-20` notes for why):
- `--no-ekf` -- skip the wheel+IMU sensor fusion (`config/ekf.yaml`) and use
  raw wheel odometry instead. Use this if `robot_localization` isn't
  installed yet, or if the fused estimate looks worse than raw odom.
- `--continuous-mapping` -- keep extending/refining the saved map live while
  navigating, instead of treating it as a frozen reference. Trade-off: a bad
  loop closure can then permanently corrupt the map, which locked mode can't
  do -- only turn this on once you trust localization.

When you're done: `./scripts/demo_ops.sh stop` tears everything down cleanly
(kills orphaned launch children by name, resets the ros2 daemon).

---

## Dependency status (checked 2026-07-22)

`ros-humble-robot-localization` (wheel+IMU EKF fusion, `nav`/`mapfree`'s
default) and `ros-humble-apriltag-ros` are both **installed** on this
machine now -- the `--no-ekf` flag / "install this first" instructions
below are no longer needed day-to-day, kept only as a rollback if EKF fusion
ever misbehaves.

A disabled-by-default AprilTag-landmark feature also exists
(`use_apriltag_landmarks` in `rtabmap_mapping.launch.py`, not yet wired
through `demo_ops.sh`/`navigate_mission.launch.py`) -- the package is
installed, but this still needs physical tags printed, measured, and placed,
and the actual wiring done, before it does anything. Not part of the normal
flow yet.

---

## Quick Reference: simple_nav_planner parameters

These apply whether you're in `sim_navigation.launch.py` or
`navigate_mission.launch.py` (via `demo_ops.sh nav`):

```bash
robot_radius:=0.58                   # safety clearance (worst-case turning footprint)
inflation_radius:=0.40               # A* path-planning inflation (straight-line half-width)
max_linear_velocity:=0.3             # max forward speed (m/s)
max_angular_velocity:=1.0            # max turning speed (rad/s)
goal_tolerance:=0.2                  # how close counts as "arrived" (meters)
lookahead_distance:=0.5              # pure-pursuit lookahead (meters)
enable_collision_avoidance:=true     # live /scan-based obstacle reactions
safety_distance:=0.6096              # start slowing within this clearance (2ft)
emergency_stop_distance:=0.4572      # full stop within this clearance (1.5ft)
```
