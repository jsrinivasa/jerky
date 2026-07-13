# Isaac Sim + Mobile ALOHA Nav Stack — Install & Integration Plan

Status: PLAN (nothing installed yet). Target machine = this laptop (the robot's compute).

## 0. Hardware verdict (measured on this machine, 2026-07-02)

| Resource | This machine | Isaac Sim min / rec | Verdict |
|---|---|---|---|
| GPU | **RTX 2000 Ada Laptop, 8 GB VRAM** | RTX min 8 GB / rec 16 GB+ | ✅ runs, ⚠️ below recommended |
| Driver | 570.211.01 (CUDA 12.8) | ≥ 535 | ✅ modern, fine |
| RAM | 30 GB | 32 min / 64 rec | ⚠️ borderline |
| Disk | 733 GB free | ~30 GB needed | ✅ |
| OS / Python | Ubuntu 22.04, Py 3.10 | 22.04 + Py 3.10 | ✅ ideal for pip install |
| ROS 2 | Humble | Humble supported by bridge | ✅ |
| Vulkan | nvidia_icd.json present; `vulkan-tools` NOT installed | Vulkan required | ⚠️ install vulkan-tools to verify |
| Display | current session is **TTY, DISPLAY empty**; PRIME = on-demand | GUI needs a display | ⚠️ decision needed (see §2) |

**Bottom line:** viable for nav testing with *modest* scenes. The 8 GB VRAM + 30 GB RAM
means: keep scenes simple, avoid path-traced rendering, expect to close other GPU apps.
Large photoreal building floorplans may be too heavy — start small. If Isaac proves
too heavy, Gazebo (ros_gz) is the lighter fallback for the same nav-in-sim goal.

## 1. Pre-flight (cheap, do first — no big download)

```bash
sudo apt update && sudo apt install -y vulkan-tools    # provides vulkaninfo
# Verify NVIDIA is a Vulkan device (run at the physical desktop, not this TTY):
__NV_PRIME_RENDER_OFFLOAD=1 vulkaninfo | grep -i "deviceName"
```
Also confirm the lightweight RViz sim already works (validates the planner loop before Isaac):
```bash
cd ~/interbotix_ws && source install/setup.bash
ros2 launch aloha sim_navigation.launch.py       # click 2D Goal Pose in RViz
```

## 2. DECISION: how will you view the GUI?

This session is headless (TTY). Isaac Sim's editor needs a display. Options:
- **A. Run at the laptop's physical screen** (simplest). Log into the desktop, open a terminal there.
- **B. Headless + WebRTC livestream** — run Isaac Sim headless on the laptop, view from a browser
  on another machine. Heavier setup, good if the laptop is normally accessed remotely.

## 3. DECISION: install method

- **A. pip install (recommended here)** — `isaacsim` wheels into a dedicated venv. Cleanest on
  Ubuntu 22.04 / Py 3.10, no Omniverse Launcher, easy to delete. Best match for this machine.
- **B. Workstation binary / Omniverse** — GUI installer, heavier, more disk.
- Version: **Isaac Sim 4.5** (stable, proven ROS 2 Humble bridge) unless you want 5.0 (open-source, newest).

### 3A pip install sketch (fill in exact version at run time)
```bash
python3.10 -m venv ~/isaacsim_venv
source ~/isaacsim_venv/bin/activate
pip install --upgrade pip
pip install isaacsim[all]==<VERSION> --extra-index-url https://pypi.nvidia.com
# first run downloads shaders/assets (several GB), accept EULA:
isaacsim   # or: python -m isaacsim
```

## 4. Enable the ROS 2 bridge

- In Isaac Sim: Window → Extensions → enable `isaacsim.ros2.bridge` (auto-loads Humble libs).
- Source ROS 2 Humble in the SAME shell BEFORE launching Isaac so the bridge finds it:
  `source /opt/ros/humble/setup.bash` then launch Isaac.
- Sanity check: run Isaac's ROS2 "talker/listener" sample, confirm `ros2 topic list` sees it.
- Set `use_sim_time:=true` on ALL aloha nav nodes; Isaac publishes `/clock`.

## 5. Robot into the sim

Nav only needs the **mobile base footprint + a 2D scan source** — arms are optional at first.
- Import an ALOHA/Tracer-style base URDF via the URDF Importer (check
  `interbotix_ros_manipulators`, `trossen_arm_description`, and Interbotix mobile-base descriptions).
- Add a differential/holonomic drive controller on the base.
- Add a sensor that produces `/scan`: either an Isaac RTX Lidar, or a depth camera +
  `depthimage_to_laserscan` (matches how the real stack builds `/scan`, see NAVIGATION §depth).

## 6. Bridge the topics to the aloha nav stack

Match the real topic contract so the existing planner runs unmodified:
| Signal | Real robot topic | Isaac action-graph node |
|---|---|---|
| Velocity cmd (in) | `/mobile_base/cmd_vel` (Twist) | ROS2 Subscribe Twist → drive controller |
| Odometry (out) | `/odom` | ROS2 Publish Odometry |
| Laser scan (out) | `/scan` | RTX Lidar → ROS2 Publish LaserScan |
| TF (out) | `map→odom→base_link→sensors` | ROS2 Publish Transform Tree |
| Clock (out) | `/clock` | ROS2 Publish Clock |

Then launch localization + planner (`simple_navigation.launch.py` / `navigate_mission`)
pointed at the sim, and send goals via `/goal_pose` exactly as on hardware.

## 7. Scene / map

- Start with a built-in warehouse or a simple room to prove planning + obstacle avoidance.
- Then align a real scene: import `Desktop/mapping/cloud.ply` (point cloud) or reconstruct
  the building; keep the occupancy map (`floorplan_real_2_nav_walls.yaml`) consistent so
  goals/room coordinates line up with the SVG-floorplan nav (see `steps_floorplan_nav.txt`).

## 8. Evaluate planning

- Compare A*/Pure-Pursuit paths vs Nav2 planners in sim, tune inflation/lookahead,
  test dynamic obstacles (drop objects into the path), measure replanning.

---
### Open decisions to confirm before install
1. GUI mode: physical screen (A) vs headless WebRTC (B)?
2. Install method: pip venv (A) vs workstation binary (B)?
3. Isaac Sim version: 4.5 (stable) vs 5.0 (newest/open-source)?
