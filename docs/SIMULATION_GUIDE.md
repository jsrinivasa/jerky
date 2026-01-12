# ALOHA Path Planner Simulation Guide

Complete guide for testing the path planner in simulation before using it on real hardware.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Simulation Modes](#simulation-modes)
- [Interactive Demo](#interactive-demo)
- [RViz Visualization](#rviz-visualization)
- [Testing Your Own Code](#testing-your-own-code)
- [Troubleshooting](#troubleshooting)

---

## Overview

The simulation environment allows you to:
- ✅ **Test path planning** without hardware
- ✅ **Visualize trajectories** in RViz before execution
- ✅ **Experiment safely** with different goals
- ✅ **Debug motion plans** interactively
- ✅ **Develop offline** without robot access

### Two Simulation Options

1. **Fake Hardware (Recommended)** - Lightweight, no physics
   - Fast startup (~5 seconds)
   - Low CPU/memory usage
   - Perfect for path planning testing
   - No collision simulation

2. **Gazebo Classic** - Full physics simulation
   - Realistic physics
   - Collision simulation
   - Slower startup (~15 seconds)
   - Higher resource usage

---

## Quick Start

### Option 1: Fake Hardware (Easiest)

```bash
# Terminal 1: Launch simulation with MoveIt + Interactive demo
ros2 launch aloha path_planner_sim.launch.py

# That's it! The interactive demo will start automatically.
```

### Option 2: Gazebo Simulation

```bash
# Terminal 1: Launch with Gazebo
ros2 launch aloha path_planner_sim.launch.py use_gazebo:=true

# Wait for Gazebo to fully load (~15 seconds)
# The interactive demo will start automatically
```

### What You'll See

1. **RViz window** opens showing:
   - Robot model
   - Planning scene
   - Interactive markers
   - Trajectory visualization

2. **Terminal shows** interactive menu:
```
  Interactive Path Planner Menu
====================================================================

📍 Predefined Poses:
  1. Home position
  2. Forward reach
  3. Left reach
  4. Right reach
  5. High reach
  6. Low reach
  7. Upright position (all zeros)

🎯 Custom Goals:
  8. Enter custom Cartesian pose (x, y, z)
  9. Enter custom joint angles

🎨 Demonstrations:
  10. Draw a circle
  11. Draw a square
  12. Auto demo (all predefined poses)

📊 Information:
  i. Show current state
  h. Show workspace limits

  q. Quit
====================================================================

Your choice:
```

---

## Simulation Modes

### Fake Hardware Mode (Default)

**Best for:** Path planning development, quick testing

```bash
ros2 launch aloha path_planner_sim.launch.py
```

**Features:**
- Instant feedback
- No physics overhead
- Joint states follow commands perfectly
- Great for algorithm testing

**Limitations:**
- No collision physics
- No gravity/dynamics
- Simplified motion

### Gazebo Mode

**Best for:** Realistic testing, collision validation

```bash
ros2 launch aloha path_planner_sim.launch.py use_gazebo:=true
```

**Features:**
- Full physics simulation
- Accurate collision detection
- Realistic joint dynamics
- Gravity and inertia

**Limitations:**
- Slower startup
- Higher CPU usage
- Requires more memory

### Specify Robot Model

```bash
# For different robot models
ros2 launch aloha path_planner_sim.launch.py robot_model:=wx200
ros2 launch aloha path_planner_sim.launch.py robot_model:=vx300s
```

---

## Interactive Demo

### Using the Interactive Menu

Once launched, you have several options:

#### 1. Predefined Poses

Try safe, tested poses to understand workspace:

```
Your choice: 1    # Go to home position
Your choice: 2    # Reach forward
Your choice: 3    # Reach left
```

**What happens:**
- Path planner computes trajectory
- Trajectory is visualized in RViz
- Robot executes motion
- Status is printed to terminal

#### 2. Custom Cartesian Goals

Enter your own 3D positions:

```
Your choice: 8
  x (meters, forward, e.g., 0.3): 0.3
  y (meters, left/right, e.g., 0.0): 0.1
  z (meters, height, e.g., 0.15): 0.2
  
  Orientation (press Enter for default: pointing down)
  roll (radians, default 0.0): 
  pitch (radians, default 1.57 [π/2]): 
  yaw (radians, default 0.0): 
```

**Tips:**
- Start with positions you know are reachable
- Use `h` command to see workspace limits
- Default orientation (pointing down) usually works

#### 3. Custom Joint Angles

Specify exact joint configurations:

```
Your choice: 9
  Use degrees? (y/n, default n): y
  waist: 0
  shoulder: -45
  elbow: 90
  forearm_roll: 0
  wrist_angle: -30
  wrist_rotate: 0
```

#### 4. Geometric Patterns

Test Cartesian path planning:

**Circle:**
```
Your choice: 10
  Radius (meters, default 0.05): 0.06
  Number of points (default 16): 20
```

**Square:**
```
Your choice: 11
  Size (meters, default 0.1): 0.08
```

**What happens:**
- Waypoints are generated
- Cartesian path is computed
- Trajectory shows smooth interpolation
- Robot follows the path

#### 5. Auto Demo

See all capabilities automatically:

```
Your choice: 12
```

This will:
1. Go to home
2. Visit all predefined poses
3. Return to home
4. Take 30-60 seconds total

#### 6. Information Commands

**Current State** (`i`):
```
Your choice: i

  Current Robot State
====================================================================

📐 Joint Angles (radians):
  waist          :   0.000 rad (  0.00°)
  shoulder       :  -0.960 rad (-55.01°)
  elbow          :   1.160 rad ( 66.46°)
  forearm_roll   :   0.000 rad (  0.00°)
  wrist_angle    :  -0.300 rad (-17.19°)
  wrist_rotate   :   0.000 rad (  0.00°)

📍 End-Effector Pose:
  Position:
    x:   0.287 m
    y:   0.000 m
    z:   0.201 m
  Orientation (quaternion):
    x:   0.707
    y:   0.000
    z:   0.000
    w:   0.707
```

**Workspace Limits** (`h`):
```
Your choice: h

  Workspace Limits (Typical for wx250s)
====================================================================

📏 Position Limits:
  x: 0.1 to 0.5 m (forward)
  y: -0.3 to 0.3 m (left/right)
  z: -0.1 to 0.4 m (height)

💡 Recommended Safe Zone:
  x: 0.2 to 0.4 m
  y: -0.2 to 0.2 m
  z: 0.05 to 0.3 m

⚠️  Goals outside this range may not be reachable!
```

---

## RViz Visualization

### What You See in RViz

1. **Robot Model** (gray/orange)
   - Current configuration
   - Updates in real-time

2. **Planning Scene** (green grid)
   - Collision objects
   - Workspace boundaries

3. **Planned Trajectory** (animated line)
   - Shows path before execution
   - Color indicates time/speed

4. **Goal Markers** (red arrow + label)
   - Shows where robot is planning to go
   - Appears when you select a goal

### RViz Configuration

The simulation automatically loads with proper visualization settings. You can customize:

**Views:**
- Orbit around robot: Middle mouse button
- Zoom: Scroll wheel
- Pan: Shift + middle mouse button

**Display Options:**
- Toggle robot mesh: Check/uncheck "RobotModel"
- Show planning scene: Check "PlanningScene"
- Show trajectory: Check "Trajectory"

### Capturing What You See

**Screenshots:**
- RViz menu → File → Save Screenshot

**Videos:**
```bash
# Record RViz window
ros2 run image_tools cam2image  # If needed
```

---

## Testing Your Own Code

### Write Test Scripts

Create `my_sim_test.py`:

```python
#!/usr/bin/env python3
import rclpy
from aloha.simple_path_planner import SimplePathPlanner

def test_my_motions():
    rclpy.init()
    node = rclpy.create_node('my_sim_test')
    
    # Create planner (same code as real robot!)
    planner = SimplePathPlanner(node, "interbotix_arm")
    
    # Your test sequence
    planner.move_to_joint_goal([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    
    target = planner.create_pose(0.3, 0.1, 0.15, 0, 1.57, 0)
    planner.move_to_pose_goal(target)
    
    print("Test complete!")
    rclpy.shutdown()

if __name__ == '__main__':
    test_my_motions()
```

Run it:
```bash
# Terminal 1: Launch simulation
ros2 launch aloha path_planner_sim.launch.py

# Terminal 2: Run your test
python3 my_sim_test.py
```

### Test Before Real Robot

**Simulation → Reality Checklist:**

1. ✅ Test in simulation first
2. ✅ Verify all motions complete successfully
3. ✅ Check trajectories look safe
4. ✅ Confirm no unexpected movements
5. ✅ Same code works in sim? → Try on real robot!

### Using Same Code for Sim and Real

Your code doesn't need to change!

```python
# This code works in BOTH simulation AND real hardware
from aloha.simple_path_planner import SimplePathPlanner

planner = SimplePathPlanner(node, "interbotix_arm")
planner.move_to_pose_goal(target)
```

**Switch between sim and real by changing the launch file:**

```bash
# Simulation
ros2 launch aloha path_planner_sim.launch.py

# Real robot
ros2 launch aloha aloha_bringup.launch.py
```

---

## Advanced Usage

### Launch Components Separately

If you want more control:

```bash
# Terminal 1: Robot simulation only
ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py \
    robot_model:=wx250s \
    hardware_type:=fake \
    use_rviz:=true

# Terminal 2: Your custom demo
ros2 run aloha sim_planner_interactive

# Or your own script
python3 my_test_script.py
```

### Test Specific Scenarios

**Test collision avoidance:**
```bash
# Add obstacles in RViz:
# - Click "PlanningScene" → "Scene Objects" → "Add"
# - Define box/sphere/cylinder
# - Plan around it
```

**Test different speeds:**
```python
planner.set_velocity_scaling(0.1)  # Very slow
planner.move_to_pose_goal(target)

planner.set_velocity_scaling(0.5)  # Faster
planner.move_to_pose_goal(target)
```

**Test failure cases:**
```python
# Try unreachable goal (should fail gracefully)
far_away = planner.create_pose(2.0, 0.0, 0.5, 0, 0, 0)
success = planner.move_to_pose_goal(far_away)
if not success:
    print("Correctly detected unreachable goal!")
```

---

## Troubleshooting

### Issue: "Failed to initialize planner"

**Cause:** MoveIt not ready

**Solution:**
```bash
# Make sure simulation launched successfully
# Check for errors in terminal
# Wait 5-10 seconds after launch before testing
```

### Issue: "No IK solution found" in simulation

**Cause:** Goal is outside workspace

**Solution:**
- Use `h` command to see limits
- Try goals closer to robot
- Check orientation is reasonable

### Issue: Robot moves strangely

**Cause:** Joint limits or singularities

**Solution:**
- Use predefined poses first (they're tested safe)
- Avoid extreme joint angles
- Stay within recommended workspace

### Issue: RViz doesn't show trajectory

**Check:**
1. RViz "Trajectory" display is enabled
2. Planning actually succeeded (check terminal)
3. Topic is correct: `/display_planned_path`

```bash
# Verify topic is publishing
ros2 topic echo /display_planned_path --once
```

### Issue: Simulation is slow/laggy

**Gazebo mode:**
```bash
# Use fake hardware instead
ros2 launch aloha path_planner_sim.launch.py use_gazebo:=false
```

**RViz is slow:**
- Reduce visual quality in RViz
- Disable unnecessary displays
- Close other applications

### Issue: "moveit_commander not found"

**Solution:**
```bash
sudo apt install ros-${ROS_DISTRO}-moveit-commander
```

---

## Example Session

Here's a complete example session:

```bash
# Terminal 1: Launch simulation
$ ros2 launch aloha path_planner_sim.launch.py

# Wait for initialization...
# Interactive menu appears

# Try home position
Your choice: 1
🎯 Executing: Home position
✅ Goal reached!

# Check current state
Your choice: i
📐 Joint Angles: ...
📍 Position: x: 0.287, y: 0.000, z: 0.201

# Try forward reach
Your choice: 2
🎯 Executing: Reaching forward
✅ Goal reached!

# Draw a circle
Your choice: 10
  Radius: 0.05
  Number of points: 16
✓ Path planned: 100.0% achieved
✅ Circle completed!

# Custom pose
Your choice: 8
  x: 0.35
  y: 0.05
  z: 0.12
  [using defaults for orientation]
🎯 Planning to: x=0.350, y=0.050, z=0.120
✅ Goal reached!

# Quit
Your choice: q
👋 Goodbye!
```

---

## Next Steps

### After Testing in Simulation

1. **Confident with results?** → Try on real hardware
2. **Want to customize?** → Modify the interactive demo script
3. **Need more features?** → Check the full PATH_PLANNER_GUIDE.md
4. **Ready for dual-arm?** → See dual_arm examples

### Real Hardware Transition

```bash
# 1. Test everything in simulation ✓
# 2. Review safety procedures
# 3. Launch real robot:
ros2 launch aloha aloha_bringup.launch.py

# 4. Run SAME CODE (it just works!)
ros2 run aloha quick_start_planner
```

---

## Tips for Effective Simulation Testing

1. **Start simple:** Home position → nearby goals → complex motions
2. **Visualize first:** Always watch in RViz before executing
3. **Test edge cases:** Try workspace boundaries, extreme orientations
4. **Use information commands:** `i` and `h` are your friends
5. **Build confidence gradually:** Predefined → custom → your code

---

## Resources

- **This guide:** Simulation-specific tips
- **Path Planner Guide:** `PATH_PLANNER_GUIDE.md` - Full API
- **Quick Reference:** `PATH_PLANNER_README.md`
- **MoveIt Docs:** https://moveit.picknik.ai/

---

**Ready to simulate? Let's go!** 🚀

```bash
ros2 launch aloha path_planner_sim.launch.py
```


