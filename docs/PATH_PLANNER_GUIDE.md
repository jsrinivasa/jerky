# ALOHA Path Planner Guide

Comprehensive guide for using MoveIt-based path planning with the ALOHA robotic system.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The ALOHA Path Planner provides a high-level Python interface for motion planning using MoveIt2. It supports:

- **Single arm planning**: Control individual arms with joint or Cartesian space goals
- **Dual arm coordination**: Synchronized bimanual manipulation
- **Collision avoidance**: Automatic collision checking with the environment
- **Trajectory optimization**: Smooth, efficient motion trajectories
- **Flexible planning**: Multiple planning algorithms (RRT, RRTConnect, etc.)

## Features

### Single Arm Planning

- ✅ Joint space planning (specify target joint angles)
- ✅ Cartesian space planning (specify target end-effector pose)
- ✅ Waypoint following (follow a sequence of poses)
- ✅ Trajectory visualization in RViz
- ✅ Adjustable velocity and acceleration limits
- ✅ Configurable planning time and attempts

### Dual Arm Coordination

- ✅ Synchronized motion planning
- ✅ Independent arm control
- ✅ Mirrored motion patterns
- ✅ Bimanual manipulation tasks

### Safety Features

- ✅ Automatic collision detection
- ✅ Joint limit checking
- ✅ Velocity and acceleration scaling
- ✅ Plan validation before execution

---

## Installation

### Prerequisites

Ensure you have MoveIt2 installed in your ROS 2 workspace:

```bash
sudo apt install ros-${ROS_DISTRO}-moveit
```

### Setup

The path planner is already integrated into your ALOHA package. Build your workspace:

```bash
cd ~/interbotix_ws
colcon build --packages-select aloha
source install/setup.bash
```

### Dependencies

The path planner requires:
- `rclpy`
- `moveit_py`
- `geometry_msgs`
- `trajectory_msgs`
- `numpy`
- `scipy`

Install Python dependencies:

```bash
pip3 install numpy scipy
```

---

## Quick Start

### Basic Single Arm Example

```python
import rclpy
from rclpy.node import Node
from aloha.path_planner import AlohaPathPlanner

# Initialize ROS 2
rclpy.init()
node = rclpy.create_node('my_planner_node')

# Create path planner
planner = AlohaPathPlanner(node)

# Plan to home position
home_joints = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
result, trajectory = planner.plan_to_joint_goal(home_joints)

if result == PlanningResult.SUCCESS:
    planner.execute_trajectory(trajectory)

# Spin node
rclpy.spin(node)
```

### Basic Dual Arm Example

```python
from aloha.path_planner import AlohaDualArmPlanner

# Create dual arm planner
dual_planner = AlohaDualArmPlanner(
    node,
    left_arm_group="left_arm",
    right_arm_group="right_arm"
)

# Plan synchronized motion
left_joints = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
right_joints = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]

success, left_traj, right_traj = dual_planner.plan_synchronized_motion(
    left_joints, right_joints
)

if success:
    dual_planner.execute_synchronized_motion(left_traj, right_traj)
```

---

## API Reference

### AlohaPathPlanner Class

#### Initialization

```python
planner = AlohaPathPlanner(
    node,                              # ROS 2 node
    arm_group="interbotix_arm",       # MoveIt planning group
    gripper_group="interbotix_gripper",
    planning_time=5.0,                # Max planning time (seconds)
    num_planning_attempts=10,         # Number of attempts
    max_velocity_scaling=0.3,         # Velocity scale (0-1)
    max_acceleration_scaling=0.3      # Acceleration scale (0-1)
)
```

#### Core Methods

##### `plan_to_joint_goal(joint_positions)`

Plan a trajectory to reach specified joint positions.

**Parameters:**
- `joint_positions` (List[float]): Target joint angles in radians

**Returns:**
- `(PlanningResult, Optional[RobotTrajectory])`: Result status and trajectory

**Example:**
```python
result, traj = planner.plan_to_joint_goal([0.0, -0.5, 0.8, 0.0, -0.3, 0.0])
```

---

##### `plan_to_pose_goal(target_pose)`

Plan a trajectory to reach specified end-effector pose.

**Parameters:**
- `target_pose` (Pose): Target pose for end-effector
- `start_state` (Optional[RobotState]): Starting state (current if None)
- `reference_frame` (str): Reference frame (default: "world")

**Returns:**
- `(PlanningResult, Optional[RobotTrajectory])`: Result status and trajectory

**Example:**
```python
target = planner.create_pose(0.3, 0.0, 0.2, 0.0, 1.57, 0.0)
result, traj = planner.plan_to_pose_goal(target)
```

---

##### `plan_cartesian_path(waypoints)`

Plan a Cartesian path through multiple waypoints.

**Parameters:**
- `waypoints` (List[Pose]): List of poses to follow
- `eef_step` (float): Interpolation step size (meters)
- `jump_threshold` (float): Max joint space jump
- `avoid_collisions` (bool): Check for collisions

**Returns:**
- `(PlanningResult, Optional[RobotTrajectory], float)`: Result, trajectory, and fraction achieved

**Example:**
```python
waypoints = [pose1, pose2, pose3]
result, traj, fraction = planner.plan_cartesian_path(waypoints, eef_step=0.01)
```

---

##### `execute_trajectory(trajectory)`

Execute a planned trajectory.

**Parameters:**
- `trajectory` (Optional[RobotTrajectory]): Trajectory (uses last if None)

**Returns:**
- `bool`: True if execution successful

**Example:**
```python
success = planner.execute_trajectory(trajectory)
```

---

##### `plan_and_execute_to_joint_goal(joint_positions)`

Plan and execute in one call (convenience method).

**Example:**
```python
success = planner.plan_and_execute_to_joint_goal([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
```

---

#### Utility Methods

##### `create_pose(x, y, z, roll, pitch, yaw)`

Create a Pose from position and Euler angles.

**Example:**
```python
pose = planner.create_pose(0.3, 0.1, 0.2, 0.0, 1.57, 0.0)
```

---

##### `get_current_joint_values()`

Get current joint positions.

**Returns:**
- `List[float]`: Current joint angles in radians

---

##### `get_current_pose()`

Get current end-effector pose.

**Returns:**
- `Pose`: Current pose of end-effector

---

##### `set_planning_parameters(...)`

Update planning parameters dynamically.

**Example:**
```python
planner.set_planning_parameters(
    planning_time=10.0,
    velocity_scaling=0.5,
    acceleration_scaling=0.5
)
```

---

##### `visualize_trajectory(trajectory)`

Publish trajectory for visualization in RViz.

---

### AlohaDualArmPlanner Class

#### Initialization

```python
dual_planner = AlohaDualArmPlanner(
    node,
    left_arm_group="left_arm",
    right_arm_group="right_arm",
    **kwargs  # Other AlohaPathPlanner parameters
)
```

#### Methods

##### `plan_synchronized_motion(left_joints, right_joints)`

Plan synchronized motion for both arms.

**Returns:**
- `(bool, Optional[RobotTrajectory], Optional[RobotTrajectory])`: Success flag and trajectories

---

##### `execute_synchronized_motion(left_traj, right_traj)`

Execute synchronized motion for both arms.

---

### PlanningResult Enum

```python
class PlanningResult(Enum):
    SUCCESS = 0            # Planning successful
    FAILURE = 1            # General failure
    INVALID_GOAL = 2       # Invalid goal configuration
    TIMEOUT = 3            # Planning timeout
    NO_IK_SOLUTION = 4     # No inverse kinematics solution
    COLLISION = 5          # Collision detected
```

---

## Examples

### Example 1: Simple Pick and Place

```python
import rclpy
from aloha.path_planner import AlohaPathPlanner, PlanningResult

rclpy.init()
node = rclpy.create_node('pick_and_place')
planner = AlohaPathPlanner(node)

# Approach object
approach = planner.create_pose(0.3, 0.1, 0.15, 0, 1.57, 0)
result, _ = planner.plan_and_execute_to_pose_goal(approach)

# Lower to grasp
grasp = planner.create_pose(0.3, 0.1, 0.05, 0, 1.57, 0)
planner.plan_and_execute_to_pose_goal(grasp)

# Close gripper here
# gripper.close()

# Lift
planner.plan_and_execute_to_pose_goal(approach)

# Move to place location
place = planner.create_pose(0.3, -0.1, 0.15, 0, 1.57, 0)
planner.plan_and_execute_to_pose_goal(place)

# Open gripper
# gripper.open()
```

### Example 2: Circular Motion

```python
import numpy as np

# Create circular waypoints
center_x, center_y, z = 0.3, 0.0, 0.15
radius = 0.05
num_points = 20

waypoints = []
for i in range(num_points):
    angle = 2 * np.pi * i / num_points
    x = center_x + radius * np.cos(angle)
    y = center_y + radius * np.sin(angle)
    waypoints.append(planner.create_pose(x, y, z, 0, 1.57, 0))

# Plan and execute
result, traj, fraction = planner.plan_cartesian_path(waypoints)
if fraction > 0.95:
    planner.execute_trajectory(traj)
```

### Example 3: Dual Arm Coordination

```python
from aloha.path_planner import AlohaDualArmPlanner

dual_planner = AlohaDualArmPlanner(node)

# Bring both arms to meet at center
left_pose = dual_planner.left_planner.create_pose(0.25, 0.1, 0.15, 0, 1.57, 0)
right_pose = dual_planner.right_planner.create_pose(0.25, -0.1, 0.15, 0, 1.57, 0)

# Plan for both arms
left_result, left_traj = dual_planner.left_planner.plan_to_pose_goal(left_pose)
right_result, right_traj = dual_planner.right_planner.plan_to_pose_goal(right_pose)

# Execute simultaneously
if left_result == PlanningResult.SUCCESS and right_result == PlanningResult.SUCCESS:
    dual_planner.execute_synchronized_motion(left_traj, right_traj)
```

---

## Best Practices

### 1. Planning Parameters

**Start Conservative:**
```python
planner.set_planning_parameters(
    planning_time=5.0,
    velocity_scaling=0.2,
    acceleration_scaling=0.2
)
```

**Increase for Production:**
```python
planner.set_planning_parameters(
    velocity_scaling=0.5,
    acceleration_scaling=0.5
)
```

### 2. Always Check Planning Results

```python
result, traj = planner.plan_to_joint_goal(target)

if result == PlanningResult.SUCCESS:
    planner.execute_trajectory(traj)
elif result == PlanningResult.NO_IK_SOLUTION:
    print("Target pose unreachable!")
elif result == PlanningResult.COLLISION:
    print("Path would cause collision!")
else:
    print(f"Planning failed: {result}")
```

### 3. Visualize Before Executing

```python
result, traj = planner.plan_to_pose_goal(target)
if result == PlanningResult.SUCCESS:
    planner.visualize_trajectory(traj)
    input("Review trajectory in RViz, then press Enter to execute...")
    planner.execute_trajectory(traj)
```

### 4. Use Cartesian Paths for Straight-Line Motion

For tasks requiring straight-line motion (e.g., insertion, drawing):

```python
waypoints = [start_pose, end_pose]
result, traj, fraction = planner.plan_cartesian_path(
    waypoints,
    eef_step=0.005  # 5mm steps for smooth motion
)

# Only execute if path is fully achieved
if fraction >= 0.99:
    planner.execute_trajectory(traj)
```

### 5. Add Intermediate Waypoints for Complex Paths

```python
# Instead of direct motion, add intermediate waypoint
current = planner.get_current_pose()
intermediate = planner.create_pose(
    (current.position.x + target.position.x) / 2,
    (current.position.y + target.position.y) / 2,
    max(current.position.z, target.position.z) + 0.05  # Go higher
)

waypoints = [current, intermediate, target]
result, traj, fraction = planner.plan_cartesian_path(waypoints)
```

### 6. Handle Joint Limits

```python
limits = planner.get_joint_limits()
for joint_name, (min_val, max_val) in limits.items():
    print(f"{joint_name}: [{min_val:.2f}, {max_val:.2f}]")

# Ensure target is within limits
target_joints = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
for i, (joint_name, (min_val, max_val)) in enumerate(limits.items()):
    target_joints[i] = np.clip(target_joints[i], min_val, max_val)
```

---

## Troubleshooting

### Issue: Planning Always Fails

**Possible causes:**
1. Target pose is unreachable (outside workspace)
2. Target would cause collision
3. Planning time too short

**Solutions:**
```python
# Increase planning time and attempts
planner.set_planning_parameters(planning_time=10.0)

# Check if pose is reachable
current_pose = planner.get_current_pose()
print(f"Current: {current_pose.position.x:.2f}, {current_pose.position.y:.2f}")

# Try different target
```

### Issue: Jerky Motion

**Cause:** High velocity/acceleration scaling

**Solution:**
```python
planner.set_planning_parameters(
    velocity_scaling=0.2,  # Reduce from 0.5
    acceleration_scaling=0.2
)
```

### Issue: "No IK Solution" Error

**Cause:** Target pose orientation or position unreachable

**Solution:**
```python
# Keep orientation closer to current
current = planner.get_current_pose()
target = planner.create_pose(
    x, y, z,
    # Use current orientation
    roll=0.0, pitch=1.57, yaw=0.0
)
```

### Issue: Collision Detected

**Cause:** Path intersects with obstacles or self-collision

**Solution:**
1. Visualize scene in RViz
2. Add intermediate waypoints to avoid obstacles
3. Update planning scene with correct collision objects

---

## Integration with Existing ALOHA Code

### Using with VLM Control

```python
from aloha.vlm_controller import VLMController
from aloha.path_planner import AlohaPathPlanner

class VLMWithPlanning(VLMController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = AlohaPathPlanner(self.node)
    
    def execute_action(self, target_pose):
        # Use path planner instead of direct motion
        result, traj = self.planner.plan_to_pose_goal(target_pose)
        if result == PlanningResult.SUCCESS:
            return self.planner.execute_trajectory(traj)
        return False
```

### Using with Teleoperation

```python
# In your teleop script
planner = AlohaPathPlanner(node)

# When recording waypoints
waypoints = []
while recording:
    current_pose = planner.get_current_pose()
    waypoints.append(current_pose)

# Replay with smooth planning
result, traj, fraction = planner.plan_cartesian_path(waypoints)
planner.execute_trajectory(traj)
```

---

## Running the Examples

### Single Arm Demo

```bash
ros2 run aloha path_planner_example.py
```

### Dual Arm Demo

```bash
ros2 run aloha dual_arm_path_planner_example.py
```

### With Visualization

In separate terminals:

```bash
# Terminal 1: Launch robot
ros2 launch aloha aloha_bringup.launch.py

# Terminal 2: Launch MoveIt
ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py

# Terminal 3: Run planner
ros2 run aloha path_planner_example.py
```

---

## Additional Resources

- [MoveIt 2 Documentation](https://moveit.picknik.ai/main/index.html)
- [MoveIt Python API](https://github.com/ros-planning/moveit2/tree/main/moveit_py)
- [Interbotix ROS Documentation](http://docs.trossenrobotics.com/)

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review MoveIt logs: `ros2 topic echo /rosout`
3. Visualize planning scene in RViz
4. Open an issue in the repository

---

**Happy Planning! 🤖**

