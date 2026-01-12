# ALOHA Path Planner

MoveIt-based path planning for the ALOHA robotic system with collision avoidance, trajectory optimization, and dual-arm coordination.

## 🚀 Quick Start

### Try in Simulation First! (No Hardware Needed)

Test the path planner safely in simulation:

```bash
# One command launches everything
ros2 launch aloha path_planner_sim.launch.py

# Interactive demo starts automatically
# Try predefined poses, custom goals, and geometric patterns
# See everything visualized in RViz
```

See **[SIMULATION_GUIDE.md](docs/SIMULATION_GUIDE.md)** for complete simulation documentation.

### Installation

```bash
# Install MoveIt2 dependencies
sudo apt install ros-${ROS_DISTRO}-moveit ros-${ROS_DISTRO}-moveit-commander

# Install Python dependencies
cd ~/interbotix_ws/src/aloha
pip3 install -r requirements.txt

# Build the workspace
cd ~/interbotix_ws
colcon build --packages-select aloha
source install/setup.bash
```

### Your First Path Plan

```bash
# Terminal 1: Launch robot
ros2 launch aloha aloha_bringup.launch.py

# Terminal 2: Launch MoveIt (if not already included in bringup)
ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py

# Terminal 3: Run quick start example
ros2 run aloha quick_start_planner
```

## 📚 Documentation

- **[Complete Guide](docs/PATH_PLANNER_GUIDE.md)** - Full API reference and tutorials
- **Examples below** - Code snippets to get started

## 🎯 Features

- ✅ **Joint space planning** - Move to specific joint configurations
- ✅ **Cartesian space planning** - Position end-effector in 3D space  
- ✅ **Waypoint following** - Follow a sequence of poses
- ✅ **Collision avoidance** - Automatic obstacle detection
- ✅ **Dual-arm coordination** - Synchronized bimanual control
- ✅ **Trajectory visualization** - Preview paths in RViz
- ✅ **Velocity/acceleration limiting** - Safe, smooth motion

## 📝 Usage Examples

### Simple Path Planner (Recommended for Beginners)

```python
import rclpy
from aloha.simple_path_planner import SimplePathPlanner

# Initialize
rclpy.init()
node = rclpy.create_node('my_planner')
planner = SimplePathPlanner(node, "interbotix_arm")

# Move to home position
home = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
planner.move_to_joint_goal(home)

# Move to a pose
target = planner.create_pose(0.3, 0.0, 0.2, 0.0, 1.57, 0.0)
planner.move_to_pose_goal(target)

# Follow waypoints
waypoints = [pose1, pose2, pose3]
plan, fraction = planner.plan_cartesian_path(waypoints)
planner.execute_plan(plan)
```

### Advanced Path Planner

```python
from aloha.path_planner import AlohaPathPlanner, PlanningResult

planner = AlohaPathPlanner(
    node,
    planning_time=5.0,
    max_velocity_scaling=0.3
)

# Plan (but don't execute yet)
result, trajectory = planner.plan_to_joint_goal(target_joints)

if result == PlanningResult.SUCCESS:
    # Visualize in RViz first
    planner.visualize_trajectory(trajectory)
    
    # Then execute
    planner.execute_trajectory(trajectory)
elif result == PlanningResult.NO_IK_SOLUTION:
    print("Target pose unreachable!")
```

### Dual Arm Coordination

```python
from aloha.path_planner import AlohaDualArmPlanner

dual_planner = AlohaDualArmPlanner(
    node,
    left_arm_group="left_arm",
    right_arm_group="right_arm"
)

# Synchronized motion
left_joints = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
right_joints = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]

success, left_traj, right_traj = dual_planner.plan_synchronized_motion(
    left_joints, right_joints
)

if success:
    dual_planner.execute_synchronized_motion(left_traj, right_traj)
```

## 🎮 Example Scripts

### Simulation (No Hardware Required)

```bash
# Interactive simulation demo (RECOMMENDED FIRST!)
ros2 launch aloha path_planner_sim.launch.py

# Features:
# - Interactive menu with predefined poses
# - Custom goal input (Cartesian and joint space)
# - Geometric patterns (circles, squares)
# - Real-time RViz visualization
# - Safe testing environment
```

### Real Hardware

Run these after testing in simulation:

```bash
# Quick start (simplest example)
ros2 run aloha quick_start_planner

# Single arm examples (joint/Cartesian/waypoints)
ros2 run aloha path_planner_example

# Dual arm coordination
ros2 run aloha dual_arm_path_planner_example
```

## 🏗️ Architecture

```
aloha/
├── aloha/
│   ├── path_planner.py           # Advanced planner with full features
│   ├── simple_path_planner.py    # Simple moveit_commander wrapper
│   └── constants.py
├── scripts/
│   ├── quick_start_planner.py    # Simplest example
│   ├── path_planner_example.py   # Comprehensive examples
│   └── dual_arm_path_planner_example.py
└── docs/
    └── PATH_PLANNER_GUIDE.md     # Complete documentation
```

## 🔧 Integration with Existing Code

### With VLM Control

```python
from aloha.vlm_controller import VLMController
from aloha.simple_path_planner import SimplePathPlanner

class VLMWithPlanning(VLMController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = SimplePathPlanner(self.node)
    
    def execute_action(self, target_pose):
        # Use collision-free path planning
        return self.planner.move_to_pose_goal(target_pose)
```

### With Teleoperation

```python
from aloha.simple_path_planner import SimplePathPlanner

# Record waypoints during teleop
planner = SimplePathPlanner(node)
waypoints = []

while recording:
    waypoints.append(planner.get_current_pose())

# Replay with smooth interpolation
plan, fraction = planner.plan_cartesian_path(waypoints)
planner.execute_plan(plan)
```

## ⚙️ Configuration

### Planning Parameters

```python
planner.set_velocity_scaling(0.5)      # 0-1 (0.3 recommended)
planner.set_acceleration_scaling(0.5)  # 0-1 (0.3 recommended)
planner.set_planning_time(10.0)        # seconds (5.0 recommended)
```

### MoveIt Configuration

The planner uses your existing MoveIt configuration in:
```
interbotix_xsarm_moveit/config/
```

Default planning group: `interbotix_arm`

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No IK solution" | Target pose outside workspace or unreachable orientation |
| Planning always fails | Increase `planning_time`, check for obstacles |
| Jerky motion | Reduce `velocity_scaling` and `acceleration_scaling` |
| Import errors | Install: `sudo apt install ros-${ROS_DISTRO}-moveit-commander` |

## 📖 API Reference

### SimplePathPlanner

| Method | Description |
|--------|-------------|
| `move_to_joint_goal(joints)` | Move to joint configuration |
| `move_to_pose_goal(pose)` | Move to end-effector pose |
| `plan_cartesian_path(waypoints)` | Plan through waypoints |
| `create_pose(x, y, z, r, p, y)` | Create pose from position/orientation |
| `get_current_joint_values()` | Get current joint angles |
| `get_current_pose()` | Get current end-effector pose |

### AlohaPathPlanner

All SimplePathPlanner methods plus:
- Advanced planning options
- Collision object management
- Constraint-based planning
- Trajectory optimization

See [PATH_PLANNER_GUIDE.md](docs/PATH_PLANNER_GUIDE.md) for complete API.

## 🎯 Common Use Cases

### Pick and Place
```python
approach = planner.create_pose(x, y, z_high, 0, 1.57, 0)
planner.move_to_pose_goal(approach)

grasp = planner.create_pose(x, y, z_low, 0, 1.57, 0)
planner.move_to_pose_goal(grasp)
# gripper.close()

planner.move_to_pose_goal(approach)
```

### Drawing a Pattern
```python
waypoints = [create_circle_points(radius=0.05, num_points=20)]
plan, fraction = planner.plan_cartesian_path(waypoints, eef_step=0.005)
planner.execute_plan(plan)
```

### Obstacle Avoidance
```python
# MoveIt automatically avoids collisions
# Just plan and execute - obstacles are handled automatically
result, traj = planner.plan_to_pose_goal(target)
if result == PlanningResult.SUCCESS:
    planner.execute_trajectory(traj)  # Safe, collision-free path
```

## 🤝 Contributing

To extend the path planner:

1. Add new methods to `AlohaPathPlanner` class
2. Test with example scripts
3. Update documentation
4. Submit PR

## 📄 License

BSD License (same as ALOHA project)

## 🙏 Credits

Built on top of:
- [MoveIt 2](https://moveit.ros.org/) - Motion planning framework
- [Interbotix ROS](https://github.com/Interbotix) - Robot drivers
- [ALOHA](https://github.com/tonyzhaozh/aloha) - Original ALOHA system

## 📚 Learn More

- [MoveIt Tutorials](https://moveit.picknik.ai/main/doc/tutorials/tutorials.html)
- [Complete Path Planner Guide](docs/PATH_PLANNER_GUIDE.md)
- [ALOHA Documentation](README.md)

---

**Happy Planning! 🤖✨**

