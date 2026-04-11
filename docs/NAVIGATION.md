# Simple Navigation Path Planner

A straightforward path planning and navigation system for your ALOHA mobile base.

## Features

- **Simple A* Path Planning**: Plans collision-free paths on your saved map
- **Pure Pursuit Controller**: Smooth path following with configurable lookahead distance
- **RViz2 Integration**: Set goals using the "2D Goal Pose" tool
- **Map-based Navigation**: Uses your saved RTAB-Map map
- **Lightweight**: No complex Nav2 setup required (though Nav2 support is optional)

## Prerequisites

Make sure you have:
1. A saved map
2. D435i camera working
3. Robot odometry publishing to `/odom`

## Quick Start

### 1. Launch Navigation

With default map location:
```bash
ros2 launch aloha simple_navigation.launch.py
```

With custom map:
```bash
ros2 launch aloha simple_navigation.launch.py map_file:=/path/to/your/map.yaml
```

### 2. Set Goals in RViz2

1. Wait for RViz2 to open
2. Click the "2D Goal Pose" button in the toolbar
3. Click on the map where you want the robot to go
4. Drag to set the desired orientation
5. Release to send the goal

The robot will:
- Plan a path (green line)
- Display waypoints (orange spheres)
- Follow the path autonomously
- Stop when it reaches the goal

## Parameters

Adjust behavior by passing parameters:

```bash
ros2 launch aloha simple_navigation.launch.py \
    max_linear_velocity:=0.5 \
    max_angular_velocity:=1.5 \
    lookahead_distance:=0.7 \
    goal_tolerance:=0.15
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `map_file` | `maps/my_map.yaml` | Path to your map file |
| `use_nav2` | `false` | Use Nav2 instead of simple planner |
| `lookahead_distance` | `0.5` | Lookahead distance for path following (m) |
| `max_linear_velocity` | `0.3` | Maximum forward speed (m/s) |
| `max_angular_velocity` | `1.0` | Maximum turning speed (rad/s) |
| `goal_tolerance` | `0.2` | Distance to goal for completion (m) |
| `use_sim_time` | `false` | Use simulation time |
| `use_rviz` | `true` | Launch RViz2 |

## Running with Nav2 (Optional)

If you prefer to use the full Nav2 stack:

```bash
# Launch Nav2 first
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=false \
    params_file:=/path/to/nav2_params.yaml

# Then launch with Nav2 enabled
ros2 launch aloha simple_navigation.launch.py use_nav2:=true
```

## Standalone Node

You can also run the navigation planner as a standalone node:

```bash
ros2 run aloha simple_nav_planner
```

Then set goals by publishing to `/goal_pose`:
```bash
ros2 topic pub /goal_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: 'map'}, \
      pose: {position: {x: 1.0, y: 2.0, z: 0.0}, \
             orientation: {w: 1.0}}}"
```

## Topics

### Subscribed Topics
- `/goal_pose` (geometry_msgs/PoseStamped): Goal pose from RViz2 or command line
- `/map` (nav_msgs/OccupancyGrid): Map for path planning
- `/odom` (nav_msgs/Odometry): Robot odometry

### Published Topics
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands to robot
- `/planned_path` (nav_msgs/Path): Planned path for visualization
- `/nav_markers` (visualization_msgs/MarkerArray): Path markers for RViz

## Troubleshooting

### "No map data yet!"
- Make sure you've created and saved a map first
- Check that `map_file` parameter points to correct location
- Verify map server is running: `ros2 topic echo /map --once`

### "No odometry data yet!"
- Check odometry topic: `ros2 topic echo /odom`
- Make sure RTAB-Map odometry or wheel odometry is running

### "Path planning failed!"
- Goal might be in occupied space or too close to obstacles
- Try setting goal in open space
- Check map visualization in RViz

### Robot doesn't move
- Verify cmd_vel is being published: `ros2 topic echo /cmd_vel`
- Check if another node is controlling the robot
- Make sure robot motors are enabled

### Robot oscillates or doesn't follow path well
- Reduce `max_linear_velocity` and `max_angular_velocity`
- Increase `lookahead_distance` for smoother but wider turns
- Decrease `lookahead_distance` for tighter following

## Algorithm Details

### Path Planning (A*)
- Uses classic A* algorithm on occupancy grid
- 8-connected grid (diagonal movement allowed)
- Heuristic: Euclidean distance to goal
- Path simplification removes redundant waypoints

### Path Following (Pure Pursuit)
- Finds lookahead point on path ahead of robot
- Calculates angular velocity to steer toward lookahead
- Reduces linear velocity during sharp turns
- Updates path index as robot progresses

## Comparison: Simple Planner vs Nav2

| Feature | Simple Planner | Nav2 |
|---------|---------------|------|
| Setup Complexity | Low | High |
| Path Planning | A* | Multiple algorithms |
| Local Planning | Pure Pursuit | DWB, TEB, etc. |
| Recovery Behaviors | None | Spin, backup, wait |
| Costmap Layers | Single map | Inflation, obstacles |
| Performance | Good for simple envs | Better for complex |
| Configuration | Few parameters | 100+ parameters |

**Recommendation**: Start with the simple planner. Switch to Nav2 if you need:
- Dynamic obstacle avoidance
- Recovery behaviors
- More sophisticated planning
- Multiple robot coordination

## Next Steps

Once this is working, you can:
1. Add waypoint navigation (visit multiple points)
2. Integrate with higher-level task planning
3. Add obstacle detection and dynamic avoidance
4. Create patrol/coverage patterns
5. Add autonomous exploration

## Files Created

- `aloha/simple_nav_planner.py`: Main navigation planner node
- `launch/simple_navigation.launch.py`: Launch file
- `scripts/simple_nav_planner_node.py`: Executable script
- `rviz/navigation.rviz`: RViz configuration

