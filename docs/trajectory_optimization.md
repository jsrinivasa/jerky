# Trajectory Optimization Guide

This document describes the trajectory optimization features added to the ALOHA navigation system.

## Overview

The navigation planner now includes advanced trajectory optimization that:
- **Smooths paths** using cubic B-spline interpolation
- **Optimizes velocity profiles** for time-optimal trajectories
- **Visualizes trajectories** in RViz2 with multiple views
- **Considers robot dynamics** (velocity/acceleration constraints)

## Features

### 1. Cubic B-Spline Path Smoothing

Raw A* paths contain sharp corners and are non-differentiable. The B-spline smoothing:
- Creates smooth, continuous trajectories
- Maintains collision-free constraints
- Interpolates through the original waypoints
- Generates dense waypoint sequences for better tracking

### 2. Velocity Profile Optimization

The velocity optimizer:
- Calculates optimal velocities based on path curvature
- Respects maximum velocity and acceleration limits
- Slows down for sharp turns
- Creates time-stamped trajectory with velocity at each point

### 3. Enhanced RViz2 Visualization

The system publishes multiple visualization layers:

| Topic | Description | Color | Line Width |
|-------|-------------|-------|------------|
| `/raw_path` | Original A* path | Gray | 0.03m |
| `/smoothed_path` | Optimized trajectory | Green | 0.08m |
| `/nav_markers` | Enhanced markers | Various | - |

Additional markers:
- **Start point**: Blue sphere
- **Goal point**: Red sphere
- **Velocity arrows**: Color-coded by speed (red=slow, green=fast)
- **Path comparison**: Both raw and smoothed paths visible simultaneously

## Parameters

Launch parameters for trajectory optimization:

```bash
ros2 launch aloha sim_navigation.launch.py \
    use_trajectory_optimization:=true \
    smoothing_weight:=0.5 \
    max_acceleration:=0.5 \
    max_linear_velocity:=0.3
```

### Parameter Details

- **`use_trajectory_optimization`** (default: `true`)
  - Enable/disable trajectory smoothing and optimization
  - If false, uses raw A* path

- **`smoothing_weight`** (default: `0.5`, range: 0.0-1.0)
  - Controls smoothness vs. path accuracy
  - Higher values = smoother but may deviate from original path
  - Lower values = closer to original but less smooth
  - Recommended: 0.3-0.7

- **`max_acceleration`** (default: `0.5` m/s²)
  - Maximum linear acceleration
  - Used for velocity profile optimization
  - Lower values = gentler acceleration

- **`max_linear_velocity`** (default: `0.3` m/s)
  - Maximum linear velocity
  - Velocity is automatically reduced in curves

- **`lookahead_distance`** (default: `0.5` m)
  - Pure pursuit controller lookahead distance
  - Higher values work better with smoother trajectories

## Usage

### Basic Usage

1. Launch the navigation system:
```bash
cd ~/interbotix_ws
source install/setup.bash
ros2 launch aloha sim_navigation.launch.py
```

2. In RViz2, use the **"2D Goal Pose"** tool to set a goal

3. Observe the visualization:
   - Gray dashed line: Original A* path
   - Green solid line: Smoothed trajectory
   - Colored arrows: Velocity profile

### Customizing Smoothing

For tighter spaces (more accuracy needed):
```bash
ros2 launch aloha sim_navigation.launch.py smoothing_weight:=0.2
```

For open spaces (more smoothness):
```bash
ros2 launch aloha sim_navigation.launch.py smoothing_weight:=0.8
```

### Disabling Optimization

To compare with raw A* paths:
```bash
ros2 launch aloha sim_navigation.launch.py use_trajectory_optimization:=false
```

## Visualization in RViz2

The RViz configuration includes:

1. **Raw Path (A*)** - Gray, thin line showing original path
2. **Smoothed Trajectory** - Green, thick line showing optimized path
3. **Velocity Profile** - Arrows showing speed at each point
4. **Start/Goal Markers** - Blue sphere (start), Red sphere (goal)
5. **Map** - Occupancy grid
6. **TF Frames** - Robot coordinate frames

## Algorithm Details

### B-Spline Smoothing

The system uses cubic (degree 3) B-splines with:
- Parametric representation: `(x(u), y(u))` where `u ∈ [0, 1]`
- Smoothness parameter `s` controlled by `smoothing_weight`
- Collision checking at interpolated points
- Fallback to raw path if smoothed path collides

### Velocity Optimization

The velocity profile optimizer:
1. Calculates path curvature at each segment
2. Computes maximum safe velocity: `v_max = sqrt(a_max / κ)`
3. Applies acceleration constraints
4. Generates time-optimal trajectory

### Collision Checking

During smoothing, the system:
- Checks if each interpolated point is collision-free
- Considers robot radius (configurable)
- Falls back to raw path if collision detected

## Troubleshooting

### Trajectory goes through obstacles

**Problem**: Smoothed path collides with obstacles

**Solution**:
- Reduce `smoothing_weight` (try 0.2-0.3)
- Increase `robot_radius` parameter
- Check map resolution and quality

### Robot oscillates on path

**Problem**: Robot weaves back and forth

**Solution**:
- Increase `lookahead_distance` (try 0.7-1.0)
- Reduce `smoothing_weight` for straighter paths
- Check angular velocity limits

### Path looks jagged despite smoothing

**Problem**: Smoothed path not smooth enough

**Solution**:
- Increase `smoothing_weight` (try 0.6-0.8)
- Ensure scipy is installed: `pip install scipy`
- Check for console errors about spline fitting

### Slow planning

**Problem**: Path planning takes too long

**Solution**:
- Smoothing adds minimal overhead (~50-100ms)
- Check A* performance (most time spent here)
- Ensure map resolution is reasonable (5cm recommended)

## Performance

Typical performance metrics:
- **A* Planning**: 50-500ms (depends on map size and path length)
- **B-Spline Smoothing**: 20-50ms
- **Velocity Optimization**: 10-20ms
- **Total**: Usually < 600ms for most scenarios

## Code Structure

Key functions in `simple_nav_planner.py`:

- `smooth_trajectory()` - Main B-spline smoothing
- `optimize_velocity_profile()` - Velocity optimization
- `is_point_collision_free()` - Collision checking
- `publish_path_visualization()` - RViz visualization

## Future Improvements

Potential enhancements:
- Minimum-snap trajectory optimization
- Dynamic obstacle avoidance
- Multi-resolution path smoothing
- RRT* or other sampling-based planners
- Model Predictive Control (MPC) for path following

## References

- B-spline interpolation: scipy.interpolate.splprep
- Pure pursuit controller: Classical path following algorithm
- A* search: Hart, P. E.; Nilsson, N. J.; Raphael, B. (1968)

## Support

For issues or questions:
1. Check console output for error messages
2. Verify scipy installation: `python3 -c "import scipy; print(scipy.__version__)"`
3. Review RViz topics: `ros2 topic list | grep path`
4. Enable debug logging: Add `--log-level DEBUG` to launch command


