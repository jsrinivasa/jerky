# Mobile ALOHA Navigation Runbook

Step-by-step instructions to go from a PDF floor plan to a robot navigating from point A to point B.

---

## Phase 1: Test with the Existing Map (Simulation, No Hardware)

This uses a map that's already in the workspace. No PDF needed. Good for verifying the software works before introducing your own map.

### Step 1: Source the workspace

```bash
cd /home/aloha/interbotix_ws
source install/setup.bash
```

You need to run this in **every new terminal** you open.

### Step 2: Launch the simulation

```bash
ros2 launch aloha sim_navigation.launch.py
```

This starts:
- **Fake odometry** — simulates the robot moving (no real wheels needed)
- **Map server** — loads the map from `my_robot_maps/maps/my_final_map_moderate.yaml`
- **Navigation planner** — A* path planning + Pure Pursuit path following
- **RViz** — visualization window

### Step 3: Navigate in RViz

1. **Wait** ~5 seconds for everything to start
2. In RViz you should see the map (black lines = walls, white = open space)
3. If the map doesn't appear:
   - Click **Add** (bottom left) → **By topic** → expand `/map` → select **Map** → OK
4. To send the robot to a destination:
   - Click **"2D Goal Pose"** in the top toolbar (green arrow icon)
   - **Click** on the map where you want the robot to go
   - **Drag** to set the direction the robot should face when it arrives
   - Release — you should see a planned path appear and the simulated robot "drive" along it

### What to Expect

- A colored line appears showing the planned path
- The robot marker moves along the path
- Terminal output shows planning/following status
- If the path goes through an obstacle, the planner will route around it

### Troubleshooting

| Problem | Fix |
|---------|-----|
| RViz opens but no map visible | Add `/map` topic manually (Add → By topic → /map → Map) |
| "No map data" in terminal | Map server may not have started yet — wait 10 seconds |
| Goal pose does nothing | Check that the goal is on a white (free) area of the map, not on a wall |
| "Could not find path" | The goal may be unreachable (surrounded by walls) or too close to an obstacle |

### Stop everything

Press `Ctrl+C` in the terminal where you launched it.

---

## Phase 2: Convert Your PDF Floor Plan & Navigate On It

### Step 1: Copy your PDF

Place your Building 16 floor plan PDF somewhere accessible, e.g.:

```bash
cp ~/Downloads/building16_floorplan.pdf /home/aloha/interbotix_ws/src/aloha/maps/
```

### Step 2: Preview the conversion

```bash
cd /home/aloha/interbotix_ws/src/aloha/maps
python3 ../scripts/pdf_to_map.py building16_floorplan.pdf --preview
```

This opens a window showing:
- **Left**: your original floor plan
- **Right**: the occupancy grid (green = free space, red = walls, gray = unknown)

Look at the right side. If walls are not being detected correctly (too much red or not enough), adjust the threshold:

```bash
# Lower threshold = more things become walls (more strict)
python3 ../scripts/pdf_to_map.py building16_floorplan.pdf --preview --wall_threshold 100

# Higher threshold = fewer things become walls (more lenient)
python3 ../scripts/pdf_to_map.py building16_floorplan.pdf --preview --wall_threshold 180
```

Press any key to close the preview.

### Step 3: Figure out the scale (resolution)

The map needs to know how many **real-world meters** each pixel represents. You need to know the length of *something* in the floor plan (a hallway, a room, anything).

**Option A: Interactive measurement (recommended)**

```bash
python3 ../scripts/pdf_to_map.py building16_floorplan.pdf --measure
```

1. A window opens showing the floor plan
2. **Click two endpoints** of something whose real length you know (e.g., a 20-meter hallway)
3. Type the real-world distance in meters when prompted
4. The script calculates the resolution for you

**Option B: Manual calculation**

If you know the building dimensions:
```
resolution = real_world_meters / pixels

Example: Building is 50 meters wide. Image is 1000 pixels wide.
resolution = 50 / 1000 = 0.05 meters per pixel
```

### Step 4: Generate the map files

```bash
python3 ../scripts/pdf_to_map.py building16_floorplan.pdf \
    --resolution 0.05 \
    --output building16 \
    --wall_threshold 128
```

Replace `0.05` with the resolution you calculated. This creates:
- `building16.pgm` — the occupancy grid image
- `building16.yaml` — the metadata file (resolution, origin, thresholds)

### Step 5: Test your map in simulation

```bash
cd /home/aloha/interbotix_ws
source install/setup.bash
ros2 launch aloha sim_navigation.launch.py \
    map_file:=/home/aloha/interbotix_ws/src/aloha/maps/building16.yaml
```

Then use RViz to navigate as described in Phase 1, Step 3.

### Step 6: Iterate

- **Map looks wrong?** Go back to Step 2 and adjust `--wall_threshold`
- **Scale is off?** Go back to Step 3 and re-measure
- **Robot paths too close to walls?** Increase robot radius:
  ```bash
  ros2 launch aloha sim_navigation.launch.py \
      map_file:=/home/aloha/interbotix_ws/src/aloha/maps/building16.yaml \
      robot_radius:=0.4
  ```
- **Robot too slow?** Increase speed:
  ```bash
  ros2 launch aloha sim_navigation.launch.py \
      map_file:=/home/aloha/interbotix_ws/src/aloha/maps/building16.yaml \
      max_linear_velocity:=0.5
  ```

---

## Phase 3: Run on Real Hardware (After Simulation Works)

**Prerequisites:**
- Phase 2 simulation works with your Building 16 map
- ALOHA hardware powered on
- RealSense cameras connected

### Step 1: Launch the robot hardware

```bash
# Terminal 1
cd /home/aloha/interbotix_ws && source install/setup.bash
ros2 launch aloha aloha_bringup.launch.py
```

### Step 2: Launch navigation with AMCL

```bash
# Terminal 2
cd /home/aloha/interbotix_ws && source install/setup.bash
ros2 launch aloha simple_navigation.launch.py \
    map_file:=/home/aloha/interbotix_ws/src/aloha/maps/building16.yaml \
    use_amcl:=true
```

### Step 3: Localize the robot

In RViz:
1. Wait for the particle cloud to appear (scattered dots on the map)
2. If the robot doesn't know where it starts, slowly rotate it — AMCL narrows down position as it sees more walls
3. If you know where the robot starts, use **"2D Pose Estimate"** in RViz to set the initial position

### Step 4: Navigate

Once the particle cloud converges (dots cluster in one area):
1. Click **"2D Goal Pose"** in RViz
2. Click the destination on the map
3. The robot should plan a path and drive there

---

## Quick Reference: All Launch Parameters

```bash
ros2 launch aloha sim_navigation.launch.py \
    map_file:=/path/to/map.yaml \       # Which map to load
    robot_radius:=0.35 \                 # Safety clearance from walls (meters)
    max_linear_velocity:=0.3 \           # Max forward speed (m/s)
    max_angular_velocity:=1.0 \          # Max turning speed (rad/s)
    goal_tolerance:=0.2 \                # How close is "arrived" (meters)
    lookahead_distance:=0.5 \            # How far ahead the controller looks (meters)
    enable_collision_avoidance:=false \   # Real-time obstacle checking
    safety_distance:=0.5 \               # Slow down within this distance (meters)
    emergency_stop_distance:=0.3 \       # Full stop within this distance (meters)
    use_rviz:=true                       # Open RViz visualization
```
