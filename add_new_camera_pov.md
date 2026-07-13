# ALOHA System Changes Summary

## 1. POV Camera Integration (Intel RealSense D435I)

### `config/rs_cam.yaml`
- Added `cam_pov` camera entry with serial number `939622075315`
- Configured for USB 2.x compatibility: `640x480@15fps` color, depth/gyro/accel disabled

### `launch/aloha_bringup.launch.py`
- Added `cam_pov_name` launch argument (default: `cam_pov`)
- Added `use_cam_pov` launch argument (default: `true`) to optionally enable/disable the POV camera
- Added a dedicated `cam_pov_node` (RealSense node) conditionally launched when both `use_cameras` and `use_cam_pov` are true
- Included `cam_pov` in the loginfo output

### `aloha/robot_utils.py` — `ImageRecorder`
- Added `cam_pov` to the camera names list (both mobile and stationary)
- Added `image_cb_cam_pov` callback for the POV camera subscription
- Subscribed cam_pov to `image_raw` topic (not `image_rect_raw`) because the D435I without depth enabled only publishes unrectified images
- Used `TRANSIENT_LOCAL` QoS durability to match the D435I publisher's QoS profile (other cameras use default `VOLATILE`)

### `aloha/constants.py`
- Added `cam_pov` to `camera_names` in all mobile task configs
- Added new task config `aloha_mobile_pick_object` (3000 timesteps, ~60s at 50fps)

### `act_plus_plus/constants.py` (separate repo)
- Added `cam_pov` to `camera_names` for ACT++ training
- Added `aloha_mobile_pick_object` task config

---

## 2. Color Fix (RGB → BGR)

### `aloha/robot_utils.py` — `ImageRecorder.image_cb`
- Changed `desired_encoding` from `'passthrough'` to `'bgr8'` in `image_cb`
- All RealSense cameras publish RGB8; OpenCV's `cv2.imencode` expects BGR
- Without this fix, red and blue channels were swapped in all saved images (e.g., brown table appeared blueish)

---

## 3. Gravity Compensation Graceful Skip

### `aloha/robot_utils.py` — `enable/disable_gravity_compensation`
- Previously, these functions blocked indefinitely when the gravity compensation service was not running
- Now checks service availability via `get_interbotix_global_node().get_service_names_and_types()` before attempting to connect
- Prints a message and skips if the service is unavailable
- This allows launching bringup with `use_gravity_compensation:=false` without deadlocking `record_episodes.py` or `sleep.py`

---

## 4. Null Image Safety

### `aloha/robot_utils.py` — `ImageRecorder.get_images`
- Returns a black `480x640x3` placeholder image if a camera hasn't delivered its first frame yet
- Prevents `cv2.imencode` crash on `None` images during the first few recording timesteps

---

## Recording Workflow

```bash
# Terminal 1: Launch bringup (without gravity compensation)
ros2 launch aloha aloha_bringup.launch.py use_gravity_compensation:=false

# Terminal 2: Record an episode
python3 ~/interbotix_ws/src/aloha/scripts/record_episodes.py --task_name aloha_mobile_pick_object

# Park arms when done
python3 ~/interbotix_ws/src/aloha/scripts/sleep.py --all
```

---

## Files Modified

| File | Changes |
|------|---------|
| `config/rs_cam.yaml` | Added cam_pov D435I config |
| `launch/aloha_bringup.launch.py` | Optional cam_pov node + launch args |
| `aloha/robot_utils.py` | cam_pov subscriber, QoS fix, BGR encoding, gravity comp skip, null image safety |
| `aloha/constants.py` | cam_pov in task configs, new pick_object task |
| `act_plus_plus/constants.py` | cam_pov in ACT++ training config |

---

## How to Add a New Sensor

### Adding a New RealSense Camera

#### Step 1: Find the serial number
```bash
rs-enumerate-devices | grep -A2 "Name\|Serial"
```

#### Step 2: Add to `config/rs_cam.yaml`
```yaml
$(var new_cam_name):
  ros__parameters:
    serial_no: "YOUR_SERIAL_NUMBER"
    rgb_camera:
      profile: '640,480,30'       # adjust fps based on USB bandwidth
    # Disable unused streams to save bandwidth (optional)
    enable_depth: false
    enable_gyro: false
    enable_accel: false
```
> **Note**: Use `15fps` for USB 2.x connections, `30fps` for USB 3.x.

#### Step 3: Add to `launch/aloha_bringup.launch.py`

1. Declare launch arguments:
```python
DeclareLaunchArgument('new_cam_name', default_value='new_cam'),
DeclareLaunchArgument('use_new_cam', default_value='true'),
```

2. Add a conditional node (follow the `cam_pov_node` pattern):
```python
new_cam_node = Node(
    package='realsense2_camera',
    namespace=LaunchConfiguration('new_cam_name'),
    name='camera',
    executable='realsense2_camera_node',
    parameters=[rs_cam_config],
    condition=IfCondition(
        AndSubstitution(
            LaunchConfiguration('use_cameras'),
            LaunchConfiguration('use_new_cam'),
        )
    ),
)
```

3. Add `new_cam_node` to the returned actions list.

#### Step 4: Add to `aloha/robot_utils.py` — `ImageRecorder`

1. Add `'new_cam'` to `self.camera_names` list.
2. Add an `elif` branch to route to the callback:
```python
elif cam_name == 'new_cam':
    callback_func = self.image_cb_new_cam
```
3. Add topic/QoS handling (check which topic your camera publishes):
```python
if cam_name == 'new_cam':
    topic = f'{cam_name}/camera/color/image_raw'  # or image_rect_raw
    qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
```
> **Important**: Check the publisher's QoS with `ros2 topic info /new_cam/camera/color/image_raw --verbose` and match the durability policy.

4. Add the callback method:
```python
def image_cb_new_cam(self, data):
    cam_name = 'new_cam'
    return self.image_cb(cam_name, data)
```

#### Step 5: Add to task configs

In `aloha/constants.py`, add `'new_cam'` to `camera_names` in relevant task configs:
```python
'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'cam_pov', 'new_cam']
```

Also update `act_plus_plus/constants.py` in the training repo.

#### Step 6: Build and test
```bash
cd ~/interbotix_ws && colcon build --symlink-install --packages-select aloha
```

---

### Adding a LiDAR Sensor

#### Step 1: Identify the LiDAR driver package
Most LiDARs have ROS 2 driver packages (e.g., `rplidar_ros`, `velodyne`, `ouster_ros`, `livox_ros_driver2`). Install the appropriate one:
```bash
sudo apt install ros-humble-<lidar-driver-package>
# or build from source in your workspace
```

#### Step 2: Add a launch node in `aloha_bringup.launch.py`
```python
lidar_node = Node(
    package='<lidar_driver_package>',
    executable='<lidar_node_executable>',
    name='lidar',
    namespace='lidar',
    parameters=[{
        'serial_port': '/dev/ttyLIDAR',  # or IP address for ethernet LiDARs
        'frame_id': 'lidar_link',
    }],
    condition=IfCondition(LaunchConfiguration('use_lidar')),
)
```

#### Step 3: Create a udev rule (for USB LiDARs)
```bash
# /etc/udev/rules.d/99-lidar.rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="XXXX", ATTRS{idProduct}=="XXXX", SYMLINK+="ttyLIDAR"
```

#### Step 4: Subscribe in your recording script
LiDAR data (`sensor_msgs/msg/LaserScan` or `sensor_msgs/msg/PointCloud2`) is handled differently from images. You'll need a custom subscriber in `robot_utils.py` or a separate recorder class, since `ImageRecorder` is image-specific.

> **Note**: LiDAR data is not directly used by ACT+ (which is image-based). If you need it for navigation or mapping, subscribe and record it separately.

---

### Adding Any Other Sensor (IMU, Force/Torque, etc.)

The general pattern is:
1. **Driver**: Install or build the ROS 2 driver package
2. **Launch**: Add a `Node()` to `aloha_bringup.launch.py` with a conditional `use_<sensor>` argument
3. **Config**: Add any parameters to a YAML config file under `config/`
4. **Subscribe**: Add a subscriber and callback in `robot_utils.py` (or a new recorder class)
5. **Record**: Update `record_episodes.py` to save the sensor data into the HDF5 file
6. **Build**: `colcon build --symlink-install --packages-select aloha`
7. **Test**: Follow the testing steps below

---

## Testing Steps (Manual)

### 1. Verify hardware detection

**RealSense cameras:**
```bash
rs-enumerate-devices | grep -A2 "Name\|Serial"
```

**USB devices:**
```bash
ls -la /dev/ttyDXL_* /dev/ttyUSB* /dev/ttyLIDAR* 2>/dev/null
```

### 2. Launch bringup and verify nodes
```bash
# Launch
ros2 launch aloha aloha_bringup.launch.py use_gravity_compensation:=false

# In another terminal, check all nodes are running
ros2 node list | grep -E "cam_|lidar|xs_sdk"
```

### 3. Verify topics are published
```bash
# List all camera topics
ros2 topic list | grep "camera/color/image"

# Check publish rate for each camera
ros2 topic hz /cam_high/camera/color/image_rect_raw
ros2 topic hz /cam_left_wrist/camera/color/image_rect_raw
ros2 topic hz /cam_right_wrist/camera/color/image_rect_raw
ros2 topic hz /cam_pov/camera/color/image_raw
```
> Each camera should show a non-zero rate (D405s ~30Hz, D435I ~12-15Hz).

### 4. Check QoS compatibility
```bash
ros2 topic info /cam_pov/camera/color/image_raw --verbose
```
> Verify **Subscription count > 0** when recording is running. If 0, there's a QoS mismatch.

### 5. Check arm communication
```bash
# Verify all 4 arm SDK nodes respond
ros2 topic hz /follower_left/joint_states
ros2 topic hz /follower_right/joint_states
ros2 topic hz /leader_left/joint_states
ros2 topic hz /leader_right/joint_states
```
> Each should show ~100Hz. If 0Hz, check USB connections.

### 6. Record a test episode and verify
```bash
# Record
python3 ~/interbotix_ws/src/aloha/scripts/record_episodes.py --task_name aloha_mobile_pick_object

# Verify HDF5 contents
python3 ~/interbotix_ws/src/aloha/scripts/test_sensor_integration.py
```

---

## Automated Testing

Run the automated test script to verify the full sensor pipeline:

```bash
# With bringup running in another terminal:
python3 ~/interbotix_ws/src/aloha/scripts/test_sensor_integration.py
```

This script checks:
- All expected ROS topics are published
- All cameras are publishing at acceptable frame rates
- All arm joint_states topics are active
- Camera images are non-zero (not black)
- QoS profiles are compatible
- Recorded HDF5 files contain valid data for all cameras

See `scripts/test_sensor_integration.py` for the full implementation.
