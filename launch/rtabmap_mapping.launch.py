#!/usr/bin/env python3

"""
RTAB-Map Mapping Launch File

Builds a 3D map (and live 2D occupancy grid) using the cam_high D435i camera
(swapped in from a D405 on 2026-07-14 -- D405 is a short-range manipulation
camera, wrong tool for room-scale mapping; D435i also brings an onboard IMU,
see use_imu below) PLUS the RPLIDAR S2 (subscribe_scan, folded in via
laser_scan_merger), paired with wheel odometry from the SLATE base.  The
lidar gives the occupancy grid far longer range and much wider coverage than
the camera's depth range alone. Run this alongside aloha_bringup (with
use_cameras:=false so the camera is not double-claimed).

Each mapping session is identified by a map_name.  The RTAB-Map database is
saved to ~/maps/<map_name>.db.  After mapping, save the 2D grid with the
SAME name so everything stays together:

    ros2 run nav2_map_server map_saver_cli -f ~/maps/<map_name> -t /rtabmap/map

You can later merge multiple databases with rtabmap-reprocess to build a
large floor plan.

Workflow:
  # Terminal 1 - start base + joystick (no cameras)
  ros2 launch aloha aloha_bringup.launch.py use_cameras:=false

  # Terminal 2 - start mapping (give your session a name)
  ros2 launch aloha rtabmap_mapping.launch.py map_name:=building16_east

  # Terminal 3 - when mapping is finished, save the 2D occupancy grid
  ros2 run nav2_map_server map_saver_cli -f ~/maps/building16_east -t /rtabmap/map

  # Files produced:
  #   ~/maps/building16_east.db    (RTAB-Map 3D database - for localization & merging)
  #   ~/maps/building16_east.pgm   (2D occupancy grid image)
  #   ~/maps/building16_east.yaml  (2D occupancy grid metadata)

  # Later, for autonomous navigation:
  ros2 launch aloha navigate_mission.launch.py \
      map_name:=building16_east \
      map_file:=~/maps/building16_east.yaml

Tips for reliable mapping:
  - Drive slowly (~0.2 m/s) and smoothly; avoid sharp turns
  - Overlap areas by driving past them from multiple directions
  - Revisit the starting location to close the loop
  - Watch the RTAB-Map node output for loop closure detections
  - If RTAB-Map reports "rejected" loop closures, the map has ambiguity;
    drive more slowly through those areas

Merging multiple maps:
  rtabmap-reprocess --Mem/IncrementalMemory true \
      ~/maps/session_a.db ~/maps/session_b.db ~/maps/merged.db
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import Node

# Updated 2026-07-20 (late session): old D435I (349522070494) was hitting a
# hardware-level "Right MIPI error" on every stream start, reproduced across
# 3 different USB ports/buses -- unit itself was bad, not the cable/port.
# User physically swapped in a new D435I. New serial confirmed via
# `rs-enumerate-devices --compact` (only device connected): 939622075315.
# NOTE: that same serial is also assigned to cam_pov_name in rs_cam.yaml --
# not a conflict for this launch (aloha_bringup always runs use_cameras:=
# false here), but worth resolving if cam_pov and cam_high are ever needed
# at the same time.
CAM_HIGH_SERIAL = '939622075315'

# Second camera added 2026-07-21: mounted upside-down on the back, facing
# backward, for rear coverage + multi-camera SLAM. IMPORTANT: this is the
# SAME physical unit flagged above as hardware-defective on 2026-07-20
# ("Right MIPI error" on every stream start, reproduced across 3 ports).
# Re-tested live 2026-07-21 standalone (color+depth both streamed, no MIPI
# error this time) -- it DID work, but had a flaky startup (several
# "device busy"/"disconnected"/"not found, will try again" cycles before
# settling). Treat as a real risk, not confirmed-fixed -- if this camera
# is unstable in practice, that's a strong, known suspect.
# Also unlike cam_high, this specific unit's RGB sensor does NOT support
# 848x480 at all (confirmed via rs-enumerate-devices -c) -- its color
# ladder is 1920x1080/1280x720/640x480/424x240 only. Using 640x480 (see
# depth/color profile below) as the closest common match.
CAM_LOW_BACK_SERIAL = '349522070494'


def generate_launch_description():
    # Was 'color/image_rect_raw' -- confirmed live 2026-07-14 that this topic
    # has zero publishers (no image_proc/rectify node exists anywhere in this
    # package). The color sensor only ever publishes 'image_raw'; whatever
    # mapping session originally populated building16_with_parking.db must
    # have gone through some other, undocumented path. RTAB-Map handles a
    # raw (unrectified) single RGB stream fine.
    default_rgb = '/cam_high/camera/color/image_raw'
    default_depth = '/cam_high/camera/aligned_depth_to_color/image_raw'
    default_info = '/cam_high/camera/color/camera_info'

    return LaunchDescription([
        DeclareLaunchArgument('rgb_topic', default_value=default_rgb),
        DeclareLaunchArgument('depth_topic', default_value=default_depth),
        DeclareLaunchArgument('camera_info_topic', default_value=default_info),
        DeclareLaunchArgument(
            'rtabmap_viz', default_value='false',
            description='Launch RTAB-Map GUI (requires X11 display).',
        ),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Launch RViz for live map visualization.',
        ),
        DeclareLaunchArgument(
            'localization', default_value='false',
            description='Run in localization mode (requires existing DB). '
                        'If false, runs in mapping mode.',
        ),
        DeclareLaunchArgument(
            'delete_db_on_start', default_value='true',
            description='Delete the RTAB-Map DB on start (mapping mode only). '
                        'Set to false to continue extending an existing DB. '
                        'Has no effect in localization mode.',
        ),
        DeclareLaunchArgument(
            'use_imu', default_value='true',
            description='Use the D435i cam_high onboard IMU (gravity-'
                        'referenced constraints for graph optimization). '
                        'The camera always publishes it regardless; this '
                        'only controls whether RTAB-Map consumes it -- set '
                        'false to roll back to camera-only behavior if it '
                        'misbehaves on a live test.',
        ),
        DeclareLaunchArgument(
            'map_name', default_value='rtabmap',
            description='Session name.  Database is saved to ~/maps/<map_name>.db. '
                        'Use the same name when saving the 2D grid.',
        ),
        # ---- AprilTag landmarks (disabled scaffold -- see node below) ----
        # Why AprilTag and not raw ArUco: RTAB-Map's own landmark support
        # (rtabmap_slam's `tag_detections` topic, remapped from `tag_topic`
        # below) is wired for apriltag_msgs/AprilTagDetectionArray, produced
        # by the `apriltag_ros` package (confirmed against the installed
        # rtabmap_launch source, 2026-07-20) -- not a generic ArUco message.
        # OFF by default: needs (1) `sudo apt install ros-humble-apriltag-ros`
        # (not installed as of 2026-07-20 -- only its message package,
        # apriltag_msgs, is), and (2) physical tags actually printed and
        # placed in the repetitive-office area, with apriltag_size measured
        # to match. Point of this: loop closures/proximity detections were
        # rare all session in a visually-repetitive cubicle office (see
        # nav-localization-independent-safety-net memory) -- a few tags at
        # fixed, known points give cheap, unambiguous relocalization
        # checkpoints no amount of vision/ICP tuning can substitute for.
        DeclareLaunchArgument(
            'use_apriltag_landmarks', default_value='false',
            description='Detect AprilTags in cam_high and feed them to '
                        'RTAB-Map as landmarks. See comment above -- '
                        'needs apriltag_ros installed AND physical tags '
                        'placed before this does anything.',
        ),
        DeclareLaunchArgument(
            'apriltag_family', default_value='36h11',
            description='AprilTag family to detect (36h11 is the common, '
                        'robust default -- pick whatever family the '
                        'printed tags actually use).',
        ),
        DeclareLaunchArgument(
            'apriltag_size', default_value='0.1',
            # TODO(robot): placeholder like the rplidar x/y/z below -- measure
            # the actual printed tag's edge length (meters, black border only)
            # once tags exist.
            description='AprilTag edge length in meters (black square only, '
                        'not the white border). PLACEHOLDER -- measure once '
                        'tags are printed.',
        ),
        DeclareLaunchArgument(
            'tag_topic', default_value='/detections',
            description='Topic the apriltag_node publishes detections on '
                        'and RTAB-Map subscribes to for landmarks. Matches '
                        'rtabmap_launch\'s own default -- override only if '
                        'you need a different namespace.',
        ),
        # ---- cam_low_back (rear D435i, upside-down, multi-camera SLAM) ----
        DeclareLaunchArgument(
            'use_cam_low_back', default_value='true',
            description='Launch the rear D435i (cam_low_back) and feed it '
                        'into RTAB-Map as a second RGBD camera (via '
                        'rgbdx_sync -- see the rgbd_cameras=0 comment near '
                        'the main rtabmap node), plus its own depth-derived '
                        'scan folded into /scan for rear obstacle coverage. '
                        'Set false to fall back to cam_high-only, e.g. if '
                        'this camera turns out to still be the flaky unit '
                        'flagged near CAM_LOW_BACK_SERIAL above.',
        ),
        DeclareLaunchArgument(
            'cam_back_x', default_value='-0.0838',
            description='base_link -> cam_low_back_link x offset (m): '
                        "measured 2026-07-21 as 6in forward of the robot's "
                        'back edge. Back edge is 9.3in behind base_link '
                        "(see navigate_mission.launch.py's robot_radius "
                        'derivation), so x = -9.3in + 6in = -3.3in = '
                        '-0.0838m. Direct user measurement of the offset '
                        'itself was not given -- this is derived from two '
                        'separate measurements, so it is the single most '
                        'likely number here to need correcting.',
        ),
        DeclareLaunchArgument(
            'cam_back_y', default_value='0.0',
            description='base_link -> cam_low_back_link y offset (m): '
                        'assumed laterally centered (not stated otherwise).',
        ),
        DeclareLaunchArgument(
            'cam_back_z', default_value='0.4953',
            description='base_link -> cam_low_back_link z offset (m): '
                        'measured 2026-07-21 as 19.5in off the ground.',
        ),
        DeclareLaunchArgument(
            'cam_back_yaw', default_value='3.14159265',
            description='Camera faces backward (opposite cam_high) -- '
                        '180deg yaw from base_link forward (+x).',
        ),
        DeclareLaunchArgument(
            'cam_back_roll', default_value='3.14159265',
            description='Camera is mounted upside-down -- 180deg roll. '
                        'This is the standard interpretation of "upside '
                        'down, facing backward"; UNVERIFIED beyond that. '
                        'NOTE: do not "verify" this by looking at the raw '
                        '/cam_low_back image -- it will always look '
                        'upside-down regardless of whether this TF is '
                        'right, since the driver publishes the sensor\'s '
                        'raw pixels unrotated and this TF only describes '
                        'the camera\'s pose for 3D geometry, it never '
                        'touches the 2D image (confirmed 2026-07-21: the '
                        'raw feed looked upside-down exactly as expected '
                        'with this TF already applied). The real check is '
                        'whether the resulting point cloud/map geometry '
                        'makes physical sense (floor at z~0, walls '
                        'vertical) once mapping with this camera.',
        ),
        DeclareLaunchArgument(
            'use_ekf_odom', default_value='true',
            description='Fuse /mobile_base/odom + the IMU gyro yaw-rate '
                        '(see config/ekf.yaml) into /odometry/filtered via '
                        'robot_localization, to cut wheel-slip yaw drift '
                        '(worst during turns). Only useful with use_imu:=true. '
                        'Does NOT publish TF (publish_tf:=false) -- topic '
                        'only, so it cannot conflict with the base driver\'s '
                        'own odom->base_footprint broadcast. Set false to '
                        'roll back to raw /mobile_base/odom if it misbehaves.',
        ),

        # ---- RPLIDAR S2 (front, near-bottom mount, rotated 90deg CW) -----
        # x corrected 2026-07-20 per user estimate: ~5-6in forward (was
        # 0.25m/~9.8in, a placeholder that was never actually measured and
        # turned out to be off by ~4in). Used the 5.5in midpoint (0.1397m) --
        # user gave a range, not a single tape-measured value, so this is an
        # estimate, not a precise measurement; refine further if exact
        # obstacle-clearance margins ever matter more than they do now.
        # y/z were already confirmed correct. yaw is a known fact, not a
        # placeholder: the sensor is physically mounted rotated 90deg, and
        # rotating its raw feed 90deg CCW recovers true robot-forward --
        # i.e. base_angle = sensor_angle + 90deg, so this TF's yaw = +pi/2.
        DeclareLaunchArgument(
            'use_rplidar', default_value='true',
            description='Launch the RPLIDAR S2 driver + TF and fold it into '
                        '/scan (both for live obstacle avoidance and RTAB-Map '
                        'subscribe_scan). Set false to fall back to cam_high '
                        'depth only.',
        ),
        DeclareLaunchArgument(
            'rplidar_port', default_value='/dev/rplidar',
            description='Serial device for the RPLIDAR S2. Defaults to the '
                        'stable udev symlink (see config/99-rplidar.rules); '
                        'override with the raw /dev/ttyUSBx path until that '
                        'rule is installed.',
        ),
        DeclareLaunchArgument(
            'rplidar_x', default_value='0.1397',
            description='base_link -> rplidar_link x offset (m): forward/back '
                        'distance from base_link origin to the lidar. '
                        '~5.5in (0.1397m), midpoint of user\'s 2026-07-20 '
                        '5-6in estimate -- not a precise tape measurement.',
        ),
        DeclareLaunchArgument(
            'rplidar_y', default_value='0.0',
            description='base_link -> rplidar_link y offset (m): left/right '
                        'offset. Confirmed 0 -- lidar is laterally centered.',
        ),
        DeclareLaunchArgument(
            'rplidar_z', default_value='0.27',
            description='base_link -> rplidar_link z offset (m): height off '
                        'the ground. Measured on robot: 0.27m.',
        ),
        DeclareLaunchArgument(
            'rplidar_yaw', default_value='1.5708',
            description='Yaw (rad) of the lidar zero-mark relative to '
                        "base_link's forward (+x) axis. Known from the "
                        'physical mount (rotated 90deg; +pi/2 = rotate the '
                        'raw feed 90deg CCW to get true robot-forward) -- '
                        'NOT a placeholder, but re-verify if the mount changes.',
        ),

        ExecuteProcess(cmd=['mkdir', '-p', os.path.expanduser('~/maps')]),

        LogInfo(msg=[
            'RTAB-Map database: ~/maps/',
            LaunchConfiguration('map_name'),
            '.db  —  Save 2D grid with:  ros2 run nav2_map_server map_saver_cli '
            '-f ~/maps/', LaunchConfiguration('map_name'), ' -t /rtabmap/map',
        ]),

        # -- base_footprint -> base_link static TF (OFF by default) --------
        # CORRECTION 2026-07-14: on this robot the SLATE driver publishes
        # odom -> base_link DIRECTLY (confirmed live: /mobile_base/odom has
        # child_frame_id=base_link, and `tf2_echo odom base_link` resolves).
        # The old assumption here -- that the base publishes odom ->
        # base_footprint and this static completes base_footprint ->
        # base_link -- is WRONG for this configuration. Publishing it anyway
        # gives base_link a SECOND parent (odom AND base_footprint), a TF
        # conflict that intermittently breaks RTAB-Map's odom->base_link
        # lookup. Nothing consumes base_footprint (camera_link/rplidar_link
        # and rtabmap's frame_id all hang off base_link), so it's gated off.
        # Re-enable only if a base config that emits odom->base_footprint is
        # ever used: `... publish_base_footprint_link:=true`.
        DeclareLaunchArgument(
            'publish_base_footprint_link', default_value='false',
            description='Publish a base_footprint -> base_link static TF. '
                        'Leave false when the base driver already publishes '
                        'odom -> base_link directly (the case on this robot); '
                        'enabling it double-parents base_link and breaks TF.',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_footprint_to_base_link',
            output='screen',
            arguments=[
                '0', '0', '0.1', '0', '0', '0',
                'base_footprint', 'base_link',
            ],
            condition=IfCondition(LaunchConfiguration('publish_base_footprint_link')),
        ),

        # -- cam_high RealSense (mapping-optimized settings) ---------------
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='cam_high',
            name='camera',
            output='screen',
            parameters=[{
                'serial_no': CAM_HIGH_SERIAL,
                # initial_reset was True -- it does a USB HARDWARE reset on
                # startup which on 2026-07-14 re-enumerated the ttyUSB bus and
                # knocked the SLATE base's serial (/dev/ttyUSB0) offline,
                # freezing wheel odometry mid-map. Disabled so bringing up the
                # camera can't kill the base. If the camera ever fails to
                # enumerate cleanly, power-cycle it by hand rather than
                # re-enabling this. (Deeper issue: the base serial shares a USB
                # controller with the cameras -- worth re-cabling onto its own.)
                'initial_reset': False,
                'enable_color': True,
                'enable_depth': True,
                'enable_infra': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'align_depth.enable': True,
                # BUG FOUND 2026-07-20: 'depth_module.profile' /
                # 'rgb_camera.profile' are NOT real parameter names for this
                # installed realsense2_camera version -- confirmed via
                # `ros2 param list /cam_high/camera | grep profile`, which
                # only shows 'depth_module.depth_profile' and
                # 'rgb_camera.color_profile'. Since the old names were never
                # declared by the node, every value ever set here (this
                # session's 848x480 attempt, its 640x480 revert, AND
                # whatever the original pre-existing value was) was silently
                # ignored -- the camera has been running at its own internal
                # defaults (confirmed live: color 1280x720x30, depth
                # 848x480x30) this whole time, for the ENTIRE project
                # history, not just tonight. Fixed to the correct names.
                # This is a strong suspect for why building16_corrected.db
                # (and likely every prior map) has ZERO real loop-closure
                # links despite 675 nodes (confirmed via direct sqlite query
                # on the Link table -- only type=0 neighbor/odometry links
                # and type=9 gravity constraints exist) -- inconsistent/
                # uncontrolled resolution-intrinsics between frames would
                # explain geometric verification failing structurally, not
                # just as an environment-difficulty issue.
                # Dropped further to 424x240x15 on 2026-07-20 (same night,
                # later): 640x480x15 was still hitting recurring USB -71
                # (uvcvideo protocol errors) on this laptop's port, silently
                # killing the video streams (RealSense stayed up, IMU kept
                # flowing, but color/depth frames never arrived) -- happened
                # on two different physical USB ports, so it's a bandwidth/
                # controller limit, not a loose cable. Lower resolution cuts
                # required USB throughput while keeping the same 15fps.
                # Bumped back to 848x480x15 on 2026-07-21: root-caused the
                # session's "always exactly 0 inliers" mystery to a 3x
                # resolution mismatch, not a registration bug -- every map in
                # this project (checked via databaseViewer's Calib field) was
                # recorded at 1280x720 (fx~911), while live localization has
                # been running at 424x240 (fx~307) since the drop above.
                # Feature positions extracted from a 3x-coarser live image
                # don't have the precision to survive geometric verification
                # against a map recorded at full resolution, regardless of
                # registration strategy/estimation type/descriptor -- all
                # tested and ruled out tonight. Standalone stability test
                # confirmed 848x480 streams cleanly today (~10Hz, zero USB
                # errors) -- the 07-20 instability may have been port/cable
                # state that's since changed, not a hard laptop limit.
                # Dropped from 848x480 to 640x480 on 2026-07-21, same day:
                # RTAB-Map's own multi-camera RGBDX handling hard-requires
                # every camera in the array to share identical image
                # dimensions (confirmed via a live FATAL crash --
                # MsgConversion.cpp:2114::convertRGBDMsgs(), "imageWidth=848
                # vs 640" -- the moment cam_low_back was added). cam_low_back
                # is a different physical D435i unit whose RGB sensor
                # doesn't support 848x480 at all (confirmed via
                # rs-enumerate-devices -c) -- 640x480 is the highest
                # resolution both units share, so cam_high has to match it.
                # Real tradeoff: this gives up some of the precision gained
                # by the 848x480 bump earlier today (still much better than
                # the original 424x240 -- see the multi-paragraph history
                # above -- just not as good as 848x480 was). If
                # use_cam_low_back:=false, there's currently no mechanism to
                # bump cam_high back to 848x480 automatically; do it by hand
                # here if running cam_high-only for an extended period.
                'depth_module.depth_profile': '640,480,15',
                'rgb_camera.color_profile': '640,480,15',
                'rgb_camera.enable_auto_exposure': True,
                'depth_module.enable_auto_exposure': True,
                # Post-processing to cut depth noise before it ever reaches
                # RTAB-Map: spatial = edge-preserving smoothing + hole fill,
                # temporal = averages across recent frames (effective here
                # since the scene is roughly static frame-to-frame while
                # mapping). Decimation deliberately left off -- it downsamples
                # depth resolution too, trading detail for noise/speed.
                'spatial_filter.enable': True,
                'temporal_filter.enable': True,
                # D435i onboard IMU -- always published (cheap); whether
                # RTAB-Map actually consumes it is gated by use_imu below.
                # linear_interpolation (2) time-aligns gyro+accel samples
                # against each other into one combined topic, which matters
                # for it to be useful as a graph-optimization constraint.
                'enable_gyro': True,
                'enable_accel': True,
                'unite_imu_method': 2,
                # NOTE: leaving IMU QoS at the realsense default (best-effort).
                # Verified offline 2026-07-16 that imu_filter_madgwick's input
                # subscription accepts a best-effort publisher fine (1001/1001
                # msgs through, oriented output), so no per-stream QoS override
                # is needed here. (An earlier attempt set gyro_qos/accel_qos to
                # 'DEFAULT' on a wrong assumption -- reverted.)
            }],
        ),

        # -- base_link -> camera_link static TF ----------------------------
        # Position of cam_high on the robot.
        # x updated 2026-07-16 per user measurement: camera is mounted 8in
        # forward of base_link's origin. (Previous 0.2098050m/~8.26in was a
        # carried-over D405 housing estimate, flagged as unverified for the
        # D435i swapped in 2026-07-14 -- superseded by this measurement.)
        # z corrected 2026-07-20 per user measurement: 45in up (1.143m), not
        # the previous 1.031778m (~40.6in) -- that was off by ~4.4in.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_tf',
            output='screen',
            arguments=[
                '--x', '0.2032',
                '--y', '0',
                '--z', '1.143',
                '--yaw', '0',
                '--pitch', '0',
                '--roll', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'camera_link',
            ],
        ),

        # -- cam_low_back RealSense (rear, upside-down) ---------------------
        # See CAM_LOW_BACK_SERIAL comment above for the hardware-history
        # caveat and the resolution-ladder difference from cam_high.
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='cam_low_back',
            name='camera',
            output='screen',
            parameters=[{
                'serial_no': CAM_LOW_BACK_SERIAL,
                # 2026-07-22: camera_name defaults to 'camera' for EVERY
                # realsense2_camera_node regardless of ROS namespace -- the
                # ROS namespace ('cam_low_back') only affects topic names,
                # not the frame_id strings embedded in message headers or
                # published via TF. With both cam_high and cam_low_back left
                # at the 'camera' default, BOTH published identical frame_ids
                # (confirmed live: both /cam_high and /cam_low_back color+
                # depth images had frame_id='camera_color_optical_frame',
                # and TF only had ONE camera_color_optical_frame node in the
                # whole tree, hanging off cam_high's camera_link). Every
                # rear-camera point ended up registered through CAM_HIGH's
                # forward-facing, right-side-up TF chain instead of the
                # camera_low_back_link chain below -- this, not a wrong
                # rotation value, is why the two clouds came out facing the
                # same direction with no z-flip (confirmed by the user
                # clicking through individual frames in rtabmap-
                # databaseViewer). Setting a unique camera_name gives this
                # camera its own frame_id namespace (cam_low_back_link,
                # cam_low_back_color_optical_frame, etc.) so its data
                # actually resolves through base_to_camera_low_back_tf below
                # instead of colliding with cam_high's.
                'camera_name': 'cam_low_back',
                'initial_reset': False,  # see cam_high's own comment on this
                'enable_color': True,
                'enable_depth': True,
                'enable_infra': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'align_depth.enable': True,
                # This unit's RGB sensor doesn't support 848x480 at all
                # (confirmed via rs-enumerate-devices -c for serial
                # 349522070494) -- 640x480 is the highest resolution both
                # its color and depth sensors share at a reasonable fps.
                'depth_module.depth_profile': '640,480,15',
                'rgb_camera.color_profile': '640,480,15',
                'rgb_camera.enable_auto_exposure': True,
                'depth_module.enable_auto_exposure': True,
                'spatial_filter.enable': True,
                'temporal_filter.enable': True,
                # Not enabling gyro/accel here -- rtabmap only takes ONE
                # imu_topic overall (still cam_high's, below), and since
                # both cameras are rigidly mounted to the same rigid body,
                # cam_high's gravity reference already describes this
                # camera's orientation too. No need for a second IMU.
                # TEMP 2026-07-21: pointcloud.enable for a manual extrinsics
                # sanity check (floor near z~0, walls vertical) in RViz --
                # remove once that check is done, not needed for normal
                # mapping (rtabmap builds its own cloud from rgb+depth).
                'pointcloud.enable': True,
            }],
            condition=IfCondition(LaunchConfiguration('use_cam_low_back')),
        ),

        # -- base_link -> cam_low_back_link static TF --------------------
        # See cam_back_x/y/z/yaw/roll args above for the measurement this is
        # built from, and the "UNVERIFIED" flag on the rotation specifically.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_low_back_tf',
            output='screen',
            arguments=[
                '--x', LaunchConfiguration('cam_back_x'),
                '--y', LaunchConfiguration('cam_back_y'),
                '--z', LaunchConfiguration('cam_back_z'),
                '--yaw', LaunchConfiguration('cam_back_yaw'),
                '--pitch', '0',
                '--roll', LaunchConfiguration('cam_back_roll'),
                '--frame-id', 'base_link',
                # Renamed from 'camera_low_back_link' 2026-07-22 to match
                # the new camera_name='cam_low_back' prefix on the realsense
                # node's own internal frame chain (see the big comment on
                # that node's parameters) -- 'camera_low_back_link' would
                # now be an orphan frame with nothing attached under it.
                '--child-frame-id', 'cam_low_back_link',
            ],
            condition=IfCondition(LaunchConfiguration('use_cam_low_back')),
        ),

        # -- Depth to LaserScan (cam_low_back) -------------------------------
        # Rear coverage for the merged /scan -- mirrors the cam_high
        # instance below. scan_height=300 is valid here too: this camera's
        # depth profile is also 480px tall (640x480), same as cam_high's
        # 848x480 -- see that node's own scan_height comment for why this
        # number has to track the live depth profile's height.
        Node(
            package='depthimage_to_laserscan',
            executable='depthimage_to_laserscan_node',
            name='depthimage_to_laserscan_back',
            output='screen',
            parameters=[{
                'scan_height': 300,
                'scan_row_step': 1,
                'scan_time': 0.033,
                'range_min': 0.1,
                'range_max': 5.0,
                'output_frame': 'cam_low_back_link',  # see camera_name rename comment above
            }],
            remappings=[
                ('depth', '/cam_low_back/camera/depth/image_rect_raw'),
                ('depth_camera_info', '/cam_low_back/camera/depth/camera_info'),
                ('scan', '/scan_depth_cam_low_back'),
            ],
            condition=IfCondition(LaunchConfiguration('use_cam_low_back')),
        ),

        # -- IMU orientation filter (Madgwick) -----------------------------
        # THE actual "IMU fusion". The D435i publishes only raw gyro+accel
        # (no orientation quaternion). RTAB-Map needs an ORIENTED IMU for its
        # gravity constraint -- fed the raw topic directly it logs "IMU
        # received doesn't have orientation set, it is ignored" for every
        # sample and, with wait_imu_to_init, never initializes (confirmed
        # 2026-07-14). Madgwick fuses raw gyro+accel -> an oriented
        # sensor_msgs/Imu on /cam_high/camera/imu/filtered, which rtabmap then
        # consumes (imu_topic below). use_mag=false: the D435i has no
        # magnetometer. publish_tf=false: don't let it inject its own TF.
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter',
            output='screen',
            parameters=[{
                'use_mag': False,
                'world_frame': 'enu',
                'publish_tf': False,
            }],
            remappings=[
                ('imu/data_raw', '/cam_high/camera/imu'),
                ('imu/data', '/cam_high/camera/imu/filtered'),
            ],
            condition=IfCondition(LaunchConfiguration('use_imu')),
        ),

        # -- Wheel odom + IMU gyro fusion (robot_localization) --------------
        # See config/ekf.yaml for the full rationale. Short version: wheel-
        # slip yaw error is worst during turns; the gyro doesn't slip. Topic
        # only (publish_tf:=false) -- consumers that want the fused estimate
        # (currently simple_nav_planner, via navigate_mission.launch.py's
        # odom remap) subscribe to /odometry/filtered instead of raw
        # /mobile_base/odom. RTAB-Map's own odom_frame_id TF lookup is
        # untouched by this -- it still reads odom->base_link via TF from
        # the base driver directly, not from this topic.
        # requires: sudo apt install ros-humble-robot-localization (not
        # installed as of 2026-07-20).
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                os.path.join(
                    get_package_share_directory('aloha'), 'config', 'ekf.yaml'
                ),
            ],
            remappings=[
                ('odometry/filtered', '/odometry/filtered'),
            ],
            condition=IfCondition(LaunchConfiguration('use_ekf_odom')),
        ),

        # -- AprilTag landmark detector (disabled by default) ---------------
        # See use_apriltag_landmarks arg above for the full rationale and
        # prerequisites. image_rect/camera_info below are cam_high's raw
        # color stream -- NOT verified to be sufficiently rectified for
        # apriltag_node's expected input; may need an image_proc rectify
        # node in front of this once tags are actually in place and this
        # gets tested for real.
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_detector',
            output='screen',
            parameters=[{
                'family': LaunchConfiguration('apriltag_family'),
                'size': LaunchConfiguration('apriltag_size'),
            }],
            remappings=[
                ('image_rect', '/cam_high/camera/color/image_raw'),
                ('camera_info', '/cam_high/camera/color/camera_info'),
                ('detections', LaunchConfiguration('tag_topic')),
            ],
            condition=IfCondition(LaunchConfiguration('use_apriltag_landmarks')),
        ),

        # -- Depth to LaserScan (cam_high) ----------------------------------
        # Narrow-FOV supplement to the RPLIDAR below, and the sole live scan
        # source if use_rplidar:=false. Own topic; laser_scan_merger folds
        # it (and the lidar) into /scan.
        Node(
            package='depthimage_to_laserscan',
            executable='depthimage_to_laserscan_node',
            name='depthimage_to_laserscan',
            output='screen',
            parameters=[{
                # Must be <= depth image height (480 for the current
                # 640,480,15 depth_module.depth_profile above -- was
                # 848,480,15, dropped to match cam_low_back's resolution
                # ceiling, see that param's own comment; height stayed 480
                # either way so this doesn't need to change). Briefly 150
                # (scaled for a 424x240 profile, since 300 silently broke
                # depthimage_to_laserscan for the whole 2026-07-21 session's
                # 424x240 window -- see git history if that value is needed
                # again). Back to the original 300 now that height is 480
                # again.
                'scan_height': 300,
                'scan_row_step': 1,
                'scan_time': 0.033,
                'range_min': 0.1,
                # Was 3.0, tuned to the old D405's short reliable range.
                # D435i is usable out to ~6m; 5.0 leaves margin before its
                # far-range noise gets bad, still supplementary to the lidar.
                'range_max': 5.0,
                'output_frame': 'camera_link',
            }],
            remappings=[
                ('depth', '/cam_high/camera/depth/image_rect_raw'),
                ('depth_camera_info', '/cam_high/camera/depth/camera_info'),
                ('scan', '/scan_depth_cam_high'),
            ],
        ),

        # -- RPLIDAR S2 (360 deg live obstacle sensing) ---------------------
        # Primary live scan source: far longer range and full 360 deg
        # coverage vs. cam_high's narrow forward cone. Feeds both live
        # obstacle avoidance (simple_nav_planner) and RTAB-Map's own
        # occupancy grid (subscribe_scan below).
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_s2',
            output='screen',
            parameters=[{
                'channel_type': 'serial',
                'serial_port': LaunchConfiguration('rplidar_port'),
                'serial_baudrate': 1000000,  # S2 requires 1Mbps (A-series use 115200)
                'frame_id': 'rplidar_link',
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': 'DenseBoost',
            }],
            remappings=[('scan', '/scan_rplidar')],
            condition=IfCondition(LaunchConfiguration('use_rplidar')),
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_rplidar_tf',
            output='screen',
            arguments=[
                '--x', LaunchConfiguration('rplidar_x'),
                '--y', LaunchConfiguration('rplidar_y'),
                '--z', LaunchConfiguration('rplidar_z'),
                '--yaw', LaunchConfiguration('rplidar_yaw'),
                '--pitch', '0', '--roll', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'rplidar_link',
            ],
            condition=IfCondition(LaunchConfiguration('use_rplidar')),
        ),

        # -- Laser Scan Merger -----------------------------------------------
        # Combines lidar + depth-camera scans into one /scan, nearest-range-
        # wins per angular bin. Always runs: with use_rplidar:=false it
        # simply forwards cam_high's scan through unchanged.
        Node(
            package='aloha',
            executable='laser_scan_merger',
            name='laser_scan_merger',
            output='screen',
            parameters=[{
                'target_frame': 'base_link',
                'input_topics': [
                    '/scan_rplidar', '/scan_depth_cam_high',
                    '/scan_depth_cam_low_back',
                ],
                'publish_rate': 15.0,
                # self_occlusion_ranges intentionally not set here: an empty
                # list can't be serialized as a launch-time parameter override
                # (ambiguous array type). Node declares it with dynamic_typing
                # so its own [] default applies. Fill in once a sector reads a
                # constant near-range hit while nothing is actually there
                # (likely the chassis): e.g. [2.6, 3.14, -3.14, -2.6] for a
                # rear wedge -- set via `ros2 param set` or add it back here
                # as a non-empty list.
            }],
        ),

        # -- RTAB-Map pipeline (wheel odom, no visual odometry) ------------
        # 2026-07-21: REWRITTEN from a single IncludeLaunchDescription(
        # rtabmap_launch's rtabmap.launch.py) to explicit nodes, to add
        # cam_low_back as a second RGBD camera. Root cause: that wrapper has
        # NO multi-camera argument at all (no rgbd_cameras passthrough), and
        # its numbered rgbd_image0/rgbd_image1 multi-camera path additionally
        # requires rtabmap_ros to have been BUILT with -DRTABMAP_SYNC_MULTI_
        # RGBD=ON (confirmed OFF by default in rtabmap_sync's CMakeLists.txt,
        # and apt-installed builds use build defaults) -- using it as-is
        # would FATAL error at runtime with 2+ cameras. The unconditionally-
        # compiled alternative (confirmed via CommonDataSubscriber.cpp: not
        # inside any #ifdef) is the RGBDX/"rgbd_images" array path:
        # rgbd_cameras=0 + a rgbdx_sync node combining N per-camera rgbd_sync
        # outputs into one rgbd_images topic. This mirrors RTAB-Map's own
        # official dual-camera example (rtabmap_examples/launch/
        # rtabmap_D405x2.launch.py) adapted to that unconditional path. Every
        # parameter/remapping below is a direct port of what rtabmap_launch's
        # wrapper was already passing this node (preserving current, tuned
        # behavior) plus the multi-camera-specific additions.
        # NOT ported: the wrapper's own `rviz`/point_cloud_xyzrgb nodes --
        # dead weight, this project always launches its own separate RViz
        # config (rtabmap_localization.rviz) from navigate_mission.launch.py
        # instead, `rviz:=true` here was never actually exercised.

        # rgbd_sync (front, cam_high) -- same role the wrapper's own
        # internal rgbd_sync node played, just given an explicit unique
        # output topic instead of the wrapper's default 'rgbd_image'.
        Node(
            package='rtabmap_sync', executable='rgbd_sync',
            name='rgbd_sync_front', output='screen',
            parameters=[{
                'approx_sync': True,
                'approx_sync_max_interval': 0.05,
                'topic_queue_size': 10,
                'sync_queue_size': 10,
                'qos': 1,
                'qos_camera_info': 1,
                'depth_scale': 1.0,
            }],
            remappings=[
                ('rgb/image', LaunchConfiguration('rgb_topic')),
                ('depth/image', LaunchConfiguration('depth_topic')),
                ('rgb/camera_info', LaunchConfiguration('camera_info_topic')),
                # use_cam_low_back:=true -> feeds rgbdx_sync below via the
                # front-specific topic name. false -> the main rtabmap node
                # needs to read this directly as its single-camera default
                # topic ('rgbd_image', unnamed) instead -- see the
                # rgbd_cameras comment on the main node for the other half
                # of this toggle. Previously hardcoded to rgbd_image_front
                # unconditionally, which left single-camera mode subscribing
                # to a topic nothing published -- confirmed via this file's
                # own prior comment flagging it as "not wired up".
                ('rgbd_image', PythonExpression([
                    "'/rtabmap/rgbd_image_front' if '",
                    LaunchConfiguration('use_cam_low_back'),
                    "' == 'true' else '/rtabmap/rgbd_image'",
                ])),
            ],
        ),

        # rgbd_sync (back, cam_low_back) -- new.
        Node(
            package='rtabmap_sync', executable='rgbd_sync',
            name='rgbd_sync_back', output='screen',
            parameters=[{
                'approx_sync': True,
                'approx_sync_max_interval': 0.05,
                'topic_queue_size': 10,
                'sync_queue_size': 10,
                'qos': 1,
                'qos_camera_info': 1,
                'depth_scale': 1.0,
            }],
            remappings=[
                ('rgb/image', '/cam_low_back/camera/color/image_raw'),
                ('depth/image', '/cam_low_back/camera/aligned_depth_to_color/image_raw'),
                ('rgb/camera_info', '/cam_low_back/camera/color/camera_info'),
                ('rgbd_image', '/rtabmap/rgbd_image_back'),
            ],
            condition=IfCondition(LaunchConfiguration('use_cam_low_back')),
        ),

        # rgbdx_sync -- combines the two per-camera rgbd_image topics above
        # into one rgbd_images (plural) array message. See rgbd_cameras=0
        # comment on the main rtabmap node below for why this path (not
        # rgbd_cameras=2 + numbered topics) is what actually works here.
        Node(
            package='rtabmap_sync', executable='rgbdx_sync',
            name='rgbdx_sync', output='screen',
            parameters=[{
                'approx_sync': True,
                'approx_sync_max_interval': 0.05,
                'topic_queue_size': 10,
                'sync_queue_size': 10,
                'qos': 1,
                'rgbd_cameras': 2,
            }],
            remappings=[
                ('rgbd_image0', '/rtabmap/rgbd_image_front'),
                ('rgbd_image1', '/rtabmap/rgbd_image_back'),
                ('rgbd_images', '/rtabmap/rgbd_images'),
            ],
            condition=IfCondition(LaunchConfiguration('use_cam_low_back')),
        ),

        # Main RTAB-Map SLAM node. namespace='rtabmap' matches what the
        # previous IncludeLaunchDescription used by default -- preserves
        # every existing external reference (RViz's /rtabmap/initialpose,
        # demo_ops.sh's -t /rtabmap/map, etc.) without needing to touch them.
        #
        # 2026-07-22: wrapped in a 10s TimerAction after the user reported
        # the map "not stitching properly" and independent analysis of
        # building16_multicam.db found 26/381 nodes (~7%) with a literal
        # identity pose -- RTAB-Map's own placeholder for "no odometry
        # available" (confirmed via CoreWrapper.cpp's own log line,
        # "Odometry is reset (identity pose detected). Increment map id!",
        # found in that session's mapping.log). ALL 16 occurrences of that
        # exact warning fired in a single ~3s burst in the first few seconds
        # after this node started -- immediately racing the EKF (ekf_node),
        # the Madgwick IMU filter, and both camera drivers, all of which
        # ALSO start at t=0 and need a moment to produce their first real,
        # stable output. rtabmap's own wait_for_transform=5.0 only waits for
        # the odom->base_link TF to exist at all, not for the upstream
        # filters feeding it to have converged past their zero-initialized
        # startup state -- so it can sail straight through a transform that
        # technically "exists" but is still the EKF's near-origin initial
        # guess. Once actually delayed 10s past when the base/EKF/camera
        # chain starts, rtabmap only ever subscribes to fresh, already-
        # flowing topics (Volatile QoS -- no stale backlog to catch up on),
        # so it should never see that startup transient at all. The
        # remaining, scattered identity-pose nodes seen up to ~80s into
        # that same recording may be a related but distinct effect (e.g.
        # rtabmap catching up a startup backlog of already-queued frames);
        # if identity-pose nodes still appear after this fix, that's the
        # next thing to dig into.
        TimerAction(period=10.0, actions=[Node(
            package='rtabmap_slam', executable='rtabmap',
            name='rtabmap', namespace='rtabmap', output='screen',
            parameters=[{
                'subscribe_depth': False,
                'subscribe_rgbd': True,
                'subscribe_rgb': False,
                'subscribe_stereo': False,
                'subscribe_scan': True,
                'subscribe_scan_cloud': False,
                'subscribe_user_data': False,
                'subscribe_odom_info': False,
                'frame_id': 'base_link',
                'map_frame_id': 'map',
                'odom_frame_id': 'odom',
                'publish_tf': True,
                'use_action_for_goal': False,
                'odom_tf_angular_variance': 0.01,
                'odom_tf_linear_variance': 0.001,
                'odom_sensor_sync': False,
                'wait_for_transform': 5.0,
                'database_path': PythonExpression([
                    "str(__import__('pathlib').Path.home() / 'maps' / ('",
                    LaunchConfiguration('map_name'),
                    "' + '.db'))",
                ]),
                'approx_sync': True,
                'topic_queue_size': 10,
                'sync_queue_size': 10,
                'qos_image': 1,
                'qos_scan': 1,
                'qos_odom': 1,
                'qos_camera_info': 1,
                # realsense2_camera publishes IMU as best-effort; qos_image=1
                # (reliable) above would otherwise make rtabmap silently drop
                # every IMU message (confirmed live 2026-07-14).
                'qos_imu': 2,
                'qos_gps': 1,
                'qos_env_sensor': 1,
                'qos_user_data': 1,
                'scan_normal_k': 0,
                'landmark_linear_variance': 0.0001,
                'landmark_angular_variance': 9999.0,
                # rgbd_cameras=0 selects the RGBDX ("rgbd_images", plural,
                # array-of-N) subscription path instead of a single
                # "rgbd_image" topic -- see the big comment above this
                # section for why this specific path (not rgbd_cameras=2)
                # is what actually works on an apt-installed rtabmap_sync.
                # 2026-07-22: this used to be hardcoded to 0 even with
                # use_cam_low_back:=false, which left rtabmap waiting on
                # rgbd_images from a rgbdx_sync that was never launched --
                # confirmed dead (this exact gap was flagged, unfixed, in
                # this same comment previously). Now conditional: 0 (RGBDX
                # array) when the rear camera is in use, matching
                # rgbd_sync_front's own conditional remap above; -1
                # (single-camera default, subscribes the plain 'rgbd_image'
                # topic) when it's cam_high-only.
                'rgbd_cameras': PythonExpression([
                    "0 if '",
                    LaunchConfiguration('use_cam_low_back'),
                    "' == 'true' else -1",
                ]),
                # Mem/IncrementalMemory is intentionally NOT set here as a
                # native parameter -- 2026-07-21: launch_ros's parameter-dict
                # YAML serialization coerces a plain "true"/"false" string
                # (even from a PythonExpression) into an actual YAML bool,
                # and rtabmap declares this parameter as string-typed
                # (all RTAB-Map Mem/*, Vis/*, etc. override params are),
                # so passing a bool crashes it at startup with
                # InvalidParameterTypeException. Set via the command-line
                # `args` string below instead, like Mem/InitWMWithAllNodes
                # already was -- command-line "--Param value" args go
                # through rtabmap's own string-based arg parser, no YAML
                # type inference involved, so this is actually the SAFER
                # mechanism for these, not a workaround.
            }],
            remappings=[
                ('map', 'map'),
                ('scan', LaunchConfiguration('scan_topic', default='/scan')),
                ('tag_detections', LaunchConfiguration('tag_topic')),
                ('odom', 'odom'),
                ('imu', '/cam_high/camera/imu/filtered'),
                ('goal_out', '/goal_pose'),
            ],
            arguments=[PythonExpression([
                "('--delete_db_on_start ' if '",
                LaunchConfiguration('localization'),
                "' != 'true' and '",
                LaunchConfiguration('delete_db_on_start'),
                "' == 'true' else '')",
                # See the parameters-dict comment above (near rgbd_cameras)
                # for why this is a command-line arg and not a native param.
                " + ('--Mem/IncrementalMemory false ' if '",
                LaunchConfiguration('localization'),
                "' == 'true' else '--Mem/IncrementalMemory true ')",
                " + '"
                " --Rtabmap/DetectionRate 1.0"
                " --Reg/Strategy 2"
                " --Reg/Force3DoF true"
                # 2026-07-22: multi-camera (rgbd_cameras=0 / RGBDX array) hits
                # a hard, unconditional error at runtime with the default
                # Vis/EstimationType=1 (PnP) -- RegistrationVis.cpp explicitly
                # requires OpenGV for multi-camera 2D-3D PnP, which this
                # apt-installed rtabmap_slam was NOT built with (confirmed via
                # #ifndef RTABMAP_OPENGV guard in the vendored source). Every
                # loop-closure attempt was silently rejected as a result.
                # Fix: force 3D-3D estimation (0), which has no such
                # dependency and is exactly what the error message itself
                # recommends ("Use 3D-3D registration approach instead for
                # multi-camera"). Single-camera mode is unaffected either way.
                " --Vis/EstimationType 0"
                # 2026-07-21: Reg/Strategy, Vis/EstimationType,
                # Vis/PnPReprojError, Vis/BundleAdjustment, and
                # Vis/FeatureType were all swapped through one at a time
                # tonight chasing "always exactly 0 inliers" -- every
                # combination failed identically, which is what pointed
                # at a resolution mismatch (see depth_module.depth_profile
                # comment above) instead of a registration bug. Reverted
                # all of them back to defaults/originals here now that
                # the real fix (848x480) is in above.
                # Reg/Strategy 2 (Vis+ICP) was on already but had no
                # explicit Icp/* tuning -- it was running on RTAB-Map's
                # bare defaults, which assume a denser/3D scan than our
                # merged 2D /scan (S2 lidar ring + cam_high depth cone).
                # PointToPlane is left OFF on purpose: it needs real
                # surface normals, and a single-ring 2D LaserScan has no
                # z-variation to compute reliable normals from -- point-
                # to-point (the default) is the correct mode here.
                " --Icp/PointToPlane false"
                " --Icp/VoxelSize 0.05"
                " --Icp/MaxCorrespondenceDistance 0.15"
                " --Icp/CorrespondenceRatio 0.3"
                " --Icp/Iterations 30"
                " --Vis/FeatureType 6"
                " --Vis/MaxFeatures 500"
                " --Vis/MinInliers 6"
                " --Vis/MinDepth 0.3"
                " --Kp/MaxFeatures 500"
                " --Kp/DetectorStrategy 6"
                " --RGBD/OptimizeMaxError 3.0"
                " --RGBD/ProximityBySpace true"
                " --RGBD/ProximityMaxGraphDepth 50"
                " --RGBD/AngularUpdate 0.05"
                " --RGBD/LinearUpdate 0.05"
                " --Mem/STMSize 30"
                " --Mem/RehearsalSimilarity 0.6"
                # Default false: in localization mode (Mem/IncrementalMemory
                # false), Working Memory only ever seeds from "the previous
                # session" per RTAB-Map's own param docs -- for us that left
                # WM permanently empty at startup (confirmed via nav.log:
                # "The working memory is empty ... no loop closure can be
                # detected", every run, regardless of map). RTAB-Map's own
                # demo launch files (turtlebot4, isaac, stereo_outdoor) all
                # set this true for exactly this localize-on-saved-map case.
                " --Mem/InitWMWithAllNodes true"
                " --Grid/MaxGroundHeight 0.15"
                " --Grid/MaxObstacleHeight 2.0"
                " --Grid/NormalsSegmentation true"
                " --Grid/CellSize 0.05"
                # Was 3.0 (matched only cam_high's depth cone). With the
                # lidar now feeding the grid too, capping at 3.0 would
                # silently throw away its extra range -- 8.0 is a
                # reasonable indoor ceiling for the S2's real range.
                " --Grid/RangeMax 8.0"
                " --Grid/RangeMin 0.3"
                " --Grid/NoiseFilteringRadius 0.1"
                " --Grid/NoiseFilteringMinNeighbors 3"
                "'",
            ])],
        )]),

        # rtabmap_viz (optional GUI, mapping sessions only -- see rtabmap_viz
        # arg). Not verified to actually render multi-camera RGBDX images
        # correctly (rgbd_cameras=0 passed through for parity, best-effort);
        # if its image panes come up blank with 2 cameras, that's a cosmetic
        # loss only -- doesn't affect what actually gets recorded/localized.
        Node(
            package='rtabmap_viz', executable='rtabmap_viz',
            name='rtabmap_viz', namespace='rtabmap', output='screen',
            parameters=[{
                'subscribe_depth': False,
                'subscribe_rgbd': True,
                'subscribe_rgb': False,
                'subscribe_stereo': False,
                'subscribe_scan': True,
                'subscribe_scan_cloud': False,
                'subscribe_user_data': False,
                'subscribe_odom_info': False,
                'frame_id': 'base_link',
                'odom_frame_id': 'odom',
                'wait_for_transform': 5.0,
                'approx_sync': True,
                'topic_queue_size': 10,
                'sync_queue_size': 10,
                'qos_image': 1,
                'qos_scan': 1,
                'qos_odom': 1,
                'qos_camera_info': 1,
                'qos_user_data': 1,
                'rgbd_cameras': 0,
            }],
            remappings=[
                ('scan', '/scan'),
                ('odom', 'odom'),
            ],
            condition=IfCondition(LaunchConfiguration('rtabmap_viz')),
        ),
    ])
