#!/usr/bin/env python3

"""
RTAB-Map Mapping Launch File

Builds a 3D map (and live 2D occupancy grid) using the cam_high D405 camera
PLUS the RPLIDAR S2 (subscribe_scan, folded in via laser_scan_merger),
paired with wheel odometry from the SLATE base.  The lidar gives the
occupancy grid far longer range and much wider coverage than the camera's
narrow ~3m depth cone alone. Run this alongside aloha_bringup (with
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
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

CAM_HIGH_SERIAL = '218622272634'


def generate_launch_description():
    pkg_rtabmap_launch = get_package_share_directory('rtabmap_launch')
    rtabmap_launch_path = os.path.join(
        pkg_rtabmap_launch, 'launch', 'rtabmap.launch.py'
    )

    default_rgb = '/cam_high/camera/color/image_rect_raw'
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
            'map_name', default_value='rtabmap',
            description='Session name.  Database is saved to ~/maps/<map_name>.db. '
                        'Use the same name when saving the 2D grid.',
        ),

        # ---- RPLIDAR S2 (front, near-bottom mount, rotated 90deg CW) -----
        # TODO(robot): x/y/z below are still placeholders -- measure on the
        # robot (same tape-measure approach as cam_high's TF below). yaw is
        # set from a known fact, not a placeholder: the sensor is physically
        # mounted rotated 90deg, and rotating its raw feed 90deg CCW recovers
        # true robot-forward -- i.e. base_angle = sensor_angle + 90deg, so
        # this TF's yaw = +pi/2.
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
            'rplidar_x', default_value='0.25',
            description='PLACEHOLDER -- still needs measuring. base_link -> '
                        'rplidar_link x offset (m): forward/back distance '
                        'from base_link origin to the lidar.',
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

        # -- base_footprint -> base_link static TF -------------------------
        # The SLATE driver publishes odom -> base_footprint.  RTAB-Map uses
        # frame_id=base_link.  This link completes the chain:
        #   odom -> base_footprint -> base_link -> camera_link -> ...
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_footprint_to_base_link',
            output='screen',
            arguments=[
                '0', '0', '0.1', '0', '0', '0',
                'base_footprint', 'base_link',
            ],
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
                'initial_reset': True,
                'enable_color': True,
                'enable_depth': True,
                'enable_infra': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'align_depth.enable': True,
                'depth_module.profile': '640,480,15',
                'rgb_camera.profile': '640,480,15',
                'rgb_camera.enable_auto_exposure': True,
                'depth_module.enable_auto_exposure': True,
            }],
        ),

        # -- base_link -> camera_link static TF ----------------------------
        # Position of cam_high on the robot (x=0.21m forward, z=1.03m up,
        # yaw=90deg).  Verify this matches the physical camera mounting.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_tf',
            output='screen',
            arguments=[
                '--x', '0.2098050',
                '--y', '0',
                '--z', '1.031778',
                '--yaw', '0',
                '--pitch', '0',
                '--roll', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'camera_link',
            ],
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
                'scan_height': 300,
                'scan_row_step': 1,
                'scan_time': 0.033,
                'range_min': 0.1,
                'range_max': 3.0,
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
                'input_topics': ['/scan_rplidar', '/scan_depth_cam_high'],
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
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rtabmap_launch_path),
            launch_arguments=[
                ('rgb_topic',          LaunchConfiguration('rgb_topic')),
                ('depth_topic',        LaunchConfiguration('depth_topic')),
                ('camera_info_topic',  LaunchConfiguration('camera_info_topic')),

                ('frame_id',           'base_link'),
                ('odom_frame_id',      'odom'),
                ('visual_odometry',    'false'),

                ('rgbd_sync',          'true'),
                ('subscribe_rgbd',     'true'),
                ('approx_sync',        'true'),
                ('approx_rgbd_sync',   'true'),
                ('approx_sync_max_interval', '0.05'),

                # Fold the (lidar + depth-cam) merged scan into RTAB-Map's
                # own occupancy grid generation -- adds real range/coverage
                # beyond cam_high's ~3m depth cone. Registration itself is
                # still vision-only (Reg/Strategy 0 below); this only feeds
                # the grid, so it's a safe, additive change -- doesn't
                # touch how loop closures/localization corrections work.
                ('subscribe_scan',    'true'),
                ('scan_topic',        '/scan'),

                ('wait_for_transform', '5.0'),
                ('qos',               '1'),
                ('topic_queue_size',  '10'),
                ('queue_size',        '10'),

                ('rtabmap_viz',        LaunchConfiguration('rtabmap_viz')),
                ('rviz',               LaunchConfiguration('rviz')),

                ('localization',       LaunchConfiguration('localization')),
                ('database_path',     PythonExpression([
                    "str(__import__('pathlib').Path.home() / 'maps' / ('",
                    LaunchConfiguration('map_name'),
                    "' + '.db'))",
                ])),

                ('args', PythonExpression([
                    "('--delete_db_on_start ' if '",
                    LaunchConfiguration('localization'),
                    "' != 'true' and '",
                    LaunchConfiguration('delete_db_on_start'),
                    "' == 'true' else '')",
                    " + '"
                    " --Rtabmap/DetectionRate 1.0"
                    " --Reg/Strategy 0"
                    " --Reg/Force3DoF true"
                    " --Vis/FeatureType 6"
                    " --Vis/MaxFeatures 500"
                    " --Vis/MinInliers 10"
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
                ])),
            ],
        ),
    ])
