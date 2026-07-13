#!/usr/bin/env python3

"""
RTAB-Map Mapping Launch File

Builds a 3D map (and live 2D occupancy grid) using the cam_high D405 camera
paired with wheel odometry from the SLATE base.  Run this alongside
aloha_bringup (with use_cameras:=false so the camera is not double-claimed).

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
                    " --Grid/RangeMax 3.0"
                    " --Grid/RangeMin 0.3"
                    " --Grid/NoiseFilteringRadius 0.1"
                    " --Grid/NoiseFilteringMinNeighbors 3"
                    "'",
                ])),
            ],
        ),
    ])
