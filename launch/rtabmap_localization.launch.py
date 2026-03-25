#!/usr/bin/env python3

"""
RTAB-Map Visual Localization Launch File

Localizes on a previously built RTAB-Map database (~/maps/<map_name>.db)
using visual feature matching.  The camera sees visual landmarks it recorded
during mapping and uses them to determine its position.  Publishes the
map -> odom TF transform so Nav2 and other systems know where the robot is.

Prerequisites:
    # Build a map first (give it a name):
    ros2 launch aloha rtabmap_mapping.launch.py map_name:=building16_east
    ros2 run nav2_map_server map_saver_cli -f ~/maps/building16_east -t /rtabmap/map

    # Start the robot base (no cameras -- this launch handles the camera):
    ros2 launch aloha aloha_bringup.launch.py use_cameras:=false

Usage:
    # Localize on a specific map:
    ros2 launch aloha rtabmap_localization.launch.py map_name:=building16_east

    # Without RViz (headless):
    ros2 launch aloha rtabmap_localization.launch.py map_name:=building16_east use_rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    aloha_pkg = FindPackageShare('aloha')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz with top-down localization view',
    )
    rtabmap_viz_arg = DeclareLaunchArgument(
        'rtabmap_viz', default_value='false',
        description='Launch RTAB-Map GUI (shows feature matching)',
    )
    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='rtabmap',
        description='Session name.  Loads ~/maps/<map_name>.db for localization.',
    )

    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([aloha_pkg, 'launch', 'rtabmap_mapping.launch.py'])
        ),
        launch_arguments=[
            ('localization', 'true'),
            ('rviz', 'false'),
            ('rtabmap_viz', LaunchConfiguration('rtabmap_viz')),
            ('map_name', LaunchConfiguration('map_name')),
        ],
    )

    rviz_config = PathJoinSubstitution([
        aloha_pkg, 'rviz', 'rtabmap_localization.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        use_rviz_arg,
        rtabmap_viz_arg,
        map_name_arg,
        mapping_launch,
        rviz_node,
    ])
