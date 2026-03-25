#!/usr/bin/env python3

"""
Navigate Mission Launch File

Combines RTAB-Map visual localization, a 2D map server for A* path planning,
and the simple_nav_planner into a single launch.  Pair with the
navigate_mission script to auto-localize and drive to waypoints.

Prerequisites:
    # Build a map first:
    ros2 launch aloha rtabmap_mapping.launch.py
    # Save the 2D occupancy grid:
    ros2 run nav2_map_server map_saver_cli -f ~/maps/my_map -t /rtabmap/map

    # Start the robot base (cameras handled by this launch):
    ros2 launch aloha aloha_bringup.launch.py use_cameras:=false

Usage:
    # Terminal 2: launch localization + planning + RViz
    ros2 launch aloha navigate_mission.launch.py \\
        map_file:=/home/aloha/maps/building16p3.yaml

    # Terminal 3: run the autonomous mission
    ros2 run aloha navigate_mission
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    aloha_pkg = FindPackageShare('aloha')

    # ==================== Arguments =====================================

    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=PathJoinSubstitution([aloha_pkg, 'maps', 'my_map.yaml']),
        description='Path to the 2D occupancy grid YAML (for A* planning)',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz with top-down localization view',
    )
    rtabmap_viz_arg = DeclareLaunchArgument(
        'rtabmap_viz', default_value='false',
        description='Launch RTAB-Map GUI (shows feature matching)',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
    )
    max_linear_velocity_arg = DeclareLaunchArgument(
        'max_linear_velocity', default_value='0.5',
        description='Maximum linear velocity (m/s)',
    )
    max_angular_velocity_arg = DeclareLaunchArgument(
        'max_angular_velocity', default_value='1.5',
        description='Maximum angular velocity (rad/s)',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    # ==================== RTAB-Map Localization ==========================

    rtabmap_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                aloha_pkg, 'launch', 'rtabmap_localization.launch.py',
            ])
        ),
        launch_arguments=[
            ('use_rviz', LaunchConfiguration('use_rviz')),
            ('rtabmap_viz', LaunchConfiguration('rtabmap_viz')),
        ],
    )

    # ==================== Map Server (2D grid for A*) ====================

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': LaunchConfiguration('map_file'),
            'use_sim_time': use_sim_time,
        }],
    )

    map_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='map_lifecycle_manager',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['map_server'],
            'use_sim_time': use_sim_time,
        }],
    )

    # ==================== Simple Nav Planner =============================

    nav_planner_node = Node(
        package='aloha',
        executable='simple_nav_planner',
        name='simple_nav_planner',
        output='screen',
        parameters=[{
            'use_nav2': False,
            'lookahead_distance': 1.0,
            'max_linear_velocity': LaunchConfiguration('max_linear_velocity'),
            'max_angular_velocity': LaunchConfiguration('max_angular_velocity'),
            'goal_tolerance': 0.25,
            'robot_radius': 0.15,
            'occupancy_threshold': 95,
            'use_trajectory_optimization': True,
            'smoothing_weight': 0.8,
            'max_acceleration': 0.5,
            'enable_collision_avoidance': True,
            'safety_distance': 0.4,
            'emergency_stop_distance': 0.1,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/odom', '/mobile_base/odom'),
            ('/cmd_vel', '/mobile_base/cmd_vel'),
        ],
    )

    # ==================== Launch Description =============================

    return LaunchDescription([
        map_file_arg,
        use_rviz_arg,
        rtabmap_viz_arg,
        use_sim_time_arg,
        max_linear_velocity_arg,
        max_angular_velocity_arg,

        rtabmap_localization,
        map_server_node,
        map_lifecycle_node,
        nav_planner_node,
    ])
