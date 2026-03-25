#!/usr/bin/env python3

"""
Autonomous Mapping Launch File

Combines RTAB-Map (mapping mode), the simple_nav_planner reading the LIVE
occupancy grid, and the frontier-based auto_explore node so the robot can
build a complete map without manual teleoperation.

Prerequisites:
    # Start the robot base (no cameras — this launch handles the camera):
    ros2 launch aloha aloha_bringup.launch.py use_cameras:=false

Usage:
    # Terminal 2: autonomous mapping (give it a name)
    ros2 launch aloha autonomous_mapping.launch.py map_name:=building16_east_auto

    # Watch in RViz as the robot explores.  When satisfied, Ctrl-C and save:
    ros2 run nav2_map_server map_saver_cli -f ~/maps/building16_east_auto -t /rtabmap/map

    # Files produced:
    #   ~/maps/building16_east_auto.db    (RTAB-Map 3D database)
    #   ~/maps/building16_east_auto.pgm   (2D occupancy grid image)
    #   ~/maps/building16_east_auto.yaml  (2D occupancy grid metadata)
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

    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='auto_explore',
        description='Session name.  Database saved to ~/maps/<map_name>.db.',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz',
    )
    rtabmap_viz_arg = DeclareLaunchArgument(
        'rtabmap_viz', default_value='false',
        description='Launch RTAB-Map GUI',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
    )
    max_linear_velocity_arg = DeclareLaunchArgument(
        'max_linear_velocity', default_value='0.3',
        description='Maximum linear velocity during exploration (m/s)',
    )
    max_angular_velocity_arg = DeclareLaunchArgument(
        'max_angular_velocity', default_value='1.0',
        description='Maximum angular velocity during exploration (rad/s)',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    # ==================== RTAB-Map MAPPING (not localization) ============

    rtabmap_mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                aloha_pkg, 'launch', 'rtabmap_mapping.launch.py',
            ])
        ),
        launch_arguments=[
            ('rviz', LaunchConfiguration('use_rviz')),
            ('rtabmap_viz', LaunchConfiguration('rtabmap_viz')),
            ('map_name', LaunchConfiguration('map_name')),
        ],
    )

    # ==================== Simple Nav Planner =============================
    # Reads the LIVE /rtabmap/map (no map_server needed).

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
            'goal_tolerance': 0.35,
            'robot_radius': 0.27,
            'occupancy_threshold': 80,
            'use_trajectory_optimization': True,
            'smoothing_weight': 0.8,
            'max_acceleration': 0.4,
            'enable_collision_avoidance': True,
            'safety_distance': 0.4,
            'emergency_stop_distance': 0.2,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/map', '/rtabmap/map'),
            ('/odom', '/mobile_base/odom'),
            ('/cmd_vel', '/mobile_base/cmd_vel'),
        ],
    )

    # ==================== Auto Explore ==================================

    auto_explore_node = Node(
        package='aloha',
        executable='auto_explore',
        name='auto_explore',
        output='screen',
        parameters=[{
            'min_frontier_size': 0.75,
            'exploration_rate': 0.2,
            'goal_timeout': 45.0,
            'return_to_start': True,
            'fov_bonus_weight': 1.5,
            'rotation_speed': 0.4,
            'goal_tolerance': 0.40,
            'blacklist_radius': 1.0,
            'consecutive_fail_limit': 3,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/map', '/rtabmap/map'),
            ('/odom', '/mobile_base/odom'),
            ('/cmd_vel', '/mobile_base/cmd_vel'),
        ],
    )

    # ==================== Robot Pose Marker ==============================

    robot_pose_marker_node = Node(
        package='aloha',
        executable='robot_pose_marker',
        name='robot_pose_marker',
        output='screen',
        parameters=[{
            'robot_radius': 0.27,
            'arrow_length': 0.6,
            'publish_rate': 10.0,
        }],
    )

    # ==================== Launch Description =============================

    return LaunchDescription([
        map_name_arg,
        use_rviz_arg,
        rtabmap_viz_arg,
        use_sim_time_arg,
        max_linear_velocity_arg,
        max_angular_velocity_arg,

        rtabmap_mapping,
        nav_planner_node,
        auto_explore_node,
        robot_pose_marker_node,
    ])
