#!/usr/bin/env python3

"""
Simple Navigation Launch File

Launches the simple navigation planner with map and localization.
Use this with your saved map to navigate using RViz2's "2D Goal Pose" tool.

Usage:
    ros2 launch aloha simple_navigation.launch.py map_file:=<path_to_map.yaml>
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
import os


def generate_launch_description():
    
    # Package paths
    aloha_pkg = FindPackageShare('aloha')
    
    # Launch arguments
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=PathJoinSubstitution([
            aloha_pkg,
            'maps',
            'my_map.yaml'
        ]),
        description='Path to the map YAML file'
    )
    
    use_nav2_arg = DeclareLaunchArgument(
        'use_nav2',
        default_value='false',
        description='Use Nav2 instead of simple planner (requires Nav2 to be running)'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz'
    )
    
    use_rtabmap_arg = DeclareLaunchArgument(
        'use_rtabmap',
        default_value='false',
        description='Use RTAB-Map for localization (requires camera). If false, assumes you have /odom from wheels.'
    )
    
    rtabmap_database_arg = DeclareLaunchArgument(
        'rtabmap_database',
        default_value='~/.ros/rtabmap.db',
        description='Path to RTAB-Map database'
    )
    
    lookahead_distance_arg = DeclareLaunchArgument(
        'lookahead_distance',
        default_value='0.5',
        description='Lookahead distance for pure pursuit controller (meters)'
    )
    
    max_linear_velocity_arg = DeclareLaunchArgument(
        'max_linear_velocity',
        default_value='0.3',
        description='Maximum linear velocity (m/s)'
    )
    
    max_angular_velocity_arg = DeclareLaunchArgument(
        'max_angular_velocity',
        default_value='1.0',
        description='Maximum angular velocity (rad/s)'
    )
    
    goal_tolerance_arg = DeclareLaunchArgument(
        'goal_tolerance',
        default_value='0.2',
        description='Goal tolerance (meters)'
    )
    
    robot_radius_arg = DeclareLaunchArgument(
        'robot_radius',
        default_value='0.35',
        description='Robot radius for collision avoidance (meters). 24in width = 0.305m radius + safety margin'
    )
    
    use_trajectory_optimization_arg = DeclareLaunchArgument(
        'use_trajectory_optimization',
        default_value='true',
        description='Enable trajectory smoothing and optimization'
    )
    
    smoothing_weight_arg = DeclareLaunchArgument(
        'smoothing_weight',
        default_value='0.5',
        description='Trajectory smoothing weight (0.1-2.0). Higher = smoother curves with more rounded corners'
    )
    
    max_acceleration_arg = DeclareLaunchArgument(
        'max_acceleration',
        default_value='0.5',
        description='Maximum linear acceleration (m/s^2)'
    )
    
    enable_collision_avoidance_arg = DeclareLaunchArgument(
        'enable_collision_avoidance',
        default_value='false',
        description='Enable real-time collision avoidance (for dynamic obstacles)'
    )
    
    safety_distance_arg = DeclareLaunchArgument(
        'safety_distance',
        default_value='0.5',
        description='Minimum safety distance to obstacles (meters)'
    )
    
    emergency_stop_distance_arg = DeclareLaunchArgument(
        'emergency_stop_distance',
        default_value='0.3',
        description='Emergency stop distance to obstacles (meters)'
    )
    
    # Launch configurations
    map_file = LaunchConfiguration('map_file')
    use_nav2 = LaunchConfiguration('use_nav2')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    use_rtabmap = LaunchConfiguration('use_rtabmap')
    rtabmap_database = LaunchConfiguration('rtabmap_database')
    lookahead_distance = LaunchConfiguration('lookahead_distance')
    max_linear_velocity = LaunchConfiguration('max_linear_velocity')
    max_angular_velocity = LaunchConfiguration('max_angular_velocity')
    goal_tolerance = LaunchConfiguration('goal_tolerance')
    robot_radius = LaunchConfiguration('robot_radius')
    use_trajectory_optimization = LaunchConfiguration('use_trajectory_optimization')
    smoothing_weight = LaunchConfiguration('smoothing_weight')
    max_acceleration = LaunchConfiguration('max_acceleration')
    enable_collision_avoidance = LaunchConfiguration('enable_collision_avoidance')
    safety_distance = LaunchConfiguration('safety_distance')
    emergency_stop_distance = LaunchConfiguration('emergency_stop_distance')
    
    # Static map -> odom transform for testing (only when NOT using RTAB-Map)
    # TODO: Replace with proper localization (AMCL) for production
    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        condition=UnlessCondition(use_rtabmap)
    )
    
    # Note: We don't need a separate map_server because RTAB-Map publishes
    # the map directly from its database when in localization mode
    # If not using RTAB-Map, we use map_server to load the map from YAML
    
    # Map server (only when NOT using RTAB-Map)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_file,
            'use_sim_time': use_sim_time
        }],
        condition=UnlessCondition(use_rtabmap)
    )
    
    # Lifecycle manager for map server
    map_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='map_lifecycle_manager',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['map_server'],
            'use_sim_time': use_sim_time
        }],
        condition=UnlessCondition(use_rtabmap)
    )
    
    # RTAB-Map in localization mode (uses the saved map)
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[{
            'database_path': rtabmap_database,
            'frame_id': 'base_footprint',
            'odom_frame_id': 'odom',
            'subscribe_depth': True,
            'subscribe_rgb': True,
            'subscribe_scan': False,
            'approx_sync': True,
            'queue_size': 30,
            'Mem/IncrementalMemory': 'false',  # Localization mode
            'Mem/InitWMWithAllNodes': 'true',  # Load all nodes
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('rgb/image', '/camera/rgb/image_rect_color'),
            ('rgb/camera_info', '/camera/rgb/camera_info'),
            ('depth/image', '/camera/depth_registered/image_raw'),
            ('odom', '/odom'),
        ],
        condition=IfCondition(use_rtabmap)
    )
    
    # RGB-D Odometry
    rgbd_odometry_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_footprint',
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'approx_sync': True,
            'queue_size': 30,
            'Odom/Strategy': '0',
            'Odom/ResetCountdown': '1',
            'Odom/GuessMotion': 'true',
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('rgb/image', '/camera/rgb/image_rect_color'),
            ('rgb/camera_info', '/camera/rgb/camera_info'),
            ('depth/image', '/camera/depth_registered/image_raw'),
        ],
        condition=IfCondition(use_rtabmap)
    )
    
    # Simple Navigation Planner
    nav_planner_node = Node(
        package='aloha',
        executable='simple_nav_planner',
        name='simple_nav_planner',
        output='screen',
        parameters=[{
            'use_nav2': use_nav2,
            'lookahead_distance': lookahead_distance,
            'max_linear_velocity': max_linear_velocity,
            'max_angular_velocity': max_angular_velocity,
            'goal_tolerance': goal_tolerance,
            'robot_radius': robot_radius,
            'use_trajectory_optimization': use_trajectory_optimization,
            'smoothing_weight': smoothing_weight,
            'max_acceleration': max_acceleration,
            'enable_collision_avoidance': enable_collision_avoidance,
            'safety_distance': safety_distance,
            'emergency_stop_distance': emergency_stop_distance,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/odom', '/mobile_base/odom'),  # Remap to SLATE base odometry
            ('/cmd_vel', '/mobile_base/cmd_vel'),  # Send commands to mobile base
        ]
    )
    
    # RViz
    rviz_config_file = PathJoinSubstitution([
        aloha_pkg,
        'rviz',
        'navigation.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz)
    )
    
    return LaunchDescription([
        # Arguments
        map_file_arg,
        use_nav2_arg,
        use_sim_time_arg,
        use_rviz_arg,
        use_rtabmap_arg,
        rtabmap_database_arg,
        lookahead_distance_arg,
        max_linear_velocity_arg,
        max_angular_velocity_arg,
        goal_tolerance_arg,
        robot_radius_arg,
        use_trajectory_optimization_arg,
        smoothing_weight_arg,
        max_acceleration_arg,
        enable_collision_avoidance_arg,
        safety_distance_arg,
        emergency_stop_distance_arg,
        
        # Nodes
        map_to_odom_tf,
        map_server_node,
        map_lifecycle_node,
        rgbd_odometry_node,
        rtabmap_node,
        nav_planner_node,
        rviz_node,
    ])

