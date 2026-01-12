#!/usr/bin/env python3

"""
Simulation Navigation Launch File

Test the navigation planner in simulation without real hardware.
Provides fake odometry and TF for testing.

Usage:
    ros2 launch aloha sim_navigation.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    
    # Package paths
    aloha_pkg = FindPackageShare('aloha')
    
    # Launch arguments
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='/home/aloha/interbotix_ws/src/my_robot_maps/maps/my_final_map.yaml',
        description='Path to the map YAML file'
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
    
    # Launch configurations
    map_file = LaunchConfiguration('map_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    lookahead_distance = LaunchConfiguration('lookahead_distance')
    max_linear_velocity = LaunchConfiguration('max_linear_velocity')
    max_angular_velocity = LaunchConfiguration('max_angular_velocity')
    goal_tolerance = LaunchConfiguration('goal_tolerance')
    robot_radius = LaunchConfiguration('robot_radius')
    
    # Static TF publishers for simulation
    # Publish map -> odom transform (robot starts at origin)
    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    
    # Publish odom -> base_footprint transform (will be updated by fake odometry)
    # This is just initial, the fake_odometry node will publish updates
    odom_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_base_footprint',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint']
    )
    
    # Publish base_footprint -> base_link transform
    base_footprint_to_base_link_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_to_base_link',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_footprint', 'base_link']
    )
    
    # Fake odometry publisher (publishes /odom and TF)
    fake_odometry_node = Node(
        package='aloha',
        executable='fake_odometry',
        name='fake_odometry',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'update_rate': 30.0,
        }]
    )
    
    # Map server
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_file,
            'use_sim_time': use_sim_time
        }]
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
        }]
    )
    
    # Simple Navigation Planner
    nav_planner_node = Node(
        package='aloha',
        executable='simple_nav_planner',
        name='simple_nav_planner',
        output='screen',
        parameters=[{
            'use_nav2': False,
            'lookahead_distance': lookahead_distance,
            'max_linear_velocity': max_linear_velocity,
            'max_angular_velocity': max_angular_velocity,
            'goal_tolerance': goal_tolerance,
            'robot_radius': robot_radius,
            'use_sim_time': use_sim_time,
        }]
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
        use_sim_time_arg,
        use_rviz_arg,
        lookahead_distance_arg,
        max_linear_velocity_arg,
        max_angular_velocity_arg,
        goal_tolerance_arg,
        robot_radius_arg,
        
        # TF publishers
        map_to_odom_tf,
        base_footprint_to_base_link_tf,
        
        # Nodes
        fake_odometry_node,
        map_server_node,
        map_lifecycle_node,
        nav_planner_node,
        rviz_node,
    ])

