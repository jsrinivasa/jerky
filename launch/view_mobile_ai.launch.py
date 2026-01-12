#!/usr/bin/env python3
"""
Simplified mobile_ai visualization launch file.

This launches just the robot visualization without MoveIt.

Usage:
    ros2 launch aloha view_mobile_ai.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    Command,
    FindExecutable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    
    # Declare arguments
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz',
        )
    )
    
    use_rviz = LaunchConfiguration('use_rviz')
    
    # Get robot description
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            FindPackageShare('trossen_arm_description'),
            'urdf',
            'mobile_ai.urdf.xacro'
        ]),
        ' ros2_control_hardware_type:=mock_components',
        ' ip_address:=192.168.1.1',
    ])
    
    robot_description = {'robot_description': robot_description_content}
    
    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )
    
    # Joint State Publisher (allows manual control in RViz)
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )
    
    # RViz
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('trossen_arm_description'),
        'rviz',
        'mobile_ai.rviz'
    ])
    
    # If config doesn't exist, use default
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(use_rviz),
        on_exit=None,
    )
    
    nodes_to_start = [
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node,
    ]
    
    return LaunchDescription(declared_arguments + nodes_to_start)

