#!/usr/bin/env python3
"""
Launch file for mobile_ai robot simulation with path planner.

This launches the mobile_ai dual-arm mobile robot in simulation
with MoveIt and the interactive path planner demo.

Usage:
    # Fake hardware (lightweight)
    ros2 launch aloha mobile_ai_sim.launch.py

    # Gazebo (full physics)
    ros2 launch aloha mobile_ai_sim.launch.py use_gazebo:=true

    # Specify arm for single-arm planning
    ros2 launch aloha mobile_ai_sim.launch.py arm:=left
    ros2 launch aloha mobile_ai_sim.launch.py arm:=right
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    Command,
    FindExecutable,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # Declare arguments
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_gazebo',
            default_value='false',
            description='Use Gazebo simulation (true) or fake hardware (false)',
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz for visualization',
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'arm',
            default_value='left',
            choices=['left', 'right', 'both'],
            description='Which arm to control (left, right, or both)',
        )
    )
    
    # Get launch arguments
    use_gazebo = LaunchConfiguration('use_gazebo')
    use_rviz = LaunchConfiguration('use_rviz')
    arm = LaunchConfiguration('arm')
    
    # Robot description from trossen_arm_description package
    # Need to pass parameters for wxai macro
    robot_description = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            FindPackageShare('trossen_arm_description'),
            'urdf',
            'mobile_ai.urdf.xacro'
        ]),
        ' ros2_control_hardware_type:=mock_components',
        ' ip_address:=192.168.1.1',  # Dummy IP for simulation
    ])
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_gazebo,
        }]
    )
    
    # Joint state publisher (for fake hardware)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': use_gazebo,
        }],
        condition=UnlessCondition(use_gazebo),
    )
    
    # RViz
    rviz_config = PathJoinSubstitution([
        FindPackageShare('trossen_arm_description'),
        'rviz',
        'mobile_ai.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz),
    )
    
    # MoveIt configuration (if you have it)
    # For now, we'll launch basic visualization
    # You can add MoveIt launch here once configured
    
    # Interactive demo for left arm
    left_arm_demo = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='aloha',
                executable='sim_planner_interactive',
                name='left_arm_planner',
                output='screen',
                parameters=[{
                    'robot_name': 'follower_left',
                    'planning_group': 'follower_left_arm',
                    'use_sim_time': use_gazebo,
                }],
                remappings=[
                    ('/robot_description', '/robot_description'),
                ],
                # Only launch if arm is 'left' or 'both'
                # Note: LaunchConfiguration conditions need OpaqueFunction
            )
        ],
    )
    
    return LaunchDescription(
        declared_arguments + [
            robot_state_publisher,
            joint_state_publisher,
            rviz_node,
            # left_arm_demo,  # Enable once MoveIt is configured
        ]
    )

