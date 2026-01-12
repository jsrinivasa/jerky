#!/usr/bin/env python3
"""
Launch file for path planner simulation demo.

This launches:
1. Robot in simulation (fake hardware or Gazebo)
2. MoveIt for motion planning
3. RViz for visualization
4. Interactive path planner demo

Usage:
    # Using fake hardware (lightweight, no Gazebo)
    ros2 launch aloha path_planner_sim.launch.py

    # Using Gazebo (full physics simulation)
    ros2 launch aloha path_planner_sim.launch.py use_gazebo:=true

    # Specify robot model
    ros2 launch aloha path_planner_sim.launch.py robot_model:=wx250s
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # Declare arguments
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_model',
            default_value='wx250s',
            description='Interbotix robot model (e.g., wx200, wx250s, vx300s)',
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name',
            default_value='aloha_sim',
            description='Robot namespace',
        )
    )
    
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
            'demo_mode',
            default_value='interactive',
            choices=['interactive', 'auto', 'click'],
            description='Demo mode: interactive (CLI), auto (automatic), click (RViz)',
        )
    )
    
    # Get launch arguments
    robot_model = LaunchConfiguration('robot_model')
    robot_name = LaunchConfiguration('robot_name')
    use_gazebo = LaunchConfiguration('use_gazebo')
    use_rviz = LaunchConfiguration('use_rviz')
    demo_mode = LaunchConfiguration('demo_mode')
    
    # Launch MoveIt with simulation
    # hardware_type: 'gz_classic' for Gazebo, 'fake' for fake controllers
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_moveit'),
                'launch',
                'xsarm_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_model': robot_model,
            'robot_name': robot_name,
            'hardware_type': ['gz_classic'],
            'use_rviz': use_rviz,
            'use_sim_time': 'true',
        }.items(),
        condition=IfCondition(use_gazebo),
    )
    
    # Launch MoveIt with fake hardware (lighter weight)
    moveit_fake_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_moveit'),
                'launch',
                'xsarm_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_model': robot_model,
            'robot_name': robot_name,
            'hardware_type': 'fake',
            'use_rviz': use_rviz,
            'use_sim_time': 'false',
        }.items(),
        condition=UnlessCondition(use_gazebo),
    )
    
    # Launch interactive demo after delay (let MoveIt initialize)
    interactive_demo = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='aloha',
                executable='sim_planner_interactive',
                name='sim_planner_interactive',
                output='screen',
                parameters=[{
                    'robot_name': robot_name,
                    'use_sim_time': use_gazebo,
                }],
            )
        ],
    )
    
    return LaunchDescription(
        declared_arguments + [
            moveit_launch,
            moveit_fake_launch,
            interactive_demo,
        ]
    )


