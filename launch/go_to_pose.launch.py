#!/usr/bin/env python3

"""
Go To Named Pose — All-in-one launch file

Reads the target pose from config/named_poses.yaml, selects the correct map,
launches the full navigation stack (RTAB-Map localization + map server +
planner), and starts the go_to_pose node.

Usage:
    ros2 launch aloha go_to_pose.launch.py pose_name:=elevator
    ros2 launch aloha go_to_pose.launch.py pose_name:=conference_room
    ros2 launch aloha go_to_pose.launch.py pose_name:=team_room_31
    ros2 launch aloha go_to_pose.launch.py pose_name:=parking

Prerequisites:
    ros2 launch aloha aloha_bringup.launch.py use_cameras:=false

Optional arguments:
    use_rviz:=true|false   (default true)
    rtabmap_viz:=false     (default false)
"""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _load_pose_config(poses_file: str, pose_name: str) -> dict:
    """Load a single pose entry from named_poses.yaml."""
    if not os.path.isfile(poses_file):
        raise FileNotFoundError(f"named_poses.yaml not found: {poses_file}")
    with open(poses_file, 'r') as fh:
        data = yaml.safe_load(fh)
    if pose_name not in data:
        available = ', '.join(data.keys())
        raise KeyError(
            f"Pose '{pose_name}' not found in named_poses.yaml.\n"
            f"Available: {available}"
        )
    return data[pose_name]


def launch_setup(context, *args, **kwargs):
    aloha_share = get_package_share_directory('aloha')
    poses_file  = os.path.join(aloha_share, 'config', 'named_poses.yaml')

    pose_name   = LaunchConfiguration('pose_name').perform(context)
    use_rviz    = LaunchConfiguration('use_rviz').perform(context)
    rtabmap_viz = LaunchConfiguration('rtabmap_viz').perform(context)

    # ---- Read pose + map from YAML ----------------------------------------
    try:
        pose = _load_pose_config(poses_file, pose_name)
    except (FileNotFoundError, KeyError) as exc:
        raise RuntimeError(str(exc))

    map_name = pose.get('map_name', 'rtabmap')
    map_file = pose.get(
        'map_file',
        str(Path.home() / 'maps' / f'{map_name}.yaml')
    )

    # ---- navigate_mission stack (localization + map server + planner) ------
    navigate_mission = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(aloha_share, 'launch', 'navigate_mission.launch.py')
        ),
        launch_arguments=[
            ('map_name',    map_name),
            ('map_file',    map_file),
            ('use_rviz',    use_rviz),
            ('rtabmap_viz', rtabmap_viz),
        ],
    )

    # ---- go_to_pose node (delayed so the nav stack has time to start) ------
    go_to_pose_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='aloha',
                executable='go_to_pose',
                name='go_to_pose',
                output='screen',
                parameters=[{
                    'pose_name':  pose_name,
                    'poses_file': poses_file,
                }],
            )
        ],
    )

    return [navigate_mission, go_to_pose_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'pose_name',
            description='Name of the target pose from named_poses.yaml '
                        '(e.g. elevator, conference_room, team_room_31, parking)',
        ),
        DeclareLaunchArgument('use_rviz',    default_value='true'),
        DeclareLaunchArgument('rtabmap_viz', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
