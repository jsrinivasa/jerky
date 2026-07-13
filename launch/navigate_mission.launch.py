#!/usr/bin/env python3

"""
Navigate Mission Launch File

Combines RTAB-Map visual localization, a 2D map server for A* path planning,
and the simple_nav_planner into a single launch.  Pair with the
navigate_mission script to auto-localize and drive to waypoints.

Prerequisites:
    # Build a map first (give it a name):
    ros2 launch aloha rtabmap_mapping.launch.py map_name:=building16_east
    # Save the 2D occupancy grid with the same name:
    ros2 run nav2_map_server map_saver_cli -f ~/maps/building16_east -t /rtabmap/map

    # Start the robot base (cameras handled by this launch):
    ros2 launch aloha aloha_bringup.launch.py use_cameras:=false

Usage:
    # Terminal 2: launch localization + planning + RViz (use same map_name)
    ros2 launch aloha navigate_mission.launch.py \\
        map_name:=building16_east

    # Terminal 3: run the autonomous mission
    ros2 run aloha navigate_mission
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    aloha_pkg = FindPackageShare('aloha')

    # ==================== Arguments =====================================

    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='rtabmap',
        description='Session name.  Loads ~/maps/<map_name>.db for localization '
                    'and ~/maps/<map_name>.yaml for A* planning.',
    )
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=PythonExpression([
            "str(__import__('pathlib').Path.home() / 'maps' / ('",
            LaunchConfiguration('map_name'),
            "' + '.yaml'))",
        ]),
        description='Path to the 2D occupancy grid YAML (for A* planning). '
                    'Derived from map_name by default; override to use a different file.',
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
    enable_collision_avoidance_arg = DeclareLaunchArgument(
        'enable_collision_avoidance', default_value='true',
        description='React to live /scan obstacles (from cam_high depth '
                    'and/or the RPLIDAR), not just the static map, while '
                    'following a path.',
    )

    # RPLIDAR + depth-cam scan + merger now live in rtabmap_mapping.launch.py
    # (included below via rtabmap_localization), since that's the file used
    # standalone for actual mapping sessions too -- see it for the node
    # definitions and TODO(robot) measurement notes. These just pass
    # overrides through the include chain.
    use_rplidar_arg = DeclareLaunchArgument('use_rplidar', default_value='true')
    rplidar_port_arg = DeclareLaunchArgument('rplidar_port', default_value='/dev/rplidar')
    rplidar_x_arg = DeclareLaunchArgument('rplidar_x', default_value='0.25')
    rplidar_y_arg = DeclareLaunchArgument('rplidar_y', default_value='0.0')
    rplidar_z_arg = DeclareLaunchArgument('rplidar_z', default_value='0.27')
    rplidar_yaw_arg = DeclareLaunchArgument('rplidar_yaw', default_value='1.5708')

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
            ('map_name', LaunchConfiguration('map_name')),
            ('use_rplidar', LaunchConfiguration('use_rplidar')),
            ('rplidar_port', LaunchConfiguration('rplidar_port')),
            ('rplidar_x', LaunchConfiguration('rplidar_x')),
            ('rplidar_y', LaunchConfiguration('rplidar_y')),
            ('rplidar_z', LaunchConfiguration('rplidar_z')),
            ('rplidar_yaw', LaunchConfiguration('rplidar_yaw')),
        ],
    )

    # ==================== Map Server (2D grid for A*) ====================
    # Uses a simple custom publisher instead of nav2_map_server to avoid
    # lifecycle management issues (nav2_map_server gets stuck in 'unconfigured'
    # on rapid restarts due to bond timeout failures).

    map_server_node = Node(
        package='aloha',
        executable='static_map_publisher',
        name='static_map_publisher',
        output='screen',
        parameters=[{
            'map_file': LaunchConfiguration('map_file'),
            'republish_interval': 5.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # Lidar + depth-cam scan + merger are launched by rtabmap_mapping.launch.py
    # (via rtabmap_localization above) -- /scan is already available here for
    # simple_nav_planner below without relaunching those nodes.

    # ==================== Robot Pose Marker (large disc + heading arrow) ==

    robot_pose_marker_node = Node(
        package='aloha',
        executable='robot_pose_marker',
        name='robot_pose_marker',
        output='screen',
        parameters=[{
            'robot_radius': 0.26,
            'arrow_length': 0.6,
            'publish_rate': 10.0,
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
            'robot_radius': 0.25,
            'occupancy_threshold': 95,
            'use_trajectory_optimization': False,
            'smoothing_weight': 0.8,
            'max_acceleration': 0.3,
            'enable_collision_avoidance': LaunchConfiguration('enable_collision_avoidance'),
            'safety_distance': 0.5,
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
        map_name_arg,
        map_file_arg,
        use_rviz_arg,
        rtabmap_viz_arg,
        use_sim_time_arg,
        max_linear_velocity_arg,
        max_angular_velocity_arg,
        enable_collision_avoidance_arg,
        use_rplidar_arg,
        rplidar_port_arg,
        rplidar_x_arg,
        rplidar_y_arg,
        rplidar_z_arg,
        rplidar_yaw_arg,

        rtabmap_localization,
        map_server_node,
        nav_planner_node,
        robot_pose_marker_node,
    ])
