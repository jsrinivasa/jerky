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
    rplidar_x_arg = DeclareLaunchArgument('rplidar_x', default_value='0.1397')
    rplidar_y_arg = DeclareLaunchArgument('rplidar_y', default_value='0.0')
    rplidar_z_arg = DeclareLaunchArgument('rplidar_z', default_value='0.27')
    rplidar_yaw_arg = DeclareLaunchArgument('rplidar_yaw', default_value='1.5708')
    use_ekf_odom_arg = DeclareLaunchArgument(
        'use_ekf_odom', default_value='true',
        description='Feed simple_nav_planner from /odometry/filtered '
                    '(wheel odom + IMU gyro yaw-rate fusion, see '
                    'config/ekf.yaml) instead of raw /mobile_base/odom. '
                    'Set false to roll back if the EKF misbehaves.',
    )
    continuous_mapping_arg = DeclareLaunchArgument(
        'continuous_mapping', default_value='false',
        description='false (default): locked localization against map_name, '
                    'read-only. true: keeps extending/refining the existing '
                    'DB live while navigating instead of freezing it -- see '
                    'rtabmap_localization.launch.py for the trade-off.',
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
            ('map_name', LaunchConfiguration('map_name')),
            ('use_rplidar', LaunchConfiguration('use_rplidar')),
            ('rplidar_port', LaunchConfiguration('rplidar_port')),
            ('rplidar_x', LaunchConfiguration('rplidar_x')),
            ('rplidar_y', LaunchConfiguration('rplidar_y')),
            ('rplidar_z', LaunchConfiguration('rplidar_z')),
            ('rplidar_yaw', LaunchConfiguration('rplidar_yaw')),
            ('use_ekf_odom', LaunchConfiguration('use_ekf_odom')),
            ('continuous_mapping', LaunchConfiguration('continuous_mapping')),
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
            'robot_length': 0.7112,  # 28in
            'robot_width': 0.5588,   # 22in
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
            # Was 0.25, then 0.46 (half-diagonal of a 22x28in footprint
            # assuming the pivot/base_link is at the geometric center).
            # 2026-07-16: it isn't -- pivot sits toward the back, ~18.7in to
            # the front edge but only ~9.3in to the back edge (measured along
            # the 28in length), half-width 11in. Worst case (front corners,
            # the direction that matters for forward driving) is
            # sqrt(18.7^2 + 11^2) in = sqrt(0.4742^2 + 0.2794^2) m ~= 0.55m,
            # not 0.46m -- the old value under-covered the front by ~9cm.
            # 0.58m = that 0.55m plus a bit of extra margin.
            # robot_radius is now used ONLY for live safety (emergency stop /
            # raw scan floor) -- it needs to cover the worst case (turning).
            'robot_radius': 0.58,
            # inflation_radius is used for A* path-planning inflation only.
            # 2026-07-16: was sharing robot_radius, which used the worst-case
            # diagonal (0.58m) even for straight-through corridors, where
            # only the ~11in (0.2794m) half-width matters -- rejected valid
            # paths through gaps the robot actually fits through. 0.40m =
            # half-width + ~4.5in margin for path-tracking imprecision.
            'inflation_radius': 0.40,
            'occupancy_threshold': 95,
            'use_trajectory_optimization': False,
            'smoothing_weight': 0.8,
            'max_acceleration': 0.3,
            'enable_collision_avoidance': LaunchConfiguration('enable_collision_avoidance'),
            'safety_distance': 0.6096,  # 2ft -- start slowing
            'emergency_stop_distance': 0.4572,  # 1.5ft -- full stop
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            # use_ekf_odom:=true (default) -> the wheel+IMU fusion output
            # (see config/ekf.yaml); false -> raw wheel odom, pre-fusion
            # behavior, as a rollback switch.
            ('/odom', PythonExpression([
                "'/odometry/filtered' if '",
                LaunchConfiguration('use_ekf_odom'),
                "' == 'true' else '/mobile_base/odom'",
            ])),
            # Planner output goes to an INTERMEDIATE topic, gated to the base
            # by the deadman below -- never straight to /mobile_base/cmd_vel.
            ('/cmd_vel', '/nav_cmd_vel'),
        ],
    )

    # ==================== Deadman "hold-to-run" safety gate ==============
    # Passes /nav_cmd_vel -> /mobile_base/cmd_vel ONLY while L2 (button 6) is
    # held on /mobile_base/joy; otherwise publishes zero at 20Hz (robot stays
    # put). Release L2 / drop the controller / planner stalls -> instant stop.
    # NOTE: the joystick TELEOP node (teleop_twist_joy) must NOT be running in
    # nav mode or it fights this gate on /mobile_base/cmd_vel -- run joy_node
    # only. The base's physical E-STOP is the independent hardware kill.
    nav_deadman_node = Node(
        package='aloha',
        executable='nav_deadman',
        name='nav_deadman',
        output='screen',
        parameters=[{
            'enable_button': 6,          # L2
            'joy_topic': '/mobile_base/joy',
            'input_topic': '/nav_cmd_vel',
            'output_topic': '/mobile_base/cmd_vel',
            'rate_hz': 20.0,
            'joy_timeout': 0.5,
            'cmd_timeout': 0.5,
        }],
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
        use_ekf_odom_arg,
        continuous_mapping_arg,

        rtabmap_localization,
        map_server_node,
        nav_planner_node,
        nav_deadman_node,
        robot_pose_marker_node,
    ])
