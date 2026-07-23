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
    use_ekf_odom_arg = DeclareLaunchArgument(
        'use_ekf_odom', default_value='true',
        description='Feed simple_nav_planner from /odometry/filtered '
                    '(wheel odom + IMU gyro yaw-rate fusion -- rtabmap_mapping '
                    'already runs ekf_node when this is true) instead of raw '
                    '/mobile_base/odom. Set false to roll back if it misbehaves.',
    )
    use_auto_explore_arg = DeclareLaunchArgument(
        'use_auto_explore', default_value='false',
        description='true: frontier-based autonomous exploration drives the '
                    'robot itself (original use of this launch file). '
                    'false (default): no autonomous driving -- use with '
                    'nav_web_viewer for user-clicked goals instead. Left off '
                    'by default because auto_explore also currently publishes '
                    'straight to /mobile_base/cmd_vel, bypassing nav_deadman.',
    )
    use_rplidar_arg = DeclareLaunchArgument(
        'use_rplidar', default_value='true',
        description='Include the RPLIDAR S2 in the merged /scan. false to '
                    'isolate/rule out lidar contribution (e.g. comparing '
                    'against camera-only obstacle data).',
    )
    use_cam_low_back_arg = DeclareLaunchArgument(
        'use_cam_low_back', default_value='true',
        description='Include the rear camera as a second RGBD source. false '
                    'for cam_high-only mode (see rtabmap_mapping.launch.py '
                    'for the rgbd_image/rgbd_cameras single-camera wiring '
                    'this now correctly switches to).',
    )
    use_odom_locked_map_arg = DeclareLaunchArgument(
        'use_odom_locked_map', default_value='true',
        description='true (default): map->odom TF comes from nav_web_viewer\'s '
                    'anchor (wheel+IMU odometry only), not rtabmap\'s own '
                    'SLAM-corrected (but jittery/jumpy) pose. See '
                    'rtabmap_mapping.launch.py for the full rationale.',
    )
    use_floorplan_map_arg = DeclareLaunchArgument(
        'use_floorplan_map', default_value='true',
        description='true (default): simple_nav_planner plans A* against '
                    '/floorplan_map -- the architectural floorplan\'s own '
                    'walls, warped into the robot frame via nav_web_viewer\'s '
                    'anchor (published fresh on every Confirm Anchor) -- '
                    'instead of the live, noisy SLAM-built /rtabmap/map. '
                    'Live sensors still gate real-time collision avoidance '
                    'either way, unaffected by this. false rolls back to the '
                    'live grid (also needed if nav_web_viewer/no anchor has '
                    'been set yet -- /floorplan_map won\'t exist until then).',
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
            ('use_ekf_odom', LaunchConfiguration('use_ekf_odom')),
            ('use_rplidar', LaunchConfiguration('use_rplidar')),
            ('use_cam_low_back', LaunchConfiguration('use_cam_low_back')),
            ('use_odom_locked_map', LaunchConfiguration('use_odom_locked_map')),
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
            # +2in (0.0508m) over the previous 0.27 -- 2026-07-22, reported
            # slight physical bumping ON TURNS specifically. This is the
            # WORST-CASE corner-to-pivot radius used by the LIVE safety
            # checks (check_collision_ahead/_raw_forward_clearance), so a
            # turn sweeping the physical footprint wider than this value
            # accounts for is exactly what a live-reactive-but-not-quite-
            # generous-enough radius would miss.
            'robot_radius': 0.3208,
            # Back to 0.6048 (default 0.3 + 12in) -- 2026-07-22 tried
            # bumping this globally (first to 0.681, then 0.643) for a
            # "bumping on turns" report, but a GLOBAL inflation bump also
            # pushes every straight-line segment further from walls, which
            # overcorrected (routes hugging corridor centers unnecessarily)
            # for a problem that was only ever about turns specifically.
            # Reverted here; the actual fix is simple_nav_planner.py's
            # _widen_turns() -- a post-plan-path step that nudges ONLY
            # sharp-direction-change waypoints outward a little, leaving
            # every straight run exactly as A* planned it. This value is
            # the A* PLANNED-path clearance from walls in general (not
            # robot_radius, not baked into the floorplan obstacle grid
            # itself, see that comment's history below).
            'inflation_radius': 0.6048,
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
            ('/map', PythonExpression([
                "'/floorplan_map' if '",
                LaunchConfiguration('use_floorplan_map'),
                "' == 'true' else '/rtabmap/map'",
            ])),
            # use_ekf_odom:=true (default) -> the wheel+IMU fusion output
            # rtabmap_mapping.launch.py's ekf_node publishes; false -> raw
            # wheel odom, same rollback pattern as navigate_mission.launch.py.
            ('/odom', PythonExpression([
                "'/odometry/filtered' if '",
                LaunchConfiguration('use_ekf_odom'),
                "' == 'true' else '/mobile_base/odom'",
            ])),
            # Planner output goes to an INTERMEDIATE topic, gated to the base
            # by nav_deadman below -- never straight to /mobile_base/cmd_vel.
            ('/cmd_vel', '/nav_cmd_vel'),
        ],
    )

    # ==================== Deadman "hold-to-run" safety gate ==============
    # Same as navigate_mission.launch.py: passes /nav_cmd_vel ->
    # /mobile_base/cmd_vel ONLY while L2 (button 6) is held on
    # /mobile_base/joy; otherwise publishes zero at 20Hz.
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

    # ==================== Auto Explore (off by default) ==================
    # NOTE: this node still publishes straight to /mobile_base/cmd_vel,
    # bypassing nav_deadman -- fine while gated off by default, but fix that
    # remap the same way as nav_planner_node above before ever enabling it
    # for a real drive.

    auto_explore_node = Node(
        package='aloha',
        executable='auto_explore',
        name='auto_explore',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_auto_explore')),
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
            'robot_radius': 0.3208,  # kept in sync with nav_planner_node's value above
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
        use_ekf_odom_arg,
        use_auto_explore_arg,
        use_rplidar_arg,
        use_cam_low_back_arg,
        use_odom_locked_map_arg,
        use_floorplan_map_arg,

        rtabmap_mapping,
        nav_planner_node,
        nav_deadman_node,
        auto_explore_node,
        robot_pose_marker_node,
    ])
