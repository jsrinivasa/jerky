#!/usr/bin/env python3

"""
Navigate with SVG Floor Plan Map

Uses AMCL localization (laser scan vs 2D wall map) instead of RTAB-Map
(which requires a visual database).  This is the launch file for navigating
with maps generated from SVG floor plans via svg_to_map.py.

Pipeline:
    Depth cameras → laser scans → merged scan → AMCL → map→odom TF
    SVG map → map_server → simple_nav_planner (A* planning)
    rooms JSON → room_label_publisher (RViz labels)

Prerequisites:
    # Terminal 1: Start the robot base (cameras NOT needed here — this launch starts them)
    ros2 launch aloha aloha_bringup.launch.py use_cameras:=false

Usage:
    # Global localization (robot figures out position from scratch):
    ros2 launch aloha navigate_floorplan.launch.py \\
        map_file:=/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_walls.yaml

    # With a starting position hint (faster convergence):
    ros2 launch aloha navigate_floorplan.launch.py \\
        map_file:=/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_walls.yaml \\
        global_localization:=false \\
        initial_pose_x:=-3.85 initial_pose_y:=-21.73 initial_pose_yaw:=0.0

    # With room labels visible in RViz:
    ros2 launch aloha navigate_floorplan.launch.py \\
        map_file:=/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_walls.yaml \\
        rooms_json:=/home/aloha/interbotix_ws/src/aloha/maps/floorplan_real_2_nav_rooms.json
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
from launch_ros.descriptions import ParameterValue


def generate_launch_description():

    aloha_pkg = FindPackageShare('aloha')

    # ==================== Arguments =====================================

    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=PathJoinSubstitution([
            aloha_pkg, 'maps', 'floorplan_real_2_nav_walls.yaml',
        ]),
        description='Path to the 2D occupancy grid YAML (SVG-derived map)',
    )
    rooms_json_arg = DeclareLaunchArgument(
        'rooms_json',
        default_value='',
        description='Path to rooms JSON file for RViz labels (empty = no labels)',
    )
    global_localization_arg = DeclareLaunchArgument(
        'global_localization', default_value='true',
        description='true = robot localizes from scratch (slow). '
                    'false = use initial_pose hint (fast).',
    )
    initial_pose_x_arg = DeclareLaunchArgument(
        'initial_pose_x', default_value='0.0',
        description='Initial X position hint (only when global_localization:=false)',
    )
    initial_pose_y_arg = DeclareLaunchArgument(
        'initial_pose_y', default_value='0.0',
        description='Initial Y position hint (only when global_localization:=false)',
    )
    initial_pose_yaw_arg = DeclareLaunchArgument(
        'initial_pose_yaw', default_value='0.0',
        description='Initial yaw in radians (only when global_localization:=false)',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
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
    camera_serial_arg = DeclareLaunchArgument(
        'camera_serial', default_value='218622272634',
        description='Serial number of cam_high RealSense D405',
    )
    launch_camera_arg = DeclareLaunchArgument(
        'launch_camera', default_value='true',
        description='Launch camera nodes (false if already running)',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    launch_camera = LaunchConfiguration('launch_camera')
    global_localization = LaunchConfiguration('global_localization')
    rooms_json = LaunchConfiguration('rooms_json')

    # ==================== TF Setup ======================================

    base_footprint_to_base_link_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_to_base_link',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_footprint', 'base_link'],
    )

    # >>> [FIX #1] Camera TF: base_link → camera_link
    # >>> The yaw value MUST match the physical D405 camera mounting angle.
    # >>> yaw=0 means camera depth sensor faces straight ahead (robot X-axis).
    # >>> yaw=1.5708 means camera is rotated 90° CW on the mount.
    # >>> IMPORTANT: camera_static_tf_publisher.py defaults to yaw=1.5708 —
    # >>> if that node is ever used instead, keep values consistent.
    # >>> DIAGNOSIS: In RViz, display /scan on the map. If beams don't align
    # >>> with walls, flip this yaw.
    camera_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_base_link',
        arguments=[
            '--x', '0.2098050', '--y', '0', '--z', '1.031778',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'base_link', '--child-frame-id', 'camera_link'
        ],
    )

    rear_camera_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_rear_camera_tf',
        arguments=[
            '--x', '0.0', '--y', '0', '--z', '1.4146',
            '--yaw', '3.14159', '--pitch', '0', '--roll', '0',
            '--frame-id', 'base_link', '--child-frame-id', 'rear_cam_link',
        ],
    )

    # ==================== Front Camera (D405) ===========================

    camera_front_node = Node(
        package='realsense2_camera',
        namespace='cam_high',
        name='camera',
        executable='realsense2_camera_node',
        output='screen',
        parameters=[{
            'initial_reset': True,
            'serial_no': ParameterValue(
                LaunchConfiguration('camera_serial'), value_type=str),
            'enable_depth': True,
            'enable_color': True,
            'align_depth.enable': True,
            'depth_module.profile': '640,480,15',
            'rgb_camera.profile': '640,480,15',
            'enable_infra': False,
            'enable_infra1': False,
            'enable_infra2': False,
        }],
        condition=IfCondition(launch_camera),
    )

    # ==================== Rear Camera (D435i) ===========================

    camera_rear_node = Node(
        package='realsense2_camera',
        namespace='cam_rear',
        name='rear_cam',
        executable='realsense2_camera_node',
        output='screen',
        parameters=[{
            'camera_name': 'rear_cam',
            'initial_reset': True,
            'serial_no': '349522070494',
            'enable_depth': True,
            'enable_color': False,
            'align_depth.enable': False,
            'depth_module.profile': '640,480,15',
            'enable_infra': False,
            'enable_infra1': False,
            'enable_infra2': False,
        }],
        condition=IfCondition(launch_camera),
    )

    # ==================== Depth → LaserScan =============================

    depthimage_to_laserscan_front = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan_front',
        output='screen',
        parameters=[{
            'scan_height': 300,
            'scan_row_step': 1,
            'scan_time': 0.033,
            'range_min': 0.1,
            'range_max': 3.0,
            'output_frame': 'camera_link',
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('depth', '/cam_high/camera/depth/image_rect_raw'),
            ('depth_camera_info', '/cam_high/camera/depth/camera_info'),
            ('scan', '/scan_front'),
        ],
    )

    depthimage_to_laserscan_rear = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan_rear',
        output='screen',
        parameters=[{
            'scan_height': 300,
            'scan_row_step': 1,
            'scan_time': 0.033,
            'range_min': 0.1,
            'range_max': 5.0,
            'output_frame': 'rear_cam_link',
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('depth', '/cam_rear/rear_cam/depth/image_rect_raw'),
            ('depth_camera_info', '/cam_rear/rear_cam/depth/camera_info'),
            ('scan', '/scan_rear'),
        ],
    )

    # ==================== Laser Scan Merger ==============================

    laser_scan_merger_node = Node(
        package='aloha',
        executable='laser_scan_merger',
        name='laser_scan_merger',
        output='screen',
        parameters=[{
            'target_frame': 'base_link',
            'range_min': 0.1,
            'range_max': 5.0,
            'publish_rate': 15.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # ==================== Map Server ====================================

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': LaunchConfiguration('map_file'),
            'use_sim_time': use_sim_time,
        }],
    )

    map_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='map_lifecycle_manager',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['map_server'],
            'use_sim_time': use_sim_time,
        }],
    )

    # ==================== AMCL Localization ==============================

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,

            # Motion model — trust odometry heavily; SVG map has clutter
            # the lidar never sees, so let odom drive and use AMCL for drift correction.
            'alpha1': 0.05,   # rotation noise from rotation
            'alpha2': 0.05,   # rotation noise from translation
            'alpha3': 0.05,   # translation noise from translation
            'alpha4': 0.05,   # translation noise from rotation
            'alpha5': 0.05,   # extra translational noise

            # Frames
            'base_frame_id': 'base_link',
            'global_frame_id': 'map',
            'odom_frame_id': 'odom',

            # >>> [FIX #2] Laser model — tightened for walls-only SVG maps.
            # >>> Previous values (z_hit=0.5, z_rand=0.4) gave nearly equal
            # >>> weight to wall-matching and random noise.  Particles converged
            # >>> weakly with a hint but drifted immediately on motion.
            # >>> New values: 70% wall-matching weight, 20% random tolerance,
            # >>> tighter sigma (30cm), more beams (180) to compensate for
            # >>> sparse depth-camera coverage of the 360° merged scan.
            'laser_model_type': 'likelihood_field',
            'laser_max_range': 3.5,
            'laser_min_range': 0.15,
            'max_beams': 180,          # was 60 — need more samples from sparse depth-camera scan
            'do_beamskip': True,
            'beam_skip_distance': 0.5,
            'beam_skip_error_threshold': 0.9,
            'beam_skip_threshold': 0.3,
            'lambda_short': 0.1,
            'laser_likelihood_max_dist': 3.0,  # was 5.0 — tighter search window
            'sigma_hit': 0.3,          # was 0.5 — tighter wall-match tolerance (30cm)
            'z_hit': 0.7,              # was 0.5 — 70% weight on wall matching
            'z_max': 0.05,
            'z_rand': 0.2,             # was 0.4 — 20% random tolerance (was 40%)
            'z_short': 0.05,

            # Particle filter — generous count for robustness
            'max_particles': PythonExpression([
                "8000 if '", global_localization, "' == 'true' else 3000"
            ]),
            'min_particles': PythonExpression([
                "2000 if '", global_localization, "' == 'true' else 500"
            ]),

            # Resampling
            'pf_err': 0.05,
            'pf_z': 0.99,
            'resample_interval': 1,

            # Recovery — always allow random injection to escape bad convergence
            'recovery_alpha_slow': 0.001,
            'recovery_alpha_fast': 0.1,

            # Update thresholds — update frequently so pose stays fresh
            'update_min_a': 0.05,
            'update_min_d': 0.05,

            # Publish pose even when the robot is stationary
            'nomotion_update_period': 1.0,  # seconds

            # TF
            'tf_broadcast': True,
            'transform_tolerance': 1.0,
            'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
            'save_pose_rate': 0.5,

            # Topics
            'scan_topic': 'scan',
            'map_topic': 'map',

            # Initial pose
            'set_initial_pose': True,
            'initial_pose.x': LaunchConfiguration('initial_pose_x'),
            'initial_pose.y': LaunchConfiguration('initial_pose_y'),
            'initial_pose.z': 0.0,
            'initial_pose.yaw': LaunchConfiguration('initial_pose_yaw'),
        }],
        remappings=[
            ('odom', '/mobile_base/odom'),
        ],
    )

    amcl_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='amcl_lifecycle_manager',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['amcl'],
            'use_sim_time': use_sim_time,
            'bond_timeout': 10.0,
            'attempt_respawn_reconnection': True,
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
            'robot_radius': 0.30,  # was 0.27 — ALOHA is 24in (0.61m) wide, radius ~0.305m
            'occupancy_threshold': 95,
            'use_trajectory_optimization': True,
            'smoothing_weight': 0.8,
            'max_acceleration': 0.5,
            'enable_collision_avoidance': True,
            'safety_distance': 0.7,
            'emergency_stop_distance': 0.4,
            'enable_dynamic_obstacles': True,
            'dynamic_obstacle_timeout': 10.0,
            'depth_cloud_topic': '/cam_high/camera/depth/color/points',
            'use_sim_time': use_sim_time,
        }],
        remappings=[
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
            'robot_radius': 0.30,  # was 0.27 — match actual ALOHA radius
            'arrow_length': 0.6,
            'publish_rate': 10.0,
        }],
    )

    # ==================== Room Label Publisher ===========================

    room_label_node = Node(
        package='aloha',
        executable='room_label_publisher',
        name='room_label_publisher',
        output='screen',
        parameters=[{
            'rooms_json': rooms_json,
            'publish_rate': 0.5,
            'categories': ['conference_room', 'restroom', 'corridor',
                           'vertical_transport', 'shared_space', 'utility'],
        }],
        condition=IfCondition(PythonExpression([
            "str(len('", rooms_json, "')) > str(0)"
        ])),
    )

    # ==================== Localization Monitor ===========================

    localization_monitor_node = Node(
        package='aloha',
        executable='localization_monitor',
        name='localization_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'convergence_threshold': 0.5,
            'report_interval': 2.0,
        }],
    )

    # ==================== RViz ==========================================

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            aloha_pkg, 'rviz', 'amcl_localization.rviz',
        ])],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    # ==================== Launch Description =============================

    return LaunchDescription([
        # Arguments
        map_file_arg,
        rooms_json_arg,
        global_localization_arg,
        initial_pose_x_arg,
        initial_pose_y_arg,
        initial_pose_yaw_arg,
        use_rviz_arg,
        use_sim_time_arg,
        max_linear_velocity_arg,
        max_angular_velocity_arg,
        camera_serial_arg,
        launch_camera_arg,

        # TF
        base_footprint_to_base_link_tf,
        camera_transform,
        rear_camera_transform,

        # Cameras
        camera_front_node,
        camera_rear_node,

        # Sensor processing
        depthimage_to_laserscan_front,
        depthimage_to_laserscan_rear,
        laser_scan_merger_node,

        # Map
        map_server_node,
        map_lifecycle_node,

        # Localization
        amcl_node,
        amcl_lifecycle_node,
        localization_monitor_node,

        # Navigation
        nav_planner_node,
        robot_pose_marker_node,
        room_label_node,

        # Visualization
        rviz_node,
    ])
