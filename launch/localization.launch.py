#!/usr/bin/env python3

"""
Standalone Localization Launch File

Launches the localization stack (D405 camera + map server + AMCL) without any
navigation planner. The D405 depth camera is converted to a 2D laser scan, and
AMCL uses that scan + SLATE wheel odometry to localize on a previously saved map.

Prerequisites:
    # Terminal 1: Start the robot base (cameras optional - this launch starts D405)
    ros2 launch aloha aloha_bringup.launch.py use_cameras:=false

Usage:
    # Terminal 2: Localize using your map (global localization - start anywhere)
    ros2 launch aloha localization.launch.py \
        map_file:=/home/aloha/interbotix_ws/src/my_robot_maps/maps/my_final_map_moderate.yaml

    # If bringup already launched cameras, skip camera in localization:
    ros2 launch aloha localization.launch.py \
        map_file:=/path/to/map.yaml launch_camera:=false

    # Localize with a known starting position (faster convergence)
    ros2 launch aloha localization.launch.py \
        map_file:=/home/aloha/interbotix_ws/src/my_robot_maps/maps/my_final_map_moderate.yaml \
        global_localization:=false \
        initial_pose_x:=1.0 initial_pose_y:=2.0 initial_pose_yaw:=1.57

    # With localization monitor (prints convergence status to terminal)
    ros2 launch aloha localization.launch.py \
        map_file:=/path/to/map.yaml use_monitor:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from launch.conditions import IfCondition


def generate_launch_description():

    aloha_pkg = FindPackageShare('aloha')

    # ==================== Launch Arguments ====================

    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=PathJoinSubstitution([
            aloha_pkg, 'maps', 'my_map.yaml'
        ]),
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
        description='Launch RViz with localization display'
    )

    global_localization_arg = DeclareLaunchArgument(
        'global_localization',
        default_value='true',
        description='Robot figures out position from scratch (spreads particles across map). '
                    'Set false + provide initial_pose if you know starting location.'
    )

    initial_pose_x_arg = DeclareLaunchArgument(
        'initial_pose_x',
        default_value='0.0',
        description='Initial X position (only used when global_localization:=false)'
    )

    initial_pose_y_arg = DeclareLaunchArgument(
        'initial_pose_y',
        default_value='0.0',
        description='Initial Y position (only used when global_localization:=false)'
    )

    initial_pose_yaw_arg = DeclareLaunchArgument(
        'initial_pose_yaw',
        default_value='0.0',
        description='Initial yaw in radians (only used when global_localization:=false)'
    )

    use_monitor_arg = DeclareLaunchArgument(
        'use_monitor',
        default_value='false',
        description='Launch the localization monitor node (prints convergence info)'
    )

    camera_serial_arg = DeclareLaunchArgument(
        'camera_serial',
        default_value='218622272634',
        description='Serial number of cam_high RealSense D405'
    )

    launch_camera_arg = DeclareLaunchArgument(
        'launch_camera',
        default_value='true',
        description='Launch the RealSense camera node (set false if already running)'
    )

    # ==================== Launch Configurations ====================

    map_file = LaunchConfiguration('map_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    global_localization = LaunchConfiguration('global_localization')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    use_monitor = LaunchConfiguration('use_monitor')
    camera_serial = LaunchConfiguration('camera_serial')
    launch_camera = LaunchConfiguration('launch_camera')

    # ==================== TF Setup ====================

    # base_footprint -> base_link
    # SLATE publishes odom -> base_footprint; AMCL needs odom -> base_link
    base_footprint_to_base_link_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_to_base_link',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_footprint', 'base_link'],
    )

    # base_link -> camera_link (cam_high D405 mounting position)
    # yaw=1.5708 must match rtabmap_mapping.launch.py exactly, otherwise
    # the depth-to-laserscan data won't align with the map that was built.
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

    # ==================== RealSense D405 Camera ====================

    camera_node = Node(
        package='realsense2_camera',
        namespace='cam_high',
        name='camera',
        executable='realsense2_camera_node',
        output='screen',
        parameters=[{
            'initial_reset': True,
            'serial_no': ParameterValue(camera_serial, value_type=str),
            'enable_depth': True,
            'enable_color': True,
            'align_depth.enable': True,
            'depth_module.profile': '640,480,30',
            'rgb_camera.profile': '640,480,30',
            'enable_infra': False,
            'enable_infra1': False,
            'enable_infra2': False,
        }],
        condition=IfCondition(launch_camera),
    )

    # ==================== Map Server ====================

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_file,
            'use_sim_time': use_sim_time
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
            'use_sim_time': use_sim_time
        }],
    )

    # ==================== Depth to LaserScan ====================

    depthimage_to_laserscan_node = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
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
            ('scan', '/scan'),
        ],
    )

    # ==================== AMCL Localization ====================

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,

            # Motion model (differential drive)
            # Higher values = trust odometry less = particles spread more
            # = sensor corrections have more influence
            'alpha1': 0.3,
            'alpha2': 0.3,
            'alpha3': 0.3,
            'alpha4': 0.3,
            'alpha5': 0.3,

            # Frames
            'base_frame_id': 'base_link',
            'global_frame_id': 'map',
            'odom_frame_id': 'odom',

            # Laser model -- tuned for D405 depth-to-laserscan (scan_height=300)
            'laser_model_type': 'likelihood_field',
            'laser_max_range': 3.0,
            'laser_min_range': 0.1,
            'max_beams': 150,
            'beam_skip_distance': 0.5,
            'beam_skip_error_threshold': 0.9,
            'beam_skip_threshold': 0.3,
            'do_beamskip': True,
            'lambda_short': 0.1,
            'laser_likelihood_max_dist': 1.5,
            'sigma_hit': 0.2,
            'z_hit': 0.8,
            'z_max': 0.05,
            'z_rand': 0.1,
            'z_short': 0.05,

            # Particle filter
            'max_particles': PythonExpression([
                "5000 if '", global_localization, "' == 'true' else 3000"
            ]),
            'min_particles': PythonExpression([
                "1000 if '", global_localization, "' == 'true' else 500"
            ]),

            # Resampling
            'pf_err': 0.05,
            'pf_z': 0.99,
            'resample_interval': 1,

            # Recovery - enabled during global localization
            'recovery_alpha_slow': PythonExpression([
                "0.001 if '", global_localization, "' == 'true' else 0.0"
            ]),
            'recovery_alpha_fast': PythonExpression([
                "0.1 if '", global_localization, "' == 'true' else 0.0"
            ]),

            # Update more frequently so AMCL reacts faster
            'update_min_a': 0.1,
            'update_min_d': 0.1,

            # Transform
            'tf_broadcast': True,
            'transform_tolerance': 1.0,
            'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
            'save_pose_rate': 0.5,

            # Topics
            'scan_topic': 'scan',
            'map_topic': 'map',

            # Initial pose
            'set_initial_pose': True,
            'initial_pose.x': initial_pose_x,
            'initial_pose.y': initial_pose_y,
            'initial_pose.z': 0.0,
            'initial_pose.yaw': initial_pose_yaw,
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

    # ==================== Localization Monitor ====================

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
        condition=IfCondition(use_monitor),
    )

    # ==================== RViz ====================

    rviz_config_file = PathJoinSubstitution([
        aloha_pkg, 'rviz', 'amcl_localization.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    # ==================== Launch Description ====================

    return LaunchDescription([
        # Arguments
        map_file_arg,
        use_sim_time_arg,
        use_rviz_arg,
        global_localization_arg,
        initial_pose_x_arg,
        initial_pose_y_arg,
        initial_pose_yaw_arg,
        use_monitor_arg,
        camera_serial_arg,
        launch_camera_arg,

        # Camera
        camera_node,

        # TF
        base_footprint_to_base_link_tf,
        camera_transform,

        # Map
        map_server_node,
        map_lifecycle_node,

        # Sensor processing
        depthimage_to_laserscan_node,

        # AMCL
        amcl_node,
        amcl_lifecycle_node,

        # Monitor
        localization_monitor_node,

        # Visualization
        rviz_node,
    ])
