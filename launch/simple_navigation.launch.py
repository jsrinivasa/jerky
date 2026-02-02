#!/usr/bin/env python3

"""
Simple Navigation Launch File

Launches the simple navigation planner with map and localization.
Use this with your saved map to navigate using RViz2's "2D Goal Pose" tool.

Localization Options:
- use_amcl:=true (default): Uses AMCL with D435i depth camera converted to laser scan
  - global_localization:=true (default): Robot figures out its position automatically
    * Robot can start ANYWHERE on the map
    * Takes 30 seconds - 2 minutes to converge
    * Robot should rotate/move slowly to help convergence
  - global_localization:=false: Use if you know starting position
    * Set initial_pose_x, initial_pose_y, initial_pose_yaw
    * Or use RViz "2D Pose Estimate" tool
    * Faster convergence (seconds)
- use_rtabmap:=true: Uses RTAB-Map visual localization (requires pre-built database)
- Both false: Uses static transform (robot must start at origin)

Usage:
    # Global localization (robot anywhere on map):
    ros2 launch aloha simple_navigation.launch.py map_file:=<path_to_map.yaml>
    
    # Then drive/rotate robot slowly for 30-60 seconds to help it localize
    # Watch particle cloud in RViz converge to a single location
    
    # If you know starting position:
    ros2 launch aloha simple_navigation.launch.py \
        map_file:=<path> \
        global_localization:=false \
        initial_pose_x:=1.0 initial_pose_y:=2.0 initial_pose_yaw:=1.57
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
import os


def generate_launch_description():
    
    # Package paths
    aloha_pkg = FindPackageShare('aloha')
    
    # Launch arguments
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=PathJoinSubstitution([
            aloha_pkg,
            'maps',
            'my_map.yaml'
        ]),
        description='Path to the map YAML file'
    )
    
    use_nav2_arg = DeclareLaunchArgument(
        'use_nav2',
        default_value='false',
        description='Use Nav2 instead of simple planner (requires Nav2 to be running)'
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
    
    use_rtabmap_arg = DeclareLaunchArgument(
        'use_rtabmap',
        default_value='false',
        description='Use RTAB-Map for localization (requires camera). If false, assumes you have /odom from wheels.'
    )
    
    rtabmap_database_arg = DeclareLaunchArgument(
        'rtabmap_database',
        default_value='~/.ros/rtabmap.db',
        description='Path to RTAB-Map database'
    )
    
    use_amcl_arg = DeclareLaunchArgument(
        'use_amcl',
        default_value='true',
        description='Use AMCL for localization with depth camera converted to laser scan'
    )
    
    global_localization_arg = DeclareLaunchArgument(
        'global_localization',
        default_value='true',
        description='Enable global localization (robot figures out position from scratch). Set false if you know starting position.'
    )
    
    initial_pose_x_arg = DeclareLaunchArgument(
        'initial_pose_x',
        default_value='0.0',
        description='Initial X position estimate for AMCL (only used if global_localization:=false)'
    )
    
    initial_pose_y_arg = DeclareLaunchArgument(
        'initial_pose_y',
        default_value='0.0',
        description='Initial Y position estimate for AMCL (only used if global_localization:=false)'
    )
    
    initial_pose_yaw_arg = DeclareLaunchArgument(
        'initial_pose_yaw',
        default_value='0.0',
        description='Initial yaw (rotation) estimate for AMCL in radians (only used if global_localization:=false)'
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
    
    use_trajectory_optimization_arg = DeclareLaunchArgument(
        'use_trajectory_optimization',
        default_value='true',
        description='Enable trajectory smoothing and optimization'
    )
    
    smoothing_weight_arg = DeclareLaunchArgument(
        'smoothing_weight',
        default_value='0.5',
        description='Trajectory smoothing weight (0.1-2.0). Higher = smoother curves with more rounded corners'
    )
    
    max_acceleration_arg = DeclareLaunchArgument(
        'max_acceleration',
        default_value='0.5',
        description='Maximum linear acceleration (m/s^2)'
    )
    
    enable_collision_avoidance_arg = DeclareLaunchArgument(
        'enable_collision_avoidance',
        default_value='false',
        description='Enable real-time collision avoidance (for dynamic obstacles)'
    )
    
    safety_distance_arg = DeclareLaunchArgument(
        'safety_distance',
        default_value='0.5',
        description='Minimum safety distance to obstacles (meters)'
    )
    
    emergency_stop_distance_arg = DeclareLaunchArgument(
        'emergency_stop_distance',
        default_value='0.3',
        description='Emergency stop distance to obstacles (meters)'
    )
    
    # Launch configurations
    map_file = LaunchConfiguration('map_file')
    use_nav2 = LaunchConfiguration('use_nav2')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    use_rtabmap = LaunchConfiguration('use_rtabmap')
    rtabmap_database = LaunchConfiguration('rtabmap_database')
    use_amcl = LaunchConfiguration('use_amcl')
    global_localization = LaunchConfiguration('global_localization')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    lookahead_distance = LaunchConfiguration('lookahead_distance')
    max_linear_velocity = LaunchConfiguration('max_linear_velocity')
    max_angular_velocity = LaunchConfiguration('max_angular_velocity')
    goal_tolerance = LaunchConfiguration('goal_tolerance')
    robot_radius = LaunchConfiguration('robot_radius')
    use_trajectory_optimization = LaunchConfiguration('use_trajectory_optimization')
    smoothing_weight = LaunchConfiguration('smoothing_weight')
    max_acceleration = LaunchConfiguration('max_acceleration')
    enable_collision_avoidance = LaunchConfiguration('enable_collision_avoidance')
    safety_distance = LaunchConfiguration('safety_distance')
    emergency_stop_distance = LaunchConfiguration('emergency_stop_distance')
    
    # Static map -> odom transform for testing (only when NOT using RTAB-Map or AMCL)
    # This publishes a fixed transform assuming robot starts at origin
    static_condition = PythonExpression([
        '"', use_rtabmap, '" == "false" and "', use_amcl, '" == "false"'
    ])
    
    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        condition=IfCondition(static_condition)
    )
    
    # Note: We don't need a separate map_server because RTAB-Map publishes
    # the map directly from its database when in localization mode
    # If not using RTAB-Map, we use map_server to load the map from YAML
    
    # Map server (only when NOT using RTAB-Map)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_file,
            'use_sim_time': use_sim_time
        }],
        condition=UnlessCondition(use_rtabmap)
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
        }],
        condition=UnlessCondition(use_rtabmap)
    )
    
    # RTAB-Map in localization mode (uses the saved map)
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[{
            'database_path': rtabmap_database,
            'frame_id': 'base_link',  # Changed from base_footprint
            'odom_frame_id': 'odom',
            'subscribe_depth': True,
            'subscribe_rgb': True,
            'subscribe_scan': False,
            'approx_sync': True,
            'queue_size': 100,  # Increased for better sync tolerance
            'Mem/IncrementalMemory': 'false',  # Localization mode
            'Mem/InitWMWithAllNodes': 'true',  # Load all nodes
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('rgb/image', '/cam_high/camera/color/image_rect_raw'),
            ('rgb/camera_info', '/cam_high/camera/color/camera_info'),
            ('depth/image', '/cam_high/camera/depth/image_rect_raw'),
            ('odom', '/mobile_base/odom'),
        ],
        condition=IfCondition(use_rtabmap)
    )
    
    # RGB-D Odometry
    rgbd_odometry_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',  # Changed from base_footprint
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'approx_sync': True,
            'queue_size': 100,  # Increased for better sync tolerance
            'Odom/Strategy': '0',
            'Odom/ResetCountdown': '1',
            'Odom/GuessMotion': 'true',
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('rgb/image', '/cam_high/camera/color/image_rect_raw'),
            ('rgb/camera_info', '/cam_high/camera/color/camera_info'),
            ('depth/image', '/cam_high/camera/depth/image_rect_raw'),
        ],
        condition=IfCondition(use_rtabmap)
    )
    
    # ==================== AMCL Localization Setup ====================
    
    # Static transform: base_link -> camera_link (where cam_high is mounted)
    # Corrected with +90° yaw (clockwise) to fix rotation
    camera_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_base_link',
        arguments=['--x', '0.2098050', '--y', '0', '--z', '1.031778', 
                   '--yaw', '1.5708', '--pitch', '0.645772', '--roll', '0',  # +90 degrees
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link'],
        condition=IfCondition(use_amcl)
    )
    
    # Convert D435i depth image to a 2D laser scan for AMCL
    # Camera is angled down, so we use rows from upper part of image (sees farther)
    depthimage_to_laserscan_node = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
        output='screen',
        parameters=[{
            'scan_height': 50,  # Use more rows for better averaging (camera angled down)
            'scan_row_step': 1,  # Step between rows
            'scan_time': 0.033,  # Time between scans (33ms = 30Hz)
            'range_min': 0.45,  # Minimum range (meters) - D435i minimum depth
            'range_max': 4.0,   # Maximum range (meters) - reasonable for indoor navigation
            'output_frame': 'camera_depth_optical_frame',  # Frame ID for the scan
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('depth', '/cam_high/camera/depth/image_rect_raw'),
            ('depth_camera_info', '/cam_high/camera/depth/camera_info'),
            ('scan', '/scan'),
        ],
        condition=IfCondition(use_amcl)
    )
    
    # AMCL Localization Node
    # This uses the laser scan + wheel odometry to localize on the static map
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            
            # Motion model parameters
            'alpha1': 0.2,  # Rotation noise from rotation
            'alpha2': 0.2,  # Rotation noise from translation
            'alpha3': 0.2,  # Translation noise from translation
            'alpha4': 0.2,  # Translation noise from rotation
            'alpha5': 0.2,  # Strafe noise
            
            # Frame IDs
            'base_frame_id': 'base_link',  # Changed from base_footprint to match slate_base
            'global_frame_id': 'map',
            'odom_frame_id': 'odom',
            
            # Laser model configuration
            'laser_model_type': 'likelihood_field',
            'laser_max_range': 4.0,  # Match depth camera max range
            'laser_min_range': 0.45,  # Match depth camera min range
            'max_beams': 60,
            'beam_skip_distance': 0.5,
            'beam_skip_error_threshold': 0.9,
            'beam_skip_threshold': 0.3,
            'do_beamskip': False,
            'lambda_short': 0.1,
            'laser_likelihood_max_dist': 2.0,
            'sigma_hit': 0.2,
            'z_hit': 0.5,
            'z_max': 0.05,
            'z_rand': 0.5,
            'z_short': 0.05,
            
            # Particle filter - adjusted for global localization
            # These will be high for global localization, lower for pose tracking
            'max_particles': PythonExpression(["5000 if '", global_localization, "' == 'true' else 2000"]),
            'min_particles': PythonExpression(["1000 if '", global_localization, "' == 'true' else 500"]),
            
            # Resampling
            'pf_err': 0.05,
            'pf_z': 0.99,
            'resample_interval': 1,
            
            # Recovery parameters - important for global localization
            'recovery_alpha_slow': PythonExpression(["0.001 if '", global_localization, "' == 'true' else 0.0"]),
            'recovery_alpha_fast': PythonExpression(["0.1 if '", global_localization, "' == 'true' else 0.0"]),
            
            # Update thresholds
            'update_min_a': 0.2,  # Minimum angular movement before update (radians)
            'update_min_d': 0.25,  # Minimum linear movement before update (meters)
            
            # Transform settings
            'tf_broadcast': True,
            'transform_tolerance': 1.0,
            'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
            'save_pose_rate': 0.5,
            
            # Topics
            'scan_topic': 'scan',
            'map_topic': 'map',
            
            # Initial pose - always provide one, even for global localization
            # In global localization mode, high particle count will spread from this starting point
            'set_initial_pose': True,
            'initial_pose.x': initial_pose_x,
            'initial_pose.y': initial_pose_y,
            'initial_pose.z': 0.0,
            'initial_pose.yaw': initial_pose_yaw,
        }],
        remappings=[
            ('odom', '/mobile_base/odom'),  # Remap to SLATE base odometry
        ],
        condition=IfCondition(use_amcl)
    )
    
    # Lifecycle manager for AMCL
    amcl_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='amcl_lifecycle_manager',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['amcl'],
            'use_sim_time': use_sim_time
        }],
        condition=IfCondition(use_amcl)
    )
    
    # Simple Navigation Planner
    nav_planner_node = Node(
        package='aloha',
        executable='simple_nav_planner',
        name='simple_nav_planner',
        output='screen',
        parameters=[{
            'use_nav2': use_nav2,
            'lookahead_distance': lookahead_distance,
            'max_linear_velocity': max_linear_velocity,
            'max_angular_velocity': max_angular_velocity,
            'goal_tolerance': goal_tolerance,
            'robot_radius': robot_radius,
            'use_trajectory_optimization': use_trajectory_optimization,
            'smoothing_weight': smoothing_weight,
            'max_acceleration': max_acceleration,
            'enable_collision_avoidance': enable_collision_avoidance,
            'safety_distance': safety_distance,
            'emergency_stop_distance': emergency_stop_distance,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/odom', '/mobile_base/odom'),  # Remap to SLATE base odometry
            ('/cmd_vel', '/mobile_base/cmd_vel'),  # Send commands to mobile base
        ]
    )
    
    # RViz - Use AMCL-specific config by default
    rviz_config_file = PathJoinSubstitution([
        aloha_pkg,
        'rviz',
        'amcl_localization.rviz'
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
        use_nav2_arg,
        use_sim_time_arg,
        use_rviz_arg,
        use_rtabmap_arg,
        rtabmap_database_arg,
        use_amcl_arg,
        global_localization_arg,
        initial_pose_x_arg,
        initial_pose_y_arg,
        initial_pose_yaw_arg,
        lookahead_distance_arg,
        max_linear_velocity_arg,
        max_angular_velocity_arg,
        goal_tolerance_arg,
        robot_radius_arg,
        use_trajectory_optimization_arg,
        smoothing_weight_arg,
        max_acceleration_arg,
        enable_collision_avoidance_arg,
        safety_distance_arg,
        emergency_stop_distance_arg,
        
        # Nodes
        map_to_odom_tf,
        map_server_node,
        map_lifecycle_node,
        rgbd_odometry_node,
        rtabmap_node,
        camera_transform,
        depthimage_to_laserscan_node,
        amcl_node,
        amcl_lifecycle_node,
        nav_planner_node,
        rviz_node,
    ])

