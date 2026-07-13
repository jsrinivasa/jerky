#!/usr/bin/env python3
"""
Simple Navigation Path Planner for Mobile Base

This node provides a simple interface to navigate the mobile base using:
- Goal poses set in RViz2 (using "2D Goal Pose" tool)
- A* path planning on the saved map
- Pure pursuit path following controller

Author: ALOHA Team
License: MIT
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav2_msgs.action import NavigateToPose
import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped

import numpy as np
import math
from typing import List, Tuple, Optional
from enum import Enum
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize


class NavigationState(Enum):
    """Navigation state machine"""
    IDLE = 0
    PLANNING = 1
    FOLLOWING = 2
    GOAL_REACHED = 3
    FAILED = 4


class SimpleNavPlanner(Node):
    """
    Simple navigation planner for mobile base.
    
    Subscribes to:
        - /goal_pose: Goal pose from RViz2
        - /map: Occupancy grid map
        - /odom: Robot odometry
    
    Publishes:
        - /cmd_vel: Velocity commands
        - /planned_path: Visualized path
    
    Features:
        - Simple A* path planning
        - Pure pursuit path following
        - Obstacle avoidance
    """
    
    def __init__(self):
        super().__init__('simple_nav_planner')
        
        # Parameters
        self.declare_parameter('use_nav2', False)
        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('max_linear_velocity', 0.3)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('path_resolution', 0.1)
        self.declare_parameter('robot_radius', 0.3)  # Robot radius in meters (24in = 0.61m width, radius = 0.305m + safety margin)
        self.declare_parameter('use_trajectory_optimization', True)
        self.declare_parameter('smoothing_weight', 0.5)  # Higher = smoother but less accurate
        self.declare_parameter('max_acceleration', 0.5)  # m/s^2
        self.declare_parameter('enable_collision_avoidance', True)
        self.declare_parameter('safety_distance', 0.5)  # Minimum distance to obstacles (meters)
        self.declare_parameter('emergency_stop_distance', 0.3)  # Emergency stop distance (meters)
        self.declare_parameter('occupancy_threshold', 60)  # Cells with occupancy >= this are obstacles (0-100)
        self.declare_parameter('live_scan_topic', '/scan')
        self.declare_parameter('live_scan_max_age', 1.0)  # seconds; older scans are ignored (sensor stalled)

        self.use_nav2 = self.get_parameter('use_nav2').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_resolution = self.get_parameter('path_resolution').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.use_traj_opt = self.get_parameter('use_trajectory_optimization').value
        self.smoothing_weight = self.get_parameter('smoothing_weight').value
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.enable_collision_avoidance = self.get_parameter('enable_collision_avoidance').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.emergency_stop_distance = self.get_parameter('emergency_stop_distance').value
        self.occupancy_threshold = self.get_parameter('occupancy_threshold').value
        self.live_scan_topic = self.get_parameter('live_scan_topic').value
        self.live_scan_max_age = self.get_parameter('live_scan_max_age').value

        # State
        self.state = NavigationState.IDLE
        self.current_pose = None
        self.goal_pose = None
        self.current_path = None
        self.raw_path = None  # Store raw A* path (for visualization)
        self.optimized_trajectory = None  # Store trajectory with time stamps
        self.map_data = None
        self.inflated_map = None  # Pre-computed obstacle-inflated map for fast planning
        self.obstacle_map_raw = None  # Denoised but NOT inflated — for collision ray-casting
        self.latest_scan = None  # Live LaserScan (e.g. from cam_high depth) for real-time obstacles
        self.path_index = 0
        self.collision_detected = False
        self.last_collision_check_time = self.get_clock().now()
        self.emergency_stop_time = None
        self.replan_attempts = 0
        self._prev_angular_z = 0.0
        self._prev_linear_x = 0.0

        # Temporal collision filtering: require N consecutive detections
        self._collision_hit_streak = 0
        self._collision_clear_streak = 0
        self._COLLISION_CONFIRM_TICKS = 3   # ticks before acting on obstacle
        self._COLLISION_CLEAR_TICKS = 3     # ticks before declaring "all clear"

        # Obstacle recovery state machine: wait → replan → rotate → replan …
        self._obstacle_phase = 'wait'   # 'wait', 'rotate', 'replan'
        self._rotate_start_time = None
        self._rotate_vel = 0.0
        self._replan_cooldown_until = None
        
        # TF for transforming odom-frame poses into the map frame
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # Map subscription with TRANSIENT_LOCAL QoS to match map_server
        map_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Live obstacle scan (best-effort to match depthimage_to_laserscan's
        # sensor-data QoS; a reliable subscriber wouldn't receive it at all).
        scan_qos = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.live_scan_topic,
            self.scan_callback,
            scan_qos
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        path_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.path_pub = self.create_publisher(Path, '/planned_path', path_qos)
        self.raw_path_pub = self.create_publisher(Path, '/raw_path', path_qos)
        self.smoothed_path_pub = self.create_publisher(Path, '/smoothed_path', path_qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/nav_markers', 10)
        self.obstacle_marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        inflated_map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.inflated_map_pub = self.create_publisher(
            OccupancyGrid, '/inflated_map', inflated_map_qos
        )
        
        # Create timer to continuously publish path visualization
        self.viz_timer = self.create_timer(0.5, self.continuous_path_visualization)
        
        # Nav2 Action Client (optional)
        if self.use_nav2:
            self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
            self.get_logger().info('Using Nav2 for navigation')
        else:
            self.get_logger().info('Using simple path planner')
        
        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Simple Navigation Planner initialized')
        self.get_logger().info(f'Robot radius: {self.robot_radius}m (24 inches wide)')
        self.get_logger().info(f'Path planning: Using robot radius of {self.robot_radius}m for obstacle inflation')
        self.get_logger().info(f'Dynamic collision avoidance: {"Enabled" if self.enable_collision_avoidance else "Disabled"}')
        self.get_logger().info('Set a goal pose in RViz2 using "2D Goal Pose" tool')
    
    def goal_callback(self, msg: PoseStamped):
        """Handle new goal pose from RViz2."""
        # Reject duplicate goals while actively following a path
        if (self.state == NavigationState.FOLLOWING
                and self.goal_pose is not None
                and self.current_path is not None):
            dx = msg.pose.position.x - self.goal_pose.pose.position.x
            dy = msg.pose.position.y - self.goal_pose.pose.position.y
            if math.sqrt(dx * dx + dy * dy) < 0.5:
                return

        self.goal_pose = msg
        self.get_logger().info(
            f'New goal received: x={msg.pose.position.x:.2f}, '
            f'y={msg.pose.position.y:.2f}'
        )
        
        if self.use_nav2:
            self.navigate_with_nav2(msg)
        else:
            self.plan_and_navigate()
    
    def map_callback(self, msg: OccupancyGrid):
        """Store map data for path planning and create inflated map."""
        if self.map_data is None:
            self.get_logger().info('Map received! Creating inflated obstacle map...')
        self.map_data = msg
        self.create_inflated_map()
    
    def odom_callback(self, msg: Odometry):
        """Update current robot pose in the map frame via TF."""
        try:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = msg.header.frame_id
            # Use Time(0) = latest available transform, avoids ExtrapolationException
            # when odom and TF timestamps don't align exactly.
            pose_stamped.header.stamp = rclpy.time.Time().to_msg()
            pose_stamped.pose = msg.pose.pose
            transformed = self._tf_buffer.transform(
                pose_stamped, 'map', timeout=Duration(seconds=0.2)
            )
            self.current_pose = transformed.pose
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF odom→map failed ({type(e).__name__}), keeping last good pose',
                throttle_duration_sec=5.0,
            )

    def scan_callback(self, msg: LaserScan):
        """Store the latest live obstacle scan (see check_collision_ahead)."""
        self.latest_scan = msg

    def navigate_with_nav2(self, goal: PoseStamped):
        """Use Nav2 for navigation (requires Nav2 to be running)."""
        if not self.nav2_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available!')
            return
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal
        
        self.get_logger().info('Sending goal to Nav2...')
        self.nav2_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav2_feedback_callback
        )
        self.state = NavigationState.FOLLOWING
    
    def nav2_feedback_callback(self, feedback_msg):
        """Handle Nav2 feedback."""
        feedback = feedback_msg.feedback
        distance = feedback.distance_remaining
        self.get_logger().info(f'Distance to goal: {distance:.2f}m', throttle_duration_sec=1.0)
    
    def plan_and_navigate(self):
        """Plan path and start navigation using simple planner."""
        if self.current_pose is None:
            self.get_logger().warn(
                'Cannot plan: no map-frame pose yet (waiting for odom→map TF). '
                'Has the robot localized?'
            )
            return
        
        if self.map_data is None:
            self.get_logger().warn('No map data yet!')
            return
        
        # Plan path
        self.state = NavigationState.PLANNING
        self.get_logger().info('Planning path...')
        
        path = self.plan_path(
            self.current_pose.position,
            self.goal_pose.pose.position
        )
        
        if path is None or len(path) < 2:
            self.get_logger().error(
                'Path planning failed! Check the messages above for detailed reasons. '
            )
            self.state = NavigationState.FAILED
            return
        
        self.raw_path = path
        
        # Apply trajectory optimization if enabled
        if self.use_traj_opt:
            self.get_logger().info('Optimizing trajectory...')
            smoothed_path = self.smooth_trajectory(path)
            if smoothed_path is not None and len(smoothed_path) > 0:
                self.current_path = smoothed_path
                self.get_logger().info(f'Trajectory optimized: {len(path)} -> {len(smoothed_path)} waypoints')
            else:
                self.get_logger().warn('Trajectory optimization failed, using raw path')
                self.current_path = path
        else:
            self.current_path = path
        
        self.path_index = 0
        self.state = NavigationState.FOLLOWING
        
        self.get_logger().info(f'Path planned with {len(self.current_path)} waypoints')
        self.publish_path_visualization()
    
    def plan_path(self, start: Point, goal: Point) -> Optional[List[Point]]:
        """
        Simple A* path planner.
        
        Args:
            start: Start position
            goal: Goal position
        
        Returns:
            List of waypoints or None if planning failed
        """
        if self.map_data is None:
            self.get_logger().error('Path planning failed: No map data available')
            return None
        
        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start.x, start.y)
        goal_grid = self.world_to_grid(goal.x, goal.y)
        
        if start_grid is None:
            self.get_logger().error(
                f'Path planning failed: Start position ({start.x:.2f}, {start.y:.2f}) is outside map bounds. '
                f'Map origin: ({self.map_data.info.origin.position.x:.2f}, {self.map_data.info.origin.position.y:.2f}), '
                f'Map size: {self.map_data.info.width}x{self.map_data.info.height} cells, '
                f'Resolution: {self.map_data.info.resolution:.3f}m/cell'
            )
            return None
        
        if goal_grid is None:
            self.get_logger().error(
                f'Path planning failed: Goal position ({goal.x:.2f}, {goal.y:.2f}) is outside map bounds. '
                f'Map origin: ({self.map_data.info.origin.position.x:.2f}, {self.map_data.info.origin.position.y:.2f}), '
                f'Map size: {self.map_data.info.width}x{self.map_data.info.height} cells, '
                f'Resolution: {self.map_data.info.resolution:.3f}m/cell'
            )
            return None
        
        resolution = self.map_data.info.resolution
        check_radius_cells = int(self.robot_radius / resolution)
        
        self.get_logger().info(
            f'Planning path from grid ({start_grid[0]}, {start_grid[1]}) to ({goal_grid[0]}, {goal_grid[1]}). '
            f'Inflating obstacles by {check_radius_cells} cells ({self.robot_radius:.2f}m)'
        )
        
        # Simple A* implementation
        path_grid = self.astar(start_grid, goal_grid)
        
        if path_grid is None:
            self.get_logger().error(
                f'Path planning failed: A* algorithm could not find a valid path from '
                f'({start.x:.2f}, {start.y:.2f}) to ({goal.x:.2f}, {goal.y:.2f}). '
                f'The goal may be unreachable due to obstacles blocking all paths.'
            )
            return None
        
        # Convert grid path back to world coordinates
        path_world = []
        for grid_point in path_grid:
            world_point = self.grid_to_world(grid_point[0], grid_point[1])
            if world_point:
                p = Point()
                p.x = world_point[0]
                p.y = world_point[1]
                p.z = 0.0
                path_world.append(p)
        
        self.get_logger().info(f'A* path planning succeeded with {len(path_world)} raw waypoints')
        
        # DON'T simplify - return the full A* path so we can see the jagged edges
        # Smoothing will handle making it nice
        return path_world
    
    def astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* path planning algorithm.
        
        Args:
            start: Start grid cell (x, y)
            goal: Goal grid cell (x, y)
        
        Returns:
            List of grid cells forming path or None
        """
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        # Check if start is valid; snap to nearest free cell if blocked
        if not self.is_point_collision_free_grid(start[0], start[1]):
            alt = self.find_nearest_free_cell(start[0], start[1], max_radius=30)
            if alt is None:
                self.get_logger().error(
                    f'A* failed: Start cell ({start[0]}, {start[1]}) is occupied and no '
                    f'free cell found within search radius.'
                )
                return None
            self.get_logger().warn(
                f'Start cell ({start[0]}, {start[1]}) is in inflated obstacle zone. '
                f'Snapping to nearest free cell ({alt[0]}, {alt[1]}).'
            )
            start = alt
        
        # Check if goal is valid
        if not self.is_point_collision_free_grid(goal[0], goal[1]):
            self.get_logger().warn(
                f'Goal cell ({goal[0]}, {goal[1]}) is occupied or too close to obstacles. '
                f'Searching for nearest free cell with {self.robot_radius:.2f}m clearance...'
            )
            # Try to find nearest free cell
            original_goal = goal
            goal = self.find_nearest_free_cell(goal[0], goal[1])
            if goal is None:
                self.get_logger().error(
                    f'A* failed: Goal is in occupied space at ({original_goal[0]}, {original_goal[1]}) '
                    f'and no free cells found within search radius. The goal may be completely surrounded by obstacles.'
                )
                return None
            else:
                self.get_logger().info(
                    f'Adjusted goal from ({original_goal[0]}, {original_goal[1]}) to nearest free cell ({goal[0]}, {goal[1]})'
                )
        
        # A* data structures
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        max_iterations = 100000
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            
            # Get node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                self.get_logger().info(f'A* succeeded after {iteration} iterations')
                return path
            
            open_set.remove(current)
            
            # Check neighbors (8-connected)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                
                # Check if free considering robot radius (inflate obstacles)
                if not self.is_point_collision_free_grid(neighbor[0], neighbor[1]):
                    continue
                
                # Calculate cost
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    open_set.add(neighbor)
        
        if iteration >= max_iterations:
            self.get_logger().error(
                f'A* failed: Maximum iterations ({max_iterations}) reached. '
                f'Start: ({start[0]}, {start[1]}), Goal: ({goal[0]}, {goal[1]}). '
                f'Path may be extremely long or computationally expensive.'
            )
        else:
            self.get_logger().error(
                f'A* failed: Open set exhausted after {iteration} iterations. '
                f'No path exists between start ({start[0]}, {start[1]}) and goal ({goal[0]}, {goal[1]}). '
                f'All possible routes are blocked by obstacles.'
            )
        
        return None
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def is_free(self, x: int, y: int) -> bool:
        """Check if a grid cell is free."""
        if self.map_data is None:
            return False
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        
        idx = y * width + x
        if idx >= len(self.map_data.data):
            return False
        
        # Cells below occupancy_threshold are passable (free or light annotations)
        # Unknown cells (-1) are also treated as passable
        value = self.map_data.data[idx]
        return value < self.occupancy_threshold
    
    def create_inflated_map(self):
        """
        Pre-compute an inflated obstacle map for fast path planning.

        Pipeline:
          1. Threshold occupancy grid to get binary obstacle mask
          2. Morphological opening to remove isolated noise pixels
          3. Inflate remaining obstacles by robot radius
        """
        if self.map_data is None:
            return

        import cv2
        start_time = time.time()

        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        check_radius_cells = max(int(self.robot_radius / resolution), 1)

        data = np.array(self.map_data.data, dtype=np.int8).reshape((height, width))

        obstacles_raw = (data >= self.occupancy_threshold).astype(np.uint8)
        raw_count = int(np.sum(obstacles_raw))

        # Stage 1: light opening removes isolated 1-2 pixel noise (e.g. arm
        # artifacts baked into the map) while preserving real thin obstacles.
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        obstacles_opened = cv2.morphologyEx(obstacles_raw, cv2.MORPH_OPEN, open_kernel)

        # Stage 2: light closing reconnects hairline gaps in walls.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        obstacles_clean = cv2.morphologyEx(obstacles_opened, cv2.MORPH_CLOSE, close_kernel)
        clean_count = int(np.sum(obstacles_clean))

        inflate_diameter = 2 * check_radius_cells + 1
        inflate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (inflate_diameter, inflate_diameter)
        )
        inflated = cv2.dilate(obstacles_clean, inflate_kernel, iterations=1)

        self.inflated_map = inflated.astype(bool)
        self.obstacle_map_raw = obstacles_clean.astype(bool)

        elapsed = time.time() - start_time
        free_cells = int(np.sum(~self.inflated_map))
        self.get_logger().info(
            f'Inflated map created in {elapsed:.2f}s. '
            f'Obstacles: {raw_count} raw → {clean_count} after denoise. '
            f'Inflation: {check_radius_cells} cells ({self.robot_radius:.2f}m). '
            f'Free cells: {free_cells}/{width * height}'
        )
        
        # Publish inflated map for visualization in RViz
        self.publish_inflated_map()
    
    def publish_inflated_map(self):
        """Publish the inflated obstacle map as an OccupancyGrid for RViz visualization."""
        if self.inflated_map is None or self.map_data is None:
            return
        
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info = self.map_data.info
        
        # Convert: True (occupied) -> 100, False (free) -> 0
        flat = self.inflated_map.flatten().astype(np.int8)
        flat[flat == 1] = 100
        msg.data = flat.tolist()
        
        self.inflated_map_pub.publish(msg)
        self.get_logger().info('Published inflated map on /inflated_map', throttle_duration_sec=10.0)
    
    def is_point_collision_free_grid(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if a grid cell is collision-free considering robot radius.
        Uses pre-computed inflated map for fast lookup.
        
        Args:
            grid_x, grid_y: Grid coordinates
        
        Returns:
            True if cell is collision-free with robot radius
        """
        if self.inflated_map is None:
            return False
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            return False
        
        # Simple lookup in pre-computed inflated map
        return not self.inflated_map[grid_y, grid_x]
    
    def find_nearest_free_cell(self, x: int, y: int, max_radius: int = 20) -> Optional[Tuple[int, int]]:
        """Find nearest free cell to given position with robot radius clearance."""
        for radius in range(1, max_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        nx, ny = x + dx, y + dy
                        if self.is_point_collision_free_grid(nx, ny):
                            self.get_logger().info(
                                f'Found free cell at ({nx}, {ny}), distance {radius} cells from ({x}, {y})'
                            )
                            return (nx, ny)
        
        self.get_logger().warn(
            f'No free cell found within {max_radius} cells of ({x}, {y})'
        )
        return None
    
    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to grid coordinates."""
        if self.map_data is None:
            return None
        
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution
        
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)
        
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Optional[Tuple[float, float]]:
        """Convert grid coordinates to world coordinates."""
        if self.map_data is None:
            return None
        
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution
        
        world_x = origin_x + (grid_x + 0.5) * resolution
        world_y = origin_y + (grid_y + 0.5) * resolution
        
        return (world_x, world_y)
    
    def simplify_path(self, path: List[Point]) -> List[Point]:
        """Simplify path by removing unnecessary waypoints."""
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            # Keep waypoint if it changes direction significantly
            prev = path[i - 1]
            curr = path[i]
            next_pt = path[i + 1]
            
            dx1 = curr.x - prev.x
            dy1 = curr.y - prev.y
            dx2 = next_pt.x - curr.x
            dy2 = next_pt.y - curr.y
            
            # Calculate angle change
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            angle_diff = abs(angle2 - angle1)
            
            # Keep point if angle changes more than 15 degrees
            if angle_diff > 0.26:  # ~15 degrees
                simplified.append(curr)
        
        simplified.append(path[-1])
        return simplified
    
    def smooth_trajectory(self, path: List[Point]) -> Optional[List[Point]]:
        """
        Smooth trajectory using cubic B-spline interpolation with rounded corners.
        
        Args:
            path: Raw path from A*
        
        Returns:
            Smoothed path with rounded corners and many waypoints
        """
        if len(path) < 3:
            return path
        
        try:
            # Extract coordinates
            x_coords = [p.x for p in path]
            y_coords = [p.y for p in path]
            
            # Create B-spline representation
            # s parameter controls smoothness - HIGHER = SMOOTHER with more rounded corners
            # For truly smooth curves, we need s to be larger
            # Default smoothing_weight is 0.5, but we want MORE smoothing
            smoothing_factor = max(self.smoothing_weight * 2.0, 1.0)  # At least 1.0 for smooth curves
            
            # k is the degree of the spline (3 = cubic for smooth curves)
            k = min(3, len(path) - 1)
            
            self.get_logger().info(
                f'Smoothing path with {len(path)} waypoints, smoothing_factor={smoothing_factor:.2f}',
                throttle_duration_sec=2.0
            )
            
            tck, u = splprep([x_coords, y_coords], s=smoothing_factor, k=k)
            
            # Evaluate spline at MANY points for truly smooth visualization
            # More points = smoother appearance
            num_points = max(len(path) * 10, 100)  # At least 100 points for smooth curves
            u_new = np.linspace(0, 1, num_points)
            smoothed_coords = splev(u_new, tck)
            
            # Convert back to Point list and check for collisions using
            # the inflated map (same source of truth as A*).
            smoothed_path = []
            collision_count = 0
            consecutive_collisions = 0
            max_consecutive_collisions = 0

            for x, y in zip(smoothed_coords[0], smoothed_coords[1]):
                if not self._inflated_occupied(float(x), float(y)):
                    p = Point()
                    p.x = float(x)
                    p.y = float(y)
                    p.z = 0.0
                    smoothed_path.append(p)
                    consecutive_collisions = 0
                else:
                    collision_count += 1
                    consecutive_collisions += 1
                    max_consecutive_collisions = max(
                        max_consecutive_collisions, consecutive_collisions
                    )

            # If the spline cuts through a wall (3+ consecutive obstacle points),
            # removing those points leaves a "gap" — the lookahead then jumps
            # across the wall and the robot drives through it.  Fall back to the
            # raw A* path which is guaranteed collision-free.
            if max_consecutive_collisions >= 3:
                self.get_logger().warn(
                    f'Smoothed path cuts through an obstacle '
                    f'({max_consecutive_collisions} consecutive blocked points). '
                    'Falling back to raw A* path.'
                )
                return path

            collision_percentage = (collision_count / num_points) * 100
            if collision_percentage > 2.0:
                self.get_logger().warn(
                    f'Smoothed path has {collision_percentage:.1f}% collision points. '
                    f'Retrying with reduced smoothing (s={smoothing_factor/4:.2f})...'
                )
                reduced_smoothing = max(smoothing_factor / 4, 0.1)
                tck, u = splprep([x_coords, y_coords], s=reduced_smoothing, k=k)
                smoothed_coords = splev(u_new, tck)

                smoothed_path = []
                consecutive_collisions = 0
                max_consecutive_collisions = 0
                for x, y in zip(smoothed_coords[0], smoothed_coords[1]):
                    if not self._inflated_occupied(float(x), float(y)):
                        p = Point()
                        p.x = float(x)
                        p.y = float(y)
                        p.z = 0.0
                        smoothed_path.append(p)
                        consecutive_collisions = 0
                    else:
                        consecutive_collisions += 1
                        max_consecutive_collisions = max(
                            max_consecutive_collisions, consecutive_collisions
                        )

                if max_consecutive_collisions >= 3 or len(smoothed_path) < 2:
                    self.get_logger().warn(
                        'Reduced smoothing still cuts through obstacles. '
                        'Using raw A* path.'
                    )
                    return path

                self.get_logger().info('Retry successful with reduced smoothing.')

            if len(smoothed_path) < 2:
                self.get_logger().warn('Smoothed path too short after collision filtering, using raw path')
                return path
            
            self.get_logger().info(
                f'Trajectory smoothed: {len(path)} -> {len(smoothed_path)} waypoints with rounded corners',
                throttle_duration_sec=2.0
            )
            
            # Optimize velocity profile
            self.optimized_trajectory = self.optimize_velocity_profile(smoothed_path)
            
            return smoothed_path
            
        except Exception as e:
            self.get_logger().error(f'Trajectory smoothing failed: {e}')
            return path
    
    def is_point_collision_free(self, x: float, y: float) -> bool:
        """
        Check if a point is collision-free considering robot radius.
        
        Args:
            x, y: World coordinates
        
        Returns:
            True if point is collision-free
        """
        if self.map_data is None:
            return True
        
        grid_pos = self.world_to_grid(x, y)
        if grid_pos is None:
            return False
        
        resolution = self.map_data.info.resolution
        check_radius_cells = int(self.robot_radius / resolution)
        
        # Check circular area around point
        for dx in range(-check_radius_cells, check_radius_cells + 1):
            for dy in range(-check_radius_cells, check_radius_cells + 1):
                if dx*dx + dy*dy <= check_radius_cells*check_radius_cells:
                    if not self.is_free(grid_pos[0] + dx, grid_pos[1] + dy):
                        return False
        
        return True
    
    def optimize_velocity_profile(self, path: List[Point]) -> List[Tuple[Point, float, float]]:
        """
        Optimize velocity profile for time-optimal trajectory.
        
        Args:
            path: Smoothed path
        
        Returns:
            List of (point, velocity, time) tuples
        """
        if len(path) < 2:
            return [(path[0], 0.0, 0.0)]
        
        trajectory = []
        current_time = 0.0
        current_vel = 0.0
        
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            # Calculate segment length
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            segment_length = math.sqrt(dx*dx + dy*dy)
            
            if segment_length < 0.001:
                continue
            
            # Calculate curvature (rate of direction change)
            if i > 0:
                p0 = path[i - 1]
                # Approximate curvature using three points
                dx1 = p1.x - p0.x
                dy1 = p1.y - p0.y
                dx2 = p2.x - p1.x
                dy2 = p2.y - p1.y
                
                angle1 = math.atan2(dy1, dx1)
                angle2 = math.atan2(dy2, dx2)
                angle_diff = abs(self.normalize_angle(angle2 - angle1))
                
                # Reduce velocity in curves
                if segment_length > 0:
                    curvature = angle_diff / segment_length
                    max_vel_curve = min(self.max_linear_vel, math.sqrt(self.max_acceleration / max(curvature, 0.01)))
                else:
                    max_vel_curve = self.max_linear_vel
            else:
                max_vel_curve = self.max_linear_vel
            
            # Calculate target velocity considering acceleration limits
            target_vel = max_vel_curve
            
            # Accelerate or decelerate based on current velocity
            if target_vel > current_vel:
                # Accelerate
                delta_v = min(target_vel - current_vel, self.max_acceleration * 0.1)  # dt = 0.1s
                current_vel += delta_v
            else:
                # Decelerate
                delta_v = min(current_vel - target_vel, self.max_acceleration * 0.1)
                current_vel -= delta_v
            
            # Calculate time for this segment
            avg_vel = max(current_vel, 0.01)  # Avoid division by zero
            segment_time = segment_length / avg_vel
            
            trajectory.append((p1, current_vel, current_time))
            current_time += segment_time
        
        # Add final point
        trajectory.append((path[-1], 0.0, current_time))
        
        return trajectory
    
    def control_loop(self):
        """Main control loop for path following."""
        if self.state != NavigationState.FOLLOWING:
            return
        
        if self.current_pose is None or self.current_path is None:
            return
        
        # Check if we reached the goal
        if self.reached_goal():
            self.stop_robot()
            self.state = NavigationState.GOAL_REACHED
            self.get_logger().info('Goal reached!')
            return
        
        # ----- Collision avoidance with wait → replan → rotate recovery -----
        collision_info = None
        if self.enable_collision_avoidance:
            now = self.get_clock().now()

            # After a successful replan, give the robot 2 s to start moving on
            # the new path before re-checking for collisions.
            if self._replan_cooldown_until is not None:
                if now < self._replan_cooldown_until:
                    cmd_vel = self.pure_pursuit_control()
                    self.cmd_vel_pub.publish(cmd_vel)
                    return
                self._replan_cooldown_until = None

            collision_info = self.check_collision_ahead()

            # --- Temporal filtering: avoid acting on single-frame noise ---
            if collision_info['emergency_stop']:
                self._collision_hit_streak += 1
                self._collision_clear_streak = 0
            else:
                self._collision_clear_streak += 1
                if self._collision_clear_streak >= self._COLLISION_CLEAR_TICKS:
                    self._collision_hit_streak = 0

            confirmed_obstacle = (
                self._collision_hit_streak >= self._COLLISION_CONFIRM_TICKS
            )

            if confirmed_obstacle:
                self.collision_detected = True
                self.visualize_obstacles(collision_info)

                # --- First confirmed detection: enter WAIT phase ---
                if self.emergency_stop_time is None:
                    self.emergency_stop_time = now
                    self._obstacle_phase = 'wait'
                    self.stop_robot(immediate=True)
                    self.get_logger().warn(
                        f'Obstacle confirmed at {collision_info["distance"]:.2f}m — '
                        f'waiting 3 s for it to clear...'
                    )
                    return

                time_stopped = (now - self.emergency_stop_time).nanoseconds / 1e9

                # --- WAIT phase: sit still, hope obstacle moves ---
                if self._obstacle_phase == 'wait':
                    self.stop_robot(immediate=True)
                    if time_stopped >= 3.0:
                        self._obstacle_phase = 'replan'
                    return

                # --- ROTATE phase: spin in place to try a different angle ---
                if self._obstacle_phase == 'rotate':
                    rotate_elapsed = (
                        (now - self._rotate_start_time).nanoseconds / 1e9
                    )
                    if rotate_elapsed < 2.0:
                        cmd = Twist()
                        cmd.angular.z = self._rotate_vel
                        self.cmd_vel_pub.publish(cmd)
                    else:
                        self.stop_robot(immediate=True)
                        self._obstacle_phase = 'replan'
                    return

                # --- REPLAN phase: compute a new A* path around obstacle ---
                if self._obstacle_phase == 'replan':
                    self.stop_robot(immediate=True)
                    self.replan_attempts += 1
                    if self.replan_attempts <= 5:
                        self.get_logger().info(
                            f'Replan attempt {self.replan_attempts}/5...'
                        )
                        if self.attempt_replan_around_obstacle():
                            self.get_logger().info(
                                'Replan succeeded — resuming navigation'
                            )
                            self.emergency_stop_time = None
                            self.replan_attempts = 0
                            self._obstacle_phase = 'wait'
                            self._collision_hit_streak = 0
                            self._replan_cooldown_until = (
                                now + Duration(seconds=2)
                            )
                            return
                        self._obstacle_phase = 'rotate'
                        self._rotate_start_time = now
                        direction = (
                            1.0 if self.replan_attempts % 2 == 0 else -1.0
                        )
                        self._rotate_vel = direction * 0.5
                        self.get_logger().info(
                            f'Replan failed — rotating '
                            f'{"CCW" if direction > 0 else "CW"} to try from '
                            f'a different angle'
                        )
                    else:
                        self.get_logger().error(
                            'All 5 replan attempts exhausted — stopping'
                        )
                        self.state = NavigationState.FAILED
                    return

            elif collision_info['reduce_speed']:
                self.collision_detected = True
                self.visualize_obstacles(collision_info)
            elif self._collision_clear_streak >= self._COLLISION_CLEAR_TICKS:
                self.collision_detected = False
                self._reset_obstacle_state()

        # Pure pursuit control
        cmd_vel = self.pure_pursuit_control()

        if (self.enable_collision_avoidance
                and self.collision_detected
                and collision_info is not None):
            cmd_vel = self.adjust_velocity_for_obstacles(cmd_vel, collision_info)

        self.cmd_vel_pub.publish(cmd_vel)
    
    def _reset_obstacle_state(self):
        """Clear all obstacle-recovery state so the next detection starts fresh."""
        self.emergency_stop_time = None
        self.replan_attempts = 0
        self._obstacle_phase = 'wait'
        self._rotate_start_time = None
        self._rotate_vel = 0.0
        self._collision_hit_streak = 0
        self._collision_clear_streak = 0

    def reached_goal(self) -> bool:
        """Check if robot reached the goal."""
        if self.goal_pose is None or self.current_pose is None:
            return False
        
        dx = self.current_pose.position.x - self.goal_pose.pose.position.x
        dy = self.current_pose.position.y - self.goal_pose.pose.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance < self.goal_tolerance
    
    def attempt_replan_around_obstacle(self) -> bool:
        """
        Attempt to replan path from current position to goal when stuck.
        
        Returns:
            True if replanning succeeded, False otherwise
        """
        if self.current_pose is None or self.goal_pose is None:
            return False
        
        self.get_logger().info('Replanning path from current position to avoid obstacle...')
        
        # Plan new path
        new_path = self.plan_path(
            self.current_pose.position,
            self.goal_pose.pose.position
        )
        
        if new_path is None or len(new_path) < 2:
            self.get_logger().warn('Replanning failed - no valid path found')
            return False
        
        # Update paths
        self.raw_path = new_path
        
        # Apply trajectory optimization if enabled
        if self.use_traj_opt:
            smoothed_path = self.smooth_trajectory(new_path)
            if smoothed_path is not None and len(smoothed_path) > 0:
                self.current_path = smoothed_path
            else:
                self.current_path = new_path
        else:
            self.current_path = new_path
        
        self.path_index = 0
        self.publish_path_visualization()
        
        return True
    
    def pure_pursuit_control(self) -> Twist:
        """
        Pure pursuit path following controller with deceleration curve.

        Uses the standard pure pursuit curvature formula:
            κ = 2·sin(α) / L
            ω = v · κ

        Linear velocity is reduced smoothly as the robot approaches the goal
        (deceleration ramp) and is also rate-limited between ticks to prevent
        sudden speed changes that could tip the robot.
        """
        cmd = Twist()

        if self.current_path is None or len(self.current_path) == 0:
            return cmd

        lookahead_point = self.find_lookahead_point()
        if lookahead_point is None:
            lookahead_point = self.current_path[-1]

        dx = lookahead_point.x - self.current_pose.position.x
        dy = lookahead_point.y - self.current_pose.position.y
        L = math.sqrt(dx * dx + dy * dy)

        if L < 0.05:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self._prev_linear_x = 0.0
            self._prev_angular_z = 0.0
            return cmd

        robot_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        alpha = self.normalize_angle(math.atan2(dy, dx) - robot_yaw)

        # Turn in place if facing almost backwards
        if abs(alpha) > 1.0:
            cmd.angular.z = np.clip(2.0 * alpha, -self.max_angular_vel, self.max_angular_vel)
            cmd.linear.x = 0.0
            self._prev_linear_x = 0.0
            self._prev_angular_z = cmd.angular.z
            return cmd

        # --- Desired linear velocity ---
        # 1) Heading-error reduction
        desired_v = self.max_linear_vel * (1.0 - min(abs(alpha) / 1.0, 0.6))

        # 2) Goal-proximity deceleration ramp: v = max_vel * (dist / decel_radius)
        #    Start slowing down within 3× the goal tolerance.
        if self.goal_pose is not None:
            gx = self.goal_pose.pose.position.x - self.current_pose.position.x
            gy = self.goal_pose.pose.position.y - self.current_pose.position.y
            dist_to_goal = math.sqrt(gx * gx + gy * gy)
            decel_radius = max(self.goal_tolerance * 3.0, 0.6)
            if dist_to_goal < decel_radius:
                ramp = dist_to_goal / decel_radius
                desired_v = min(desired_v, self.max_linear_vel * ramp)

        desired_v = max(desired_v, 0.05)

        # 3) Rate-limit: cap change per tick (0.1 s) to max_acceleration
        dt = 0.1
        max_dv = self.max_acceleration * dt
        if desired_v > self._prev_linear_x:
            cmd.linear.x = min(desired_v, self._prev_linear_x + max_dv)
        else:
            cmd.linear.x = max(desired_v, self._prev_linear_x - max_dv)

        self._prev_linear_x = cmd.linear.x

        # --- Angular velocity (pure pursuit) ---
        curvature = 2.0 * math.sin(alpha) / L
        gain = 3.0

        raw_angular = gain * cmd.linear.x * curvature
        raw_angular = np.clip(raw_angular, -self.max_angular_vel, self.max_angular_vel)

        smoothed = 0.8 * raw_angular + 0.2 * self._prev_angular_z
        self._prev_angular_z = smoothed
        cmd.angular.z = smoothed

        return cmd
    
    def find_lookahead_point(self) -> Optional[Point]:
        """Find the lookahead point on the path."""
        if self.current_path is None:
            return None
        
        robot_pos = self.current_pose.position
        
        # Start from current path index
        for i in range(self.path_index, len(self.current_path)):
            point = self.current_path[i]
            dx = point.x - robot_pos.x
            dy = point.y - robot_pos.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance >= self.lookahead_distance:
                self.path_index = i
                return point
        
        # Return last point if no lookahead found
        return self.current_path[-1]
    
    def get_yaw_from_quaternion(self, q) -> float:
        """Extract yaw angle from quaternion."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def stop_robot(self, immediate: bool = False):
        """Stop the robot, ramping down over ~250 ms to prevent tipping.

        Args:
            immediate: If True, send a single zero-velocity command with no ramp.
                       Used for emergency stops where tipping risk is secondary.
        """
        if immediate or self._prev_linear_x < 0.02:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            self._prev_linear_x = 0.0
            self._prev_angular_z = 0.0
            return

        steps = 5
        dt = 0.05
        for i in range(steps, 0, -1):
            frac = i / steps
            cmd = Twist()
            cmd.linear.x = self._prev_linear_x * frac
            cmd.angular.z = self._prev_angular_z * frac
            self.cmd_vel_pub.publish(cmd)
            time.sleep(dt)
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self._prev_linear_x = 0.0
        self._prev_angular_z = 0.0
    
    def _inflated_occupied(self, wx: float, wy: float) -> bool:
        """Check if a world-coordinate point is occupied in the denoised inflated map.

        The inflated map already accounts for robot radius and has noise
        removed, so a single cell lookup is sufficient (no extra radius sweep).
        """
        if self.inflated_map is None or self.map_data is None:
            return False
        grid = self.world_to_grid(wx, wy)
        if grid is None:
            return True  # out-of-bounds = treat as occupied
        gx, gy = grid
        if not (0 <= gx < self.map_data.info.width and 0 <= gy < self.map_data.info.height):
            return True
        return bool(self.inflated_map[gy, gx])

    def _raw_occupied(self, wx: float, wy: float) -> bool:
        """Check if a world-coordinate point is occupied in the denoised RAW
        map (before inflation).  This gives the true distance to physical
        obstacles, unlike the inflated map which adds robot_radius padding."""
        if self.obstacle_map_raw is None or self.map_data is None:
            return False
        grid = self.world_to_grid(wx, wy)
        if grid is None:
            return True
        gx, gy = grid
        if not (0 <= gx < self.map_data.info.width and 0 <= gy < self.map_data.info.height):
            return True
        return bool(self.obstacle_map_raw[gy, gx])

    def _live_scan_points_map_frame(self) -> Optional[List[Tuple[float, float]]]:
        """Project the latest live LaserScan (e.g. cam_high depth) into the
        map frame.

        Returns None — meaning "no live data, fall back to the static map" —
        if there's no scan yet, it's older than `live_scan_max_age` (sensor
        stalled/crashed), or its TF isn't available. Never returns an empty
        list to mean "confirmed no obstacles"; a missing sensor must not look
        like a clear path.
        """
        scan = self.latest_scan
        if scan is None:
            return None

        scan_time = rclpy.time.Time.from_msg(scan.header.stamp)
        age = (self.get_clock().now() - scan_time).nanoseconds / 1e9
        if age > self.live_scan_max_age:
            return None

        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                'map', scan.header.frame_id, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

        q = tf_stamped.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        tx = tf_stamped.transform.translation.x
        ty = tf_stamped.transform.translation.y
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        points = []
        angle = scan.angle_min
        for r in scan.ranges:
            if not math.isnan(r) and not math.isinf(r) and scan.range_min <= r <= scan.range_max:
                px, py = r * math.cos(angle), r * math.sin(angle)
                points.append((cos_yaw * px - sin_yaw * py + tx,
                               sin_yaw * px + cos_yaw * py + ty))
            angle += scan.angle_increment
        return points

    def check_collision_ahead(self) -> dict:
        """
        Cast narrow rays ahead of the robot against the **raw** (non-inflated)
        obstacle map, AND against the latest live obstacle scan (cam_high
        depth -> /scan), to measure true distance to physical obstacles —
        whether or not they're on the saved map.

        Using the raw map instead of the inflated map prevents false triggers
        when the robot is merely near the inflation zone of a wall that A*
        already planned around.

        Returns:
            Dictionary with collision information:
            - emergency_stop: bool
            - reduce_speed: bool
            - distance: float (distance to nearest obstacle)
            - obstacle_positions: list of (x, y) positions
        """
        if self.current_pose is None or self.obstacle_map_raw is None:
            return {
                'emergency_stop': False,
                'reduce_speed': False,
                'distance': float('inf'),
                'obstacle_positions': []
            }

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        robot_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        min_distance = float('inf')
        obstacle_positions = []

        check_angles = [0, -0.2, 0.2]
        check_distance = max(self.safety_distance * 1.5, 0.8)
        step = self.map_data.info.resolution

        for angle_offset in check_angles:
            check_angle = robot_yaw + angle_offset
            cos_a = math.cos(check_angle)
            sin_a = math.sin(check_angle)
            dist = self.robot_radius
            while dist <= check_distance:
                check_x = robot_x + dist * cos_a
                check_y = robot_y + dist * sin_a

                if self._raw_occupied(check_x, check_y):
                    if dist < min_distance:
                        min_distance = dist
                    obstacle_positions.append((check_x, check_y))
                    break
                dist += step

        # Live obstacles (people, carts, anything not on the saved map).
        # Same rays/thresholds as above; source is /scan instead of the map.
        live_points = self._live_scan_points_map_frame()
        if live_points:
            ray_half_angle = 0.35  # rad; widens the narrow map rays into a
                                   # corridor sized for real-world sensor noise
            for px, py in live_points:
                dist = math.hypot(px - robot_x, py - robot_y)
                if dist < self.robot_radius or dist > check_distance:
                    continue
                bearing = self.normalize_angle(
                    math.atan2(py - robot_y, px - robot_x) - robot_yaw
                )
                for angle_offset in check_angles:
                    if abs(self.normalize_angle(bearing - angle_offset)) <= ray_half_angle:
                        if dist < min_distance:
                            min_distance = dist
                        obstacle_positions.append((px, py))
                        break

        emergency_stop = min_distance < self.emergency_stop_distance
        reduce_speed = min_distance < self.safety_distance

        return {
            'emergency_stop': emergency_stop,
            'reduce_speed': reduce_speed,
            'distance': min_distance,
            'obstacle_positions': obstacle_positions
        }
    
    def adjust_velocity_for_obstacles(self, cmd_vel: Twist, collision_info: dict) -> Twist:
        """
        Adjust velocity based on nearby obstacles.
        
        Args:
            cmd_vel: Original velocity command
            collision_info: Collision detection results
        
        Returns:
            Adjusted velocity command
        """
        distance = collision_info['distance']
        
        if distance == float('inf'):
            return cmd_vel
        
        # Scale velocity based on distance to obstacle
        # Linear interpolation between emergency_stop_distance and safety_distance
        if distance <= self.emergency_stop_distance:
            scale = 0.0
        elif distance >= self.safety_distance:
            scale = 1.0
        else:
            # Linear scaling between emergency stop and safety distance
            scale = (distance - self.emergency_stop_distance) / (self.safety_distance - self.emergency_stop_distance)
            scale = max(0.0, min(1.0, scale))
        
        # Apply scaling with minimum speed
        adjusted_cmd = Twist()
        adjusted_cmd.linear.x = cmd_vel.linear.x * scale
        adjusted_cmd.angular.z = cmd_vel.angular.z
        
        # Increase angular velocity to try to navigate around obstacle
        if scale < 0.5:
            adjusted_cmd.angular.z *= 1.5
        
        if scale < 1.0:
            self.get_logger().info(
                f'Obstacle at {distance:.2f}m - Reducing speed to {scale*100:.0f}%',
                throttle_duration_sec=1.0
            )
        
        return adjusted_cmd
    
    def visualize_obstacles(self, collision_info: dict):
        """
        Visualize detected obstacles in RViz.
        
        Args:
            collision_info: Collision detection results
        """
        if self.current_pose is None:
            return
            
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        
        # Delete old markers
        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.header.stamp = stamp
        delete_marker.ns = 'obstacles'
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Safety zone circle - use map frame and robot's current position
        safety_circle = Marker()
        safety_circle.header.frame_id = 'map'
        safety_circle.header.stamp = stamp
        safety_circle.ns = 'safety_zone'
        safety_circle.id = 0
        safety_circle.type = Marker.CYLINDER
        safety_circle.action = Marker.ADD
        
        # Position at robot's current location
        robot_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        safety_circle.pose.position.x = self.current_pose.position.x + (self.safety_distance / 2) * math.cos(robot_yaw)
        safety_circle.pose.position.y = self.current_pose.position.y + (self.safety_distance / 2) * math.sin(robot_yaw)
        safety_circle.pose.position.z = 0.0
        safety_circle.pose.orientation.w = 1.0
        
        safety_circle.scale.x = self.safety_distance
        safety_circle.scale.y = self.robot_radius * 2.5
        safety_circle.scale.z = 0.01
        
        # Color based on collision state
        if collision_info['emergency_stop']:
            safety_circle.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3)
        elif collision_info['reduce_speed']:
            safety_circle.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.3)
        else:
            safety_circle.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.2)
        
        marker_array.markers.append(safety_circle)
        
        # Visualize detected obstacle points
        for i, (obs_x, obs_y) in enumerate(collision_info['obstacle_positions'][:50]):  # Limit to 50 points
            obs_marker = Marker()
            obs_marker.header.frame_id = 'map'
            obs_marker.header.stamp = stamp
            obs_marker.ns = 'obstacles'
            obs_marker.id = i + 1
            obs_marker.type = Marker.SPHERE
            obs_marker.action = Marker.ADD
            obs_marker.pose.position.x = obs_x
            obs_marker.pose.position.y = obs_y
            obs_marker.pose.position.z = 0.1
            obs_marker.pose.orientation.w = 1.0
            obs_marker.scale.x = 0.1
            obs_marker.scale.y = 0.1
            obs_marker.scale.z = 0.1
            obs_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
            marker_array.markers.append(obs_marker)
        
        self.obstacle_marker_pub.publish(marker_array)
    
    def continuous_path_visualization(self):
        """Continuously publish path visualization for RViz."""
        if self.current_path is not None:
            self.publish_path_visualization()
    
    def publish_path_visualization(self):
        """Publish path for visualization in RViz with both raw and smoothed paths."""
        stamp = self.get_clock().now().to_msg()
        
        # Publish raw path if available
        if self.raw_path is not None:
            raw_path_msg = Path()
            raw_path_msg.header.frame_id = 'map'
            raw_path_msg.header.stamp = stamp
            
            for point in self.raw_path:
                pose = PoseStamped()
                pose.header = raw_path_msg.header
                pose.pose.position = point
                pose.pose.orientation.w = 1.0
                raw_path_msg.poses.append(pose)
            
            self.raw_path_pub.publish(raw_path_msg)
            self.get_logger().info(
                f'Published A* path with {len(self.raw_path)} waypoints (green line with spheres)',
                throttle_duration_sec=2.0
            )
        
        # Publish smoothed/current path
        if self.current_path is None:
            return
        
        smoothed_path_msg = Path()
        smoothed_path_msg.header.frame_id = 'map'
        smoothed_path_msg.header.stamp = stamp
        
        for point in self.current_path:
            pose = PoseStamped()
            pose.header = smoothed_path_msg.header
            pose.pose.position = point
            pose.pose.orientation.w = 1.0
            smoothed_path_msg.poses.append(pose)
        
        self.smoothed_path_pub.publish(smoothed_path_msg)
        self.path_pub.publish(smoothed_path_msg)  # Also publish to default topic (robot follows this!)
        
        # Publish enhanced markers
        marker_array = MarkerArray()
        
        # Clear old markers first
        delete_all = Marker()
        delete_all.header.frame_id = 'map'
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        
        # Raw A* path visualization (thin green line)
        if self.raw_path is not None and len(self.raw_path) > 1:
            raw_line = Marker()
            raw_line.header.frame_id = 'map'
            raw_line.header.stamp = stamp
            raw_line.ns = 'raw_path_line'
            raw_line.id = 0
            raw_line.type = Marker.LINE_STRIP
            raw_line.action = Marker.ADD
            raw_line.scale.x = 0.03  # Thin line
            raw_line.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)  # Green
            raw_line.pose.orientation.w = 1.0
            
            for point in self.raw_path:
                raw_line.points.append(point)
            
            marker_array.markers.append(raw_line)
        
        # Optimized trajectory visualization (blue line - Robot follows this!)
        smooth_line = Marker()
        smooth_line.header.frame_id = 'map'
        smooth_line.header.stamp = stamp
        smooth_line.ns = 'smoothed_path'
        smooth_line.id = 0
        smooth_line.type = Marker.LINE_STRIP
        smooth_line.action = Marker.ADD
        smooth_line.scale.x = 0.06  # Thin line
        smooth_line.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.9)  # Cyan/bright blue
        smooth_line.pose.orientation.w = 1.0
        
        for point in self.current_path:
            smooth_line.points.append(point)
        
        marker_array.markers.append(smooth_line)
        
        # Velocity profile markers (if trajectory optimization is enabled)
        # if self.optimized_trajectory is not None:
        #     max_vel = max([v for _, v, _ in self.optimized_trajectory])
        #     for i, (point, velocity, time) in enumerate(self.optimized_trajectory[::5]):  # Sample every 5th point
        #         vel_marker = Marker()
        #         vel_marker.header.frame_id = 'map'
        #         vel_marker.header.stamp = stamp
        #         vel_marker.ns = 'velocity_profile'
        #         vel_marker.id = i + 1000
        #         vel_marker.type = Marker.ARROW
        #         vel_marker.action = Marker.ADD
        #         vel_marker.pose.position = point
        #         vel_marker.pose.orientation.w = 1.0
                
        #         # Scale arrow by velocity
        #         scale = velocity / max_vel if max_vel > 0 else 0.0
        #         vel_marker.scale.x = 0.2 * scale
        #         vel_marker.scale.y = 0.05
        #         vel_marker.scale.z = 0.05
                
        #         # Color by velocity (green = fast, red = slow)
        #         vel_marker.color = ColorRGBA(
        #             r=1.0 - scale,
        #             g=scale,
        #             b=0.0,
        #             a=0.7
        #         )
        #         marker_array.markers.append(vel_marker)
        
        # Start and goal markers
        if len(self.current_path) > 0:
            # Start marker (blue sphere)
            start_marker = Marker()
            start_marker.header.frame_id = 'map'
            start_marker.header.stamp = stamp
            start_marker.ns = 'start_goal'
            start_marker.id = 10000
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD
            start_marker.pose.position = self.current_path[0]
            start_marker.pose.orientation.w = 1.0
            start_marker.scale.x = 0.2
            start_marker.scale.y = 0.2
            start_marker.scale.z = 0.2
            start_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            marker_array.markers.append(start_marker)
            
            # Goal marker (red sphere)
            goal_marker = Marker()
            goal_marker.header.frame_id = 'map'
            goal_marker.header.stamp = stamp
            goal_marker.ns = 'start_goal'
            goal_marker.id = 10001
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position = self.current_path[-1]
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.25
            goal_marker.scale.y = 0.25
            goal_marker.scale.z = 0.25
            goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            marker_array.markers.append(goal_marker)
            
            # Text label for A* path (near start)
            if self.raw_path is not None and len(self.raw_path) > 2:
                astar_label = Marker()
                astar_label.header.frame_id = 'map'
                astar_label.header.stamp = stamp
                astar_label.ns = 'path_labels'
                astar_label.id = 20000
                astar_label.type = Marker.TEXT_VIEW_FACING
                astar_label.action = Marker.ADD
                astar_label.pose.position = self.raw_path[1]
                astar_label.pose.position.z = 0.3
                astar_label.pose.orientation.w = 1.0
                astar_label.scale.z = 0.15
                astar_label.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                astar_label.text = "A* Path"
                marker_array.markers.append(astar_label)
            
            # Text label for optimized trajectory (near start) - ROBOT FOLLOWS THIS
            if len(self.current_path) > 2:
                opt_label = Marker()
                opt_label.header.frame_id = 'map'
                opt_label.header.stamp = stamp
                opt_label.ns = 'path_labels'
                opt_label.id = 20001
                opt_label.type = Marker.TEXT_VIEW_FACING
                opt_label.action = Marker.ADD
                opt_label.pose.position = self.current_path[1]
                opt_label.pose.position.z = 0.5
                opt_label.pose.orientation.w = 1.0
                opt_label.scale.z = 0.2  # Larger text
                opt_label.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0)
                opt_label.text = ">>> ROBOT FOLLOWS THIS PATH <<<"
                marker_array.markers.append(opt_label)
                
                # Additional label at midpoint
                mid_idx = len(self.current_path) // 2
                opt_label2 = Marker()
                opt_label2.header.frame_id = 'map'
                opt_label2.header.stamp = stamp
                opt_label2.ns = 'path_labels'
                opt_label2.id = 20002
                opt_label2.type = Marker.TEXT_VIEW_FACING
                opt_label2.action = Marker.ADD
                opt_label2.pose.position = self.current_path[mid_idx]
                opt_label2.pose.position.z = 0.6
                opt_label2.pose.orientation.w = 1.0
                opt_label2.scale.z = 0.18
                opt_label2.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
                opt_label2.text = "Optimized (Blue)"
                marker_array.markers.append(opt_label2)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleNavPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

