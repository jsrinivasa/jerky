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

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav2_msgs.action import NavigateToPose

import numpy as np
import math
from typing import List, Tuple, Optional
from enum import Enum


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
        
        self.use_nav2 = self.get_parameter('use_nav2').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_resolution = self.get_parameter('path_resolution').value
        
        # State
        self.state = NavigationState.IDLE
        self.current_pose = None
        self.goal_pose = None
        self.current_path = None
        self.map_data = None
        self.path_index = 0
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/nav_markers', 10)
        
        # Nav2 Action Client (optional)
        if self.use_nav2:
            self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
            self.get_logger().info('Using Nav2 for navigation')
        else:
            self.get_logger().info('Using simple path planner')
        
        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Simple Navigation Planner initialized')
        self.get_logger().info('Set a goal pose in RViz2 using "2D Goal Pose" tool')
    
    def goal_callback(self, msg: PoseStamped):
        """Handle new goal pose from RViz2."""
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
        """Store map data for path planning."""
        if self.map_data is None:
            self.get_logger().info('Map received!')
        self.map_data = msg
    
    def odom_callback(self, msg: Odometry):
        """Update current robot pose."""
        self.current_pose = msg.pose.pose
    
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
            self.get_logger().warn('No odometry data yet!')
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
            self.get_logger().error('Path planning failed!')
            self.state = NavigationState.FAILED
            return
        
        self.current_path = path
        self.path_index = 0
        self.state = NavigationState.FOLLOWING
        
        self.get_logger().info(f'Path planned with {len(path)} waypoints')
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
            return None
        
        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start.x, start.y)
        goal_grid = self.world_to_grid(goal.x, goal.y)
        
        if start_grid is None or goal_grid is None:
            self.get_logger().error('Start or goal outside map bounds')
            return None
        
        # Simple A* implementation
        path_grid = self.astar(start_grid, goal_grid)
        
        if path_grid is None:
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
        
        # Simplify path (remove intermediate points on straight lines)
        path_world = self.simplify_path(path_world)
        
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
        
        # Check if goal is valid
        if not self.is_free(goal[0], goal[1]):
            self.get_logger().warn('Goal is in occupied space!')
            # Try to find nearest free cell
            goal = self.find_nearest_free_cell(goal[0], goal[1])
            if goal is None:
                return None
        
        # A* data structures
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        max_iterations = 10000
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
                return path
            
            open_set.remove(current)
            
            # Check neighbors (8-connected)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                
                # Check if free
                if not self.is_free(neighbor[0], neighbor[1]):
                    continue
                
                # Calculate cost
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    open_set.add(neighbor)
        
        self.get_logger().error('A* failed to find path')
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
        
        # Consider cells with occupancy < 50 as free
        # Unknown cells (-1) are considered free
        value = self.map_data.data[idx]
        return value < 50
    
    def find_nearest_free_cell(self, x: int, y: int, max_radius: int = 20) -> Optional[Tuple[int, int]]:
        """Find nearest free cell to given position."""
        for radius in range(1, max_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        nx, ny = x + dx, y + dy
                        if self.is_free(nx, ny):
                            return (nx, ny)
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
        
        # Pure pursuit control
        cmd_vel = self.pure_pursuit_control()
        self.cmd_vel_pub.publish(cmd_vel)
    
    def reached_goal(self) -> bool:
        """Check if robot reached the goal."""
        if self.goal_pose is None or self.current_pose is None:
            return False
        
        dx = self.current_pose.position.x - self.goal_pose.pose.position.x
        dy = self.current_pose.position.y - self.goal_pose.pose.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance < self.goal_tolerance
    
    def pure_pursuit_control(self) -> Twist:
        """
        Pure pursuit path following controller.
        
        Returns:
            Twist command
        """
        cmd = Twist()
        
        if self.current_path is None or len(self.current_path) == 0:
            return cmd
        
        # Find lookahead point
        lookahead_point = self.find_lookahead_point()
        
        if lookahead_point is None:
            # If no lookahead point, move to last point
            lookahead_point = self.current_path[-1]
        
        # Calculate control
        dx = lookahead_point.x - self.current_pose.position.x
        dy = lookahead_point.y - self.current_pose.position.y
        
        # Get robot yaw
        robot_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        
        # Calculate angle to lookahead point
        angle_to_point = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(angle_to_point - robot_yaw)
        
        # Calculate velocities
        distance = math.sqrt(dx**2 + dy**2)
        
        # Angular velocity (proportional control)
        cmd.angular.z = np.clip(2.0 * angle_diff, -self.max_angular_vel, self.max_angular_vel)
        
        # Linear velocity (reduce when turning)
        if abs(angle_diff) > 0.5:  # ~30 degrees
            cmd.linear.x = 0.1  # Slow down for sharp turns
        else:
            cmd.linear.x = np.clip(distance, 0.0, self.max_linear_vel)
        
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
    
    def stop_robot(self):
        """Stop the robot."""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    def publish_path_visualization(self):
        """Publish path for visualization in RViz."""
        if self.current_path is None:
            return
        
        # Publish as Path message
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position = point
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        
        # Publish as markers for better visualization
        marker_array = MarkerArray()
        
        # Line strip for path
        line_marker = Marker()
        line_marker.header = path_msg.header
        line_marker.ns = 'path'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        
        for point in self.current_path:
            line_marker.points.append(point)
        
        marker_array.markers.append(line_marker)
        
        # Waypoint markers
        for i, point in enumerate(self.current_path):
            marker = Marker()
            marker.header = path_msg.header
            marker.ns = 'waypoints'
            marker.id = i + 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = point
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
            marker_array.markers.append(marker)
        
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

