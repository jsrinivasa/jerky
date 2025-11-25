#!/usr/bin/env python3

"""
Move the robot with specified velocity vector and duration.

This script moves the Interbotix Slate base for a given duration with specified
linear and angular velocities, allowing for complex motions including forward/backward,
strafing (if supported), and rotation.
"""

import argparse
import json
import sys
import time
from typing import List, Tuple


def execute_movement_sequence_ros2(movements: List[Tuple[float, float, float, float]]):
    """
    Execute a sequence of movements using ROS 2.

    :param movements: List of tuples (linear_x, linear_y, angular_z, duration)
    """
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    import threading

    class MoveSequenceNode(Node):
        def __init__(self):
            super().__init__('move_sequence_node')
            self.pub = self.create_publisher(Twist, 'mobile_base/cmd_vel', 10)
            time.sleep(0.5)
            self.get_logger().info('Publisher initialized')

        def execute_movement(self, linear_x: float, linear_y: float, angular_z: float, duration: float):
            """Execute a single movement phase."""
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.linear.y = float(linear_y)
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = float(angular_z)
            
            self.get_logger().info(
                f'Phase: linear_x={linear_x} m/s, linear_y={linear_y} m/s, '
                f'angular_z={angular_z} rad/s for {duration} seconds'
            )
            
            # Publish velocity commands at 10Hz
            rate = self.create_rate(10)
            start_time = time.time()
            
            try:
                while time.time() - start_time < duration and rclpy.ok():
                    self.pub.publish(twist)
                    rate.sleep()
            except KeyboardInterrupt:
                self.get_logger().info('Movement interrupted by user')
                raise
            
        def execute_sequence(self, movements: List[Tuple[float, float, float, float]]):
            """Execute a sequence of movements."""
            self.get_logger().info(f'Executing sequence of {len(movements)} movements...')
            
            try:
                for i, (linear_x, linear_y, angular_z, duration) in enumerate(movements, 1):
                    self.get_logger().info(f'Starting phase {i}/{len(movements)}')
                    self.execute_movement(linear_x, linear_y, angular_z, duration)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.get_logger().info('Sequence interrupted by user')
            finally:
                stop_twist = Twist()
                self.pub.publish(stop_twist)
                self.get_logger().info('Sequence complete. Robot stopped.')

    rclpy.init()
    node = MoveSequenceNode()
    
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        node.execute_sequence(movements)
    finally:
        # Clean up
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


def move_forward_ros2(linear_x: float, linear_y: float, angular_z: float, duration: float):
    """
    Move the robot using ROS 2 with specified velocity vector (single movement).

    :param linear_x: Linear velocity in x direction (m/s, positive = forward)
    :param linear_y: Linear velocity in y direction (m/s, positive = left)
    :param angular_z: Angular velocity around z axis (rad/s, positive = counter-clockwise)
    :param duration: Time duration in seconds
    """
    execute_movement_sequence_ros2([(linear_x, linear_y, angular_z, duration)])


def execute_movement_sequence_trossen(movements: List[Tuple[float, float, float, float]]):
    """
    Execute a sequence of movements using the trossen_slate library.

    :param movements: List of tuples (linear_x, linear_y, angular_z, duration)
    """
    try:
        import trossen_slate as trossen
    except ImportError:
        print("Error: trossen_slate library not found.")
        print("Please ensure the library is installed or use --method ros2")
        sys.exit(1)
    
    slate = trossen.TrossenSlate()
    
    success, result = slate.init_base()
    if not success:
        print(f"Error: Failed to initialize base. Result: {result}")
        sys.exit(1)
    
    print(f"Initialization success: {success}")
    print(f'Executing sequence of {len(movements)} movements...')
    
    try:
        for i, (linear_x, linear_y, angular_z, duration) in enumerate(movements, 1):
            print(f'\nPhase {i}/{len(movements)}:')
            print(f'  linear_x={linear_x} m/s, angular_z={angular_z} rad/s for {duration} seconds')
            if linear_y != 0.0:
                print(f'  Warning: linear_y={linear_y} specified, but may not be supported by the base')
            
            start_time = time.time()
            while time.time() - start_time < duration:
                slate.set_cmd_vel(linear_x, angular_z)  # linear_velocity, angular_velocity
                time.sleep(0.1)  
            
            # Creates a pause in between movement sets
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nSequence interrupted by user")
    finally:
        # Stop the robot
        slate.set_cmd_vel(0.0, 0.0)
        print("\nSequence complete. Robot stopped.")


def move_forward_trossen(linear_x: float, linear_y: float, angular_z: float, duration: float):
    """
    Move the robot using the trossen_slate library with specified velocity vector (single movement).

    :param linear_x: Linear velocity in x direction (m/s, positive = forward)
    :param linear_y: Linear velocity in y direction (m/s, positive = left, may not be supported)
    :param angular_z: Angular velocity around z axis (rad/s, positive = counter-clockwise)
    :param duration: Time duration in seconds
    """
    # Execute as a single-movement sequence
    execute_movement_sequence_trossen([(linear_x, linear_y, angular_z, duration)])


def parse_sequence(sequence_str: str) -> List[Tuple[float, float, float, float]]:
    """
    Parse a sequence string into a list of movement tuples.
    
    Format: "linear_x,linear_y,angular_z,duration;linear_x,linear_y,angular_z,duration;..."
    Example: "0.3,0,0,2.0;0,0,0.8,3.0;-0.3,0,0,2.0"
    
    :param sequence_str: Semicolon-separated movement phases
    :return: List of tuples (linear_x, linear_y, angular_z, duration)
    """
    movements = []
    phases = sequence_str.split(';')
    
    for i, phase in enumerate(phases, 1):
        try:
            parts = phase.strip().split(',')
            if len(parts) != 4:
                print(f"Error: Phase {i} must have exactly 4 values (linear_x,linear_y,angular_z,duration)")
                sys.exit(1)
            
            linear_x, linear_y, angular_z, duration = map(float, parts)
            
            if duration <= 0:
                print(f"Error: Phase {i} duration must be positive")
                sys.exit(1)
            
            movements.append((linear_x, linear_y, angular_z, duration))
        except ValueError:
            print(f"Error: Phase {i} contains invalid numeric values")
            sys.exit(1)
    
    return movements


def parse_sequence_file(filename: str) -> List[Tuple[float, float, float, float]]:
    """
    Parse a JSON file containing a movement sequence.
    
    JSON format:
    {
      "movements": [
        {"linear_x": 0.3, "linear_y": 0.0, "angular_z": 0.0, "duration": 2.0},
        {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.8, "duration": 3.0},
        {"linear_x": -0.3, "linear_y": 0.0, "angular_z": 0.0, "duration": 2.0}
      ]
    }
    
    :param filename: Path to JSON file
    :return: List of tuples (linear_x, linear_y, angular_z, duration)
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filename}': {e}")
        sys.exit(1)
    
    if 'movements' not in data:
        print("Error: JSON file must contain a 'movements' array")
        sys.exit(1)
    
    movements = []
    for i, movement in enumerate(data['movements'], 1):
        try:
            linear_x = movement.get('linear_x', 0.0)
            linear_y = movement.get('linear_y', 0.0)
            angular_z = movement.get('angular_z', 0.0)
            duration = movement['duration']
            
            if duration <= 0:
                print(f"Error: Movement {i} duration must be positive")
                sys.exit(1)
            
            movements.append((linear_x, linear_y, angular_z, duration))
        except KeyError:
            print(f"Error: Movement {i} is missing required 'duration' field")
            sys.exit(1)
        except (TypeError, ValueError):
            print(f"Error: Movement {i} contains invalid numeric values")
            sys.exit(1)
    
    return movements


def main():
    """Parse arguments and execute movement command."""
    parser = argparse.ArgumentParser(
        description='Move the robot with specified velocity vector and duration, or execute a sequence of movements.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Sequence arguments (mutually exclusive with single movement)
    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help='Movement sequence as semicolon-separated phases: "linear_x,linear_y,angular_z,duration;..." '
             'Example: "0.3,0,0,2.0;0,0,0.8,3.0;-0.3,0,0,2.0" (forward, spin, backward)'
    )
    
    parser.add_argument(
        '--sequence-file',
        type=str,
        default=None,
        help='Path to JSON file containing movement sequence'
    )
    
    # Single movement arguments
    parser.add_argument(
        '--linear-x',
        type=float,
        default=0.0,
        help='Linear velocity in x direction (m/s, positive = forward, recommended range: -0.7 to 0.7)'
    )
    
    parser.add_argument(
        '--linear-y',
        type=float,
        default=0.0,
        help='Linear velocity in y direction (m/s, positive = left, may not be supported by all bases)'
    )
    
    parser.add_argument(
        '--angular-z',
        type=float,
        default=0.0,
        help='Angular velocity around z axis (rad/s, positive = counter-clockwise, recommended range: -1.0 to 1.0)'
    )
    
    # Legacy argument for backward compatibility
    parser.add_argument(
        '--speed',
        type=float,
        default=None,
        help='(Legacy) Linear velocity in m/s - equivalent to --linear-x'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Time duration in seconds (required for single movements, ignored for sequences)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['ros2', 'trossen'],
        default='ros2',
        help='Method to use for controlling the robot (ros2 uses ROS 2 topics, trossen uses direct library)'
    )
    
    args = parser.parse_args()
    
    # Determine if we're executing a sequence or single movement
    is_sequence = args.sequence is not None or args.sequence_file is not None
    
    if args.sequence and args.sequence_file:
        print("Error: Cannot specify both --sequence and --sequence-file")
        sys.exit(1)
    
    # Parse the movement sequence or single movement
    if is_sequence:
        if args.sequence:
            movements = parse_sequence(args.sequence)
        else:
            movements = parse_sequence_file(args.sequence_file)
        
        print(f"Parsed {len(movements)} movement phases")
    else:
        # Single movement mode
        if args.duration is None:
            print("Error: --duration is required for single movements")
            print("Use --sequence or --sequence-file for multi-phase movements")
            sys.exit(1)
        
        # Handle legacy --speed argument for backward compatibility
        if args.speed is not None:
            if args.linear_x != 0.0:
                print("Error: Cannot specify both --speed and --linear-x")
                sys.exit(1)
            args.linear_x = args.speed
        
        # Check if at least one velocity component is specified
        if args.linear_x == 0.0 and args.linear_y == 0.0 and args.angular_z == 0.0:
            print("Error: At least one velocity component must be non-zero")
            print("Use --linear-x, --linear-y, and/or --angular-z to specify velocities")
            sys.exit(1)
        
        # Validate arguments
        if args.duration <= 0:
            print("Error: Duration must be positive")
            sys.exit(1)
        
        movements = [(args.linear_x, args.linear_y, args.angular_z, args.duration)]
    
    # Validate movement velocities
    for i, (linear_x, linear_y, angular_z, duration) in enumerate(movements, 1):
        if abs(linear_x) > 1.0 or abs(linear_y) > 1.0:
            print(f"Warning: Phase {i} linear velocity is quite high (x={linear_x}, y={linear_y} m/s)")
            print("Recommended range is -0.7 to 0.7 m/s")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted")
                sys.exit(0)
            break  # Only warn once
        
        if abs(angular_z) > 2.0:
            print(f"Warning: Phase {i} angular velocity {angular_z} rad/s is quite high")
            print("Recommended range is -1.0 to 1.0 rad/s")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted")
                sys.exit(0)
            break  # Only warn once
    
    # Execute movement based on selected method
    try:
        if args.method == 'ros2':
            execute_movement_sequence_ros2(movements)
        elif args.method == 'trossen':
            execute_movement_sequence_trossen(movements)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

