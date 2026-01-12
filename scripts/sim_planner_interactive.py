#!/usr/bin/env python3
"""
Interactive Path Planner Simulation Demo

Provides an interactive CLI to test path planning in simulation.
You can send goal poses and see the robot plan and execute in real-time.

Features:
- Interactive CLI menu
- Predefined test poses
- Custom pose input
- Joint goal input
- Circular motion demo
- Auto-demo mode

Usage:
    ros2 run aloha sim_planner_interactive
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import sys
import threading

from aloha.simple_path_planner import SimplePathPlanner


class InteractiveSimPlanner(Node):
    """Interactive simulation planner with CLI interface."""
    
    def __init__(self):
        super().__init__('interactive_sim_planner')
        
        self.get_logger().info("="*70)
        self.get_logger().info("  ALOHA Path Planner - Interactive Simulation Demo")
        self.get_logger().info("="*70)
        
        # Wait a bit for MoveIt to initialize
        self.get_logger().info("Waiting for MoveIt to initialize...")
        import time
        time.sleep(2.0)
        
        try:
            # Create path planner
            self.planner = SimplePathPlanner(
                self,
                group_name="interbotix_arm",
                velocity_scaling=0.3,
                acceleration_scaling=0.3
            )
            
            self.get_logger().info("✓ Path planner initialized!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize planner: {e}")
            self.get_logger().error("Make sure MoveIt is running!")
            raise
        
        # Publisher for goal visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/goal_pose_markers',
            10
        )
        
        # Store predefined poses
        self.predefined_poses = self._create_predefined_poses()
        
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("  Ready! Robot is in simulation mode.")
        self.get_logger().info("  You can now plan and visualize motions in RViz.")
        self.get_logger().info("="*70)
    
    def _create_predefined_poses(self):
        """Create a set of predefined test poses."""
        poses = {}
        
        # Home position
        poses['home'] = {
            'joints': [0.0, -0.96, 1.16, 0.0, -0.3, 0.0],
            'description': 'Home position'
        }
        
        # Upright
        poses['upright'] = {
            'joints': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'description': 'All joints at zero'
        }
        
        # Forward reach
        poses['forward'] = {
            'pose': self.planner.create_pose(0.3, 0.0, 0.15, 0.0, np.pi/2, 0.0),
            'description': 'Reaching forward'
        }
        
        # Left reach
        poses['left'] = {
            'pose': self.planner.create_pose(0.25, 0.15, 0.15, 0.0, np.pi/2, 0.0),
            'description': 'Reaching left'
        }
        
        # Right reach
        poses['right'] = {
            'pose': self.planner.create_pose(0.25, -0.15, 0.15, 0.0, np.pi/2, 0.0),
            'description': 'Reaching right'
        }
        
        # High reach
        poses['high'] = {
            'pose': self.planner.create_pose(0.2, 0.0, 0.25, 0.0, np.pi/2, 0.0),
            'description': 'Reaching high'
        }
        
        # Low reach
        poses['low'] = {
            'pose': self.planner.create_pose(0.25, 0.0, 0.05, 0.0, np.pi/2, 0.0),
            'description': 'Reaching low'
        }
        
        return poses
    
    def visualize_goal_pose(self, pose: Pose, label: str = "Goal"):
        """Publish marker to visualize goal pose in RViz."""
        marker_array = MarkerArray()
        
        # Arrow marker for pose
        arrow = Marker()
        arrow.header.frame_id = "world"
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = "goal_pose"
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose = pose
        arrow.scale.x = 0.1  # Length
        arrow.scale.y = 0.01  # Width
        arrow.scale.z = 0.01  # Height
        arrow.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        arrow.lifetime.sec = 30
        
        # Text marker for label
        text = Marker()
        text.header.frame_id = "world"
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = "goal_label"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = pose.position.x
        text.pose.position.y = pose.position.y
        text.pose.position.z = pose.position.z + 0.1
        text.text = label
        text.scale.z = 0.03
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.lifetime.sec = 30
        
        marker_array.markers = [arrow, text]
        self.marker_pub.publish(marker_array)
    
    def show_menu(self):
        """Display the interactive menu."""
        print("\n" + "="*70)
        print("  Interactive Path Planner Menu")
        print("="*70)
        print("\n📍 Predefined Poses:")
        print("  1. Home position")
        print("  2. Forward reach")
        print("  3. Left reach")
        print("  4. Right reach")
        print("  5. High reach")
        print("  6. Low reach")
        print("  7. Upright position (all zeros)")
        print("\n🎯 Custom Goals:")
        print("  8. Enter custom Cartesian pose (x, y, z)")
        print("  9. Enter custom joint angles")
        print("\n🎨 Demonstrations:")
        print("  10. Draw a circle")
        print("  11. Draw a square")
        print("  12. Auto demo (all predefined poses)")
        print("\n📊 Information:")
        print("  i. Show current state")
        print("  h. Show workspace limits")
        print("\n  q. Quit")
        print("="*70)
    
    def get_current_state(self):
        """Display current robot state."""
        print("\n" + "="*70)
        print("  Current Robot State")
        print("="*70)
        
        try:
            joints = self.planner.get_current_joint_values()
            pose = self.planner.get_current_pose()
            
            print("\n📐 Joint Angles (radians):")
            joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
            for name, angle in zip(joint_names, joints):
                print(f"  {name:15s}: {angle:7.3f} rad ({np.degrees(angle):7.2f}°)")
            
            print("\n📍 End-Effector Pose:")
            print(f"  Position:")
            print(f"    x: {pose.position.x:7.3f} m")
            print(f"    y: {pose.position.y:7.3f} m")
            print(f"    z: {pose.position.z:7.3f} m")
            print(f"  Orientation (quaternion):")
            print(f"    x: {pose.orientation.x:7.3f}")
            print(f"    y: {pose.orientation.y:7.3f}")
            print(f"    z: {pose.orientation.z:7.3f}")
            print(f"    w: {pose.orientation.w:7.3f}")
            
        except Exception as e:
            print(f"❌ Error getting state: {e}")
    
    def show_workspace_limits(self):
        """Display typical workspace limits."""
        print("\n" + "="*70)
        print("  Workspace Limits (Typical for wx250s)")
        print("="*70)
        print("\n📏 Position Limits:")
        print("  x: 0.1 to 0.5 m (forward)")
        print("  y: -0.3 to 0.3 m (left/right)")
        print("  z: -0.1 to 0.4 m (height)")
        print("\n💡 Recommended Safe Zone:")
        print("  x: 0.2 to 0.4 m")
        print("  y: -0.2 to 0.2 m")
        print("  z: 0.05 to 0.3 m")
        print("\n⚠️  Goals outside this range may not be reachable!")
    
    def execute_predefined_pose(self, pose_key: str) -> bool:
        """Execute a predefined pose."""
        if pose_key not in self.predefined_poses:
            print(f"❌ Unknown pose: {pose_key}")
            return False
        
        pose_data = self.predefined_poses[pose_key]
        print(f"\n🎯 Executing: {pose_data['description']}")
        
        if 'joints' in pose_data:
            # Joint goal
            print(f"   Joint goal: {[f'{j:.2f}' for j in pose_data['joints']]}")
            success = self.planner.move_to_joint_goal(pose_data['joints'])
        else:
            # Pose goal
            pose = pose_data['pose']
            print(f"   Pose goal: x={pose.position.x:.3f}, "
                  f"y={pose.position.y:.3f}, z={pose.position.z:.3f}")
            self.visualize_goal_pose(pose, pose_data['description'])
            success = self.planner.move_to_pose_goal(pose)
        
        if success:
            print("✅ Goal reached!")
        else:
            print("❌ Failed to reach goal")
        
        return success
    
    def execute_custom_cartesian(self):
        """Execute a custom Cartesian pose."""
        print("\n📍 Enter Custom Cartesian Pose")
        print("="*70)
        
        try:
            x = float(input("  x (meters, forward, e.g., 0.3): "))
            y = float(input("  y (meters, left/right, e.g., 0.0): "))
            z = float(input("  z (meters, height, e.g., 0.15): "))
            
            print("\n  Orientation (press Enter for default: pointing down)")
            roll_str = input("  roll (radians, default 0.0): ")
            pitch_str = input("  pitch (radians, default 1.57 [π/2]): ")
            yaw_str = input("  yaw (radians, default 0.0): ")
            
            roll = float(roll_str) if roll_str else 0.0
            pitch = float(pitch_str) if pitch_str else np.pi/2
            yaw = float(yaw_str) if yaw_str else 0.0
            
            pose = self.planner.create_pose(x, y, z, roll, pitch, yaw)
            
            print(f"\n🎯 Planning to: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            self.visualize_goal_pose(pose, f"Custom ({x:.2f}, {y:.2f}, {z:.2f})")
            
            success = self.planner.move_to_pose_goal(pose)
            
            if success:
                print("✅ Goal reached!")
            else:
                print("❌ Failed to reach goal - may be outside workspace")
            
        except ValueError:
            print("❌ Invalid input! Please enter numbers.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def execute_custom_joints(self):
        """Execute custom joint angles."""
        print("\n📐 Enter Custom Joint Angles")
        print("="*70)
        print("  Enter 6 joint angles in radians (or degrees if you prefer)")
        
        try:
            use_degrees = input("  Use degrees? (y/n, default n): ").lower() == 'y'
            
            joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
            joints = []
            
            for name in joint_names:
                value_str = input(f"  {name}: ")
                value = float(value_str)
                if use_degrees:
                    value = np.radians(value)
                joints.append(value)
            
            print(f"\n🎯 Planning to: {[f'{j:.3f}' for j in joints]} rad")
            
            success = self.planner.move_to_joint_goal(joints)
            
            if success:
                print("✅ Goal reached!")
            else:
                print("❌ Failed to reach goal")
                
        except ValueError:
            print("❌ Invalid input! Please enter numbers.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def draw_circle(self):
        """Draw a circular pattern."""
        print("\n🎨 Drawing a Circle")
        print("="*70)
        
        try:
            current = self.planner.get_current_pose()
            
            # Ask for parameters
            radius_str = input("  Radius (meters, default 0.05): ")
            radius = float(radius_str) if radius_str else 0.05
            
            num_points_str = input("  Number of points (default 16): ")
            num_points = int(num_points_str) if num_points_str else 16
            
            # Generate circle waypoints
            center_x = current.position.x
            center_y = current.position.y
            z = current.position.z
            
            waypoints = []
            for i in range(num_points + 1):
                angle = 2 * np.pi * i / num_points
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                waypoints.append(self.planner.create_pose(x, y, z, 0, np.pi/2, 0))
            
            print(f"\n  Planning circular path with radius {radius}m...")
            
            plan, fraction = self.planner.plan_cartesian_path(waypoints, eef_step=0.005)
            
            if fraction > 0.9:
                print(f"✓ Path planned: {fraction*100:.1f}% achieved")
                print("  Executing...")
                success = self.planner.execute_plan(plan)
                if success:
                    print("✅ Circle completed!")
                else:
                    print("❌ Execution failed")
            else:
                print(f"❌ Could only achieve {fraction*100:.1f}% of path")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def draw_square(self):
        """Draw a square pattern."""
        print("\n🎨 Drawing a Square")
        print("="*70)
        
        try:
            current = self.planner.get_current_pose()
            
            size_str = input("  Size (meters, default 0.1): ")
            size = float(size_str) if size_str else 0.1
            
            # Generate square corners
            x = current.position.x
            y = current.position.y
            z = current.position.z
            
            waypoints = [
                current,
                self.planner.create_pose(x + size, y, z, 0, np.pi/2, 0),
                self.planner.create_pose(x + size, y + size, z, 0, np.pi/2, 0),
                self.planner.create_pose(x, y + size, z, 0, np.pi/2, 0),
                self.planner.create_pose(x, y, z, 0, np.pi/2, 0),
            ]
            
            print(f"\n  Planning square path with size {size}m...")
            
            plan, fraction = self.planner.plan_cartesian_path(waypoints, eef_step=0.01)
            
            if fraction > 0.9:
                print(f"✓ Path planned: {fraction*100:.1f}% achieved")
                print("  Executing...")
                success = self.planner.execute_plan(plan)
                if success:
                    print("✅ Square completed!")
                else:
                    print("❌ Execution failed")
            else:
                print(f"❌ Could only achieve {fraction*100:.1f}% of path")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def auto_demo(self):
        """Run automatic demonstration of all poses."""
        print("\n🤖 Auto Demo - Visiting All Predefined Poses")
        print("="*70)
        
        import time
        
        pose_sequence = ['home', 'forward', 'left', 'right', 'high', 'low', 'home']
        
        for pose_key in pose_sequence:
            print(f"\n→ Moving to: {self.predefined_poses[pose_key]['description']}")
            self.execute_predefined_pose(pose_key)
            time.sleep(1.0)
        
        print("\n✅ Auto demo complete!")
    
    def run_interactive(self):
        """Run the interactive menu loop."""
        
        while rclpy.ok():
            self.show_menu()
            
            try:
                choice = input("\nYour choice: ").strip().lower()
                
                if choice == 'q':
                    print("\n👋 Goodbye!")
                    break
                elif choice == '1':
                    self.execute_predefined_pose('home')
                elif choice == '2':
                    self.execute_predefined_pose('forward')
                elif choice == '3':
                    self.execute_predefined_pose('left')
                elif choice == '4':
                    self.execute_predefined_pose('right')
                elif choice == '5':
                    self.execute_predefined_pose('high')
                elif choice == '6':
                    self.execute_predefined_pose('low')
                elif choice == '7':
                    self.execute_predefined_pose('upright')
                elif choice == '8':
                    self.execute_custom_cartesian()
                elif choice == '9':
                    self.execute_custom_joints()
                elif choice == '10':
                    self.draw_circle()
                elif choice == '11':
                    self.draw_square()
                elif choice == '12':
                    self.auto_demo()
                elif choice == 'i':
                    self.get_current_state()
                elif choice == 'h':
                    self.show_workspace_limits()
                else:
                    print("❌ Invalid choice. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\nPress Enter to continue...")


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        demo = InteractiveSimPlanner()
        
        # Run ROS 2 spinning in background thread
        spin_thread = threading.Thread(target=rclpy.spin, args=(demo,), daemon=True)
        spin_thread.start()
        
        # Run interactive menu in main thread
        demo.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()


