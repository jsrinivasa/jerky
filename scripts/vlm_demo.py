#!/usr/bin/env python3

"""
VLM Control Demo Script

This script demonstrates how to use the VLM controller programmatically.
Use this as a template for building your own VLM-based robot applications.
"""

import rclpy
from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown

# Import different VLM controllers
from aloha.vlm_openvla import OpenVLAController
from aloha.vlm_reasoning import ReasoningVLMController


def demo_basic_navigation():
    """
    Demo 1: Basic navigation using rule-based controller.
    
    This is the simplest demo - no model required, just rule-based behavior.
    """
    print("\n" + "="*60)
    print("Demo 1: Basic Navigation (Rule-Based)")
    print("="*60 + "\n")
    
    # Initialize ROS2
    rclpy.init()
    
    # Create controller (no API key = rule-based mode)
    controller = ReasoningVLMController(
        api_key=None,
        enable_base=True,
        enable_arms=False,
        control_frequency=10.0,
    )
    
    # Start robot systems
    robot_startup(controller.node)
    
    # Run a simple task
    controller.run_episode(
        task_description="Move forward and explore",
        max_steps=20,
        visualize=True,
        verbose=True,
    )
    
    # Cleanup
    controller.shutdown()
    robot_shutdown(controller.node)
    rclpy.shutdown()
    
    print("\nDemo 1 complete!")


def demo_openvla_control():
    """
    Demo 2: Using OpenVLA for intelligent control.
    
    Requires: OpenVLA model and GPU
    """
    print("\n" + "="*60)
    print("Demo 2: OpenVLA Control")
    print("="*60 + "\n")
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create OpenVLA controller
        controller = OpenVLAController(
            model_path="openvla/openvla-7b",
            device="cuda",
            load_in_8bit=True,  # Use 8-bit quantization
            enable_base=True,
            enable_arms=False,
            control_frequency=5.0,  # Slower for model inference
        )
        
        # Start robot systems
        robot_startup(controller.node)
        
        # Define a sequence of tasks
        tasks = [
            "Move forward slowly",
            "Turn right 90 degrees",
            "Move forward again",
            "Turn around and return",
        ]
        
        for i, task in enumerate(tasks, 1):
            print(f"\n--- Task {i}/{len(tasks)}: {task} ---")
            
            result = controller.run_episode(
                task_description=task,
                max_steps=30,
                visualize=True,
                verbose=True,
            )
            
            if not result['done']:
                print(f"Task {i} incomplete, continuing anyway...")
        
        # Cleanup
        controller.shutdown()
        robot_shutdown(controller.node)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OpenVLA is installed and you have a GPU")
    
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    
    print("\nDemo 2 complete!")


def demo_interactive_reasoning():
    """
    Demo 3: Interactive control with reasoning (requires API key).
    
    This demo shows how to use GPT-4V or Claude for more advanced reasoning.
    """
    print("\n" + "="*60)
    print("Demo 3: Interactive Reasoning")
    print("="*60 + "\n")
    
    import os
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Skipping demo 3: No OPENAI_API_KEY found")
        print("Set OPENAI_API_KEY environment variable to run this demo")
        return
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create reasoning controller
        controller = ReasoningVLMController(
            api_key=api_key,
            model_name="gpt-4-vision-preview",
            api_provider="openai",
            enable_base=True,
            enable_arms=False,
            control_frequency=2.0,  # Lower frequency for API calls
        )
        
        # Start robot systems
        robot_startup(controller.node)
        
        # Run with reasoning
        print("This demo uses GPT-4V to reason about the environment")
        print("and plan actions accordingly.\n")
        
        controller.run_episode(
            task_description="Explore the room and describe what you see",
            max_steps=20,
            visualize=True,
            verbose=True,
        )
        
        # Cleanup
        controller.shutdown()
        robot_shutdown(controller.node)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    
    print("\nDemo 3 complete!")


def demo_custom_task():
    """
    Demo 4: Custom task with observation monitoring.
    
    Shows how to access observations and implement custom logic.
    """
    print("\n" + "="*60)
    print("Demo 4: Custom Task with Monitoring")
    print("="*60 + "\n")
    
    # Initialize ROS2
    rclpy.init()
    
    # Create controller
    controller = ReasoningVLMController(
        api_key=None,
        enable_base=True,
        enable_arms=False,
        control_frequency=10.0,
    )
    
    # Start robot systems
    robot_startup(controller.node)
    
    # Custom task loop with observation monitoring
    task = "Move in a square pattern"
    print(f"Task: {task}\n")
    
    # Define square pattern
    movements = [
        ("Move forward", 15),
        ("Turn left 90 degrees", 10),
        ("Move forward", 15),
        ("Turn left 90 degrees", 10),
        ("Move forward", 15),
        ("Turn left 90 degrees", 10),
        ("Move forward", 15),
        ("Turn left 90 degrees", 10),
    ]
    
    for i, (movement, steps) in enumerate(movements, 1):
        print(f"\n--- Segment {i}/{len(movements)}: {movement} ---")
        
        result = controller.run_episode(
            task_description=movement,
            max_steps=steps,
            visualize=True,
            verbose=False,  # Less verbose for this demo
        )
        
        # Check result
        print(f"Completed in {result['steps']} steps")
    
    # Cleanup
    controller.stop()
    controller.shutdown()
    robot_shutdown(controller.node)
    rclpy.shutdown()
    
    print("\nDemo 4 complete! Square pattern executed.")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("VLM Control Demo Suite")
    print("="*60)
    print("\nThis script demonstrates different ways to use VLM control.")
    print("Demos will run sequentially. Press Ctrl+C to skip.\n")
    
    demos = [
        ("Basic Navigation (Rule-Based)", demo_basic_navigation),
        ("OpenVLA Control", demo_openvla_control),
        ("Interactive Reasoning (GPT-4V)", demo_interactive_reasoning),
        ("Custom Square Pattern", demo_custom_task),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'='*60}")
        print(f"Running Demo {i}/{len(demos)}: {name}")
        print(f"{'='*60}")
        
        response = input(f"\nRun this demo? (y/n/q): ").strip().lower()
        
        if response == 'q':
            print("Quitting demo suite")
            break
        elif response == 'n':
            print(f"Skipping demo {i}")
            continue
        
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            continue
        except Exception as e:
            print(f"\nDemo failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        input("\nPress Enter to continue to next demo...")
    
    print("\n" + "="*60)
    print("Demo suite complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Run specific demo
        demo_num = sys.argv[1]
        if demo_num == '1':
            demo_basic_navigation()
        elif demo_num == '2':
            demo_openvla_control()
        elif demo_num == '3':
            demo_interactive_reasoning()
        elif demo_num == '4':
            demo_custom_task()
        else:
            print(f"Unknown demo: {demo_num}")
            print("Usage: python vlm_demo.py [1|2|3|4]")
    else:
        # Run all demos interactively
        main()
