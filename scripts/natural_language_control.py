#!/usr/bin/env python3

"""
Natural Language Control for Mobile ALOHA

This script enables natural language control of the Mobile ALOHA robot
using Vision-Language-Action models. It supports multiple VLM backends:

1. LLaVA - LOCAL vision-language model (RECOMMENDED, no API needed)
2. OpenVLA - Open-source VLA model for robot manipulation  
3. GPT-4V/Claude - API-based vision-language models with reasoning
4. Rule-based - Simple fallback for testing

Usage:
    # Using LLaVA (LOCAL MODEL, RECOMMENDED - no API needed)
    python natural_language_control.py --model llava --task "explore the room"
    
    # Using OpenVLA (requires model download)
    python natural_language_control.py --model openvla --task "move forward and pick up the cup"
    
    # Using GPT-4V with API
    python natural_language_control.py --model gpt4v --api-key YOUR_KEY --task "explore the room"
    
    # Using Claude
    python natural_language_control.py --model claude --api-key YOUR_KEY --task "navigate to the table"
    
    # Interactive mode with local LLaVA
    python natural_language_control.py --model llava --interactive
    
    # Test mode (rule-based, no model required)
    python natural_language_control.py --model test --task "move forward"
"""

import argparse
import os
import sys
from typing import Optional

import rclpy
from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown


def create_controller(args: argparse.Namespace):
    """
    Create the appropriate VLM controller based on arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Initialized VLM controller
    """
    model_type = args.model.lower()
    
    if model_type == 'llava':
        from aloha.vlm_llava import LLaVAController
        
        print("Initializing LLaVA controller (local VLM, no API needed)...")
        controller = LLaVAController(
            model_path=args.model_path if hasattr(args, 'llava_path') else "llava-hf/llava-1.5-7b-hf",
            device=args.device,
            load_in_8bit=args.load_8bit,
            load_in_4bit=args.load_4bit,
            verbose_logging=args.verbose_model,
            enable_base=args.enable_base,
            enable_arms=args.enable_arms,
            control_frequency=args.frequency,
        )
        
    elif model_type == 'openvla':
        from aloha.vlm_openvla import OpenVLAController
        
        print("Initializing OpenVLA controller...")
        controller = OpenVLAController(
            model_path=args.model_path,
            device=args.device,
            load_in_8bit=args.load_8bit,
            load_in_4bit=args.load_4bit,
            verbose_logging=args.verbose_model,
            enable_base=args.enable_base,
            enable_arms=args.enable_arms,
            control_frequency=args.frequency,
        )
        
    elif model_type in ['gpt4v', 'gpt-4v', 'openai']:
        from aloha.vlm_reasoning import ReasoningVLMController
        
        api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: OpenAI API key required. Provide via --api-key or OPENAI_API_KEY env var")
            sys.exit(1)
        
        print("Initializing GPT-4V reasoning controller...")
        controller = ReasoningVLMController(
            api_key=api_key,
            model_name=args.model_name or "gpt-4-vision-preview",
            api_provider="openai",
            verbose_logging=args.verbose_model,
            enable_base=args.enable_base,
            enable_arms=args.enable_arms,
            control_frequency=args.frequency,
        )
        
    elif model_type == 'claude':
        from aloha.vlm_reasoning import ReasoningVLMController
        
        api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("Error: Anthropic API key required. Provide via --api-key or ANTHROPIC_API_KEY env var")
            sys.exit(1)
        
        print("Initializing Claude reasoning controller...")
        controller = ReasoningVLMController(
            api_key=api_key,
            model_name=args.model_name or "claude-3-opus-20240229",
            api_provider="anthropic",
            verbose_logging=args.verbose_model,
            enable_base=args.enable_base,
            enable_arms=args.enable_arms,
            control_frequency=args.frequency,
        )
        
    elif model_type == 'test' or model_type == 'rule-based':
        from aloha.vlm_reasoning import ReasoningVLMController
        
        print("Initializing rule-based test controller (no API required)...")
        controller = ReasoningVLMController(
            api_key=None,
            api_provider="local",
            verbose_logging=args.verbose_model,
            enable_base=args.enable_base,
            enable_arms=args.enable_arms,
            control_frequency=args.frequency,
        )
        
    else:
        print(f"Error: Unknown model type '{model_type}'")
        print("Supported models: llava, openvla, gpt4v, claude, test")
        sys.exit(1)
    
    return controller


def run_single_task(controller, task: str, args: argparse.Namespace):
    """
    Run a single task.
    
    Args:
        controller: VLM controller instance
        task: Task description
        args: Command line arguments
    """
    print(f"\nExecuting task: {task}")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        result = controller.run_episode(
            task_description=task,
            max_steps=args.max_steps,
            visualize=args.visualize,
            verbose=args.verbose,
        )
        
        print(f"\nTask completed in {result['steps']} steps")
        if result['done']:
            print("Status: SUCCESS ✓")
        else:
            print("Status: INCOMPLETE")
            
    except KeyboardInterrupt:
        print("\nTask interrupted by user")
        controller.stop()


def run_interactive(controller, args: argparse.Namespace):
    """
    Run in interactive mode, accepting commands from user.
    
    Args:
        controller: VLM controller instance
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print("Interactive Natural Language Control")
    print("=" * 60)
    print("\nEnter natural language commands for the robot.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'help' for examples.\n")
    
    while True:
        try:
            task = input("Enter command > ").strip()
            
            if not task:
                continue
            
            if task.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if task.lower() == 'help':
                print_help_examples()
                continue
            
            run_single_task(controller, task, args)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break
    
    controller.stop()


def print_help_examples():
    """Print example commands."""
    print("\nExample commands:")
    print("  - 'move forward 2 meters'")
    print("  - 'turn left 90 degrees'")
    print("  - 'explore the room'")
    print("  - 'go to the table'")
    print("  - 'pick up the red cup'")
    print("  - 'navigate around the obstacle'")
    print("  - 'return to starting position'")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Natural Language Control for Mobile ALOHA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['llava', 'openvla', 'gpt4v', 'claude', 'test'],
        default='llava',
        help='VLM model to use (llava=local model, best default)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='openvla/openvla-7b',
        help='Path to OpenVLA model (HuggingFace or local)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Specific model name for API providers'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for OpenAI/Anthropic (or use environment variable)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for model inference'
    )
    
    parser.add_argument(
        '--load-8bit',
        action='store_true',
        help='Load OpenVLA model in 8-bit quantization'
    )
    
    parser.add_argument(
        '--load-4bit',
        action='store_true',
        help='Load OpenVLA model in 4-bit quantization'
    )
    
    # Task specification
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='Natural language task description'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    # Control options
    parser.add_argument(
        '--enable-base',
        action='store_true',
        default=True,
        help='Enable mobile base control'
    )
    
    parser.add_argument(
        '--enable-arms',
        action='store_true',
        default=False,
        help='Enable arm control (WARNING: ensure arms are in safe position)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum steps per episode'
    )
    
    parser.add_argument(
        '--frequency',
        type=float,
        default=10.0,
        help='Control loop frequency in Hz'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Show visual feedback'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information'
    )
    
    parser.add_argument(
        '--verbose-model',
        action='store_true',
        default=True,
        help='Show detailed model input/output logging (OpenVLA)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.task:
        print("Error: Either --task or --interactive must be specified")
        parser.print_help()
        sys.exit(1)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create controller
        controller = create_controller(args)
        
        # Start robot systems
        robot_startup(controller.node)
        
        # Run task(s)
        if args.interactive:
            run_interactive(controller, args)
        else:
            run_single_task(controller, args.task, args)
        
        # Cleanup
        controller.shutdown()
        robot_shutdown(controller.node)
        
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    
    print("Natural language control terminated")


if __name__ == '__main__':
    main()
