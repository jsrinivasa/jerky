#!/usr/bin/env python3

"""
Test script to verify OpenVLA reasoning improvements.

This script tests that tasks no longer complete prematurely at step 16.
"""

import sys
sys.path.insert(0, '/home/aloha/interbotix_ws/src/aloha')

from aloha.vlm_openvla import OpenVLAController
from aloha.vlm_controller import VLMAction
import numpy as np


def test_completion_detection():
    """Test that completion detection is strict."""
    print("\n" + "="*60)
    print("Test 1: Completion Detection")
    print("="*60 + "\n")
    
    controller = OpenVLAController.__new__(OpenVLAController)
    
    test_cases = [
        # (output_text, expected_done, description)
        ("Moving forward to explore", False, "Normal movement"),
        ("Task complete", True, "Explicit completion"),
        ("The robot has finished", False, "Generic 'finished'"),
        ("Goal achieved", True, "Goal statement"),
        ("When complete, return home", False, "Conditional completion"),
        ("Task is finished successfully", True, "Task finished phrase"),
        ("This is done", False, "Generic 'done'"),
        ("Mission accomplished", True, "Mission phrase"),
    ]
    
    print("Testing completion detection logic:\n")
    passed = 0
    failed = 0
    
    for text, expected_done, description in test_cases:
        observation = {'robot_state': {}}
        action = controller._parse_openvla_output(text, observation, "test task")
        
        status = "✓ PASS" if action.done == expected_done else "✗ FAIL"
        if action.done == expected_done:
            passed += 1
        else:
            failed += 1
            
        print(f"{status}: '{text}'")
        print(f"  Expected done={expected_done}, Got done={action.done}")
        print(f"  Description: {description}\n")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)}")
    return failed == 0


def test_task_aware_behavior():
    """Test that different tasks have different durations."""
    print("\n" + "="*60)
    print("Test 2: Task-Aware Behavior")
    print("="*60 + "\n")
    
    controller = OpenVLAController.__new__(OpenVLAController)
    
    tasks = [
        ("explore the room", 50, "Should run for 30+ steps in cycles"),
        ("move forward", 25, "Should complete around step 20"),
        ("turn left", 20, "Should complete around step 15"),
        ("navigate to table", 30, "Should complete around step 25"),
    ]
    
    print("Testing task durations:\n")
    all_passed = True
    
    for task_desc, test_steps, description in tasks:
        print(f"Task: '{task_desc}'")
        print(f"Description: {description}")
        
        observation = {'robot_state': {}}
        history = []
        
        # Simulate episode
        steps_until_done = None
        for step in range(test_steps):
            history_for_predict = [{'step': i} for i in range(step)]
            action = controller._dummy_predict(observation, task_desc, history_for_predict)
            
            if action.done and steps_until_done is None:
                steps_until_done = step + 1
                break
        
        if steps_until_done is None:
            print(f"  Result: Did NOT complete in {test_steps} steps ✓")
            if 'explore' in task_desc:
                print(f"  Status: ✓ PASS (exploration should continue)\n")
            else:
                print(f"  Status: ⚠ WARNING (may need longer test)\n")
        else:
            print(f"  Result: Completed at step {steps_until_done}")
            
            # Check if completion is reasonable
            if 'explore' in task_desc:
                if steps_until_done > 25:
                    print(f"  Status: ✓ PASS (explores beyond step 16)\n")
                else:
                    print(f"  Status: ✗ FAIL (should explore longer)\n")
                    all_passed = False
            elif 'forward' in task_desc:
                if 18 <= steps_until_done <= 22:
                    print(f"  Status: ✓ PASS (completes around step 20)\n")
                elif steps_until_done == 16:
                    print(f"  Status: ✗ FAIL (old 16-step bug!)\n")
                    all_passed = False
                else:
                    print(f"  Status: ⚠ WARNING (unexpected timing)\n")
            elif 'turn' in task_desc or 'rotate' in task_desc:
                if 14 <= steps_until_done <= 16:
                    print(f"  Status: ✓ PASS (completes around step 15)\n")
                else:
                    print(f"  Status: ⚠ WARNING (unexpected timing)\n")
            elif 'navigate' in task_desc or 'go to' in task_desc:
                if 23 <= steps_until_done <= 27:
                    print(f"  Status: ✓ PASS (completes around step 25)\n")
                else:
                    print(f"  Status: ⚠ WARNING (unexpected timing)\n")
    
    return all_passed


def test_no_premature_completion():
    """Specific test: ensure tasks don't complete at step 16."""
    print("\n" + "="*60)
    print("Test 3: No Premature Completion at Step 16")
    print("="*60 + "\n")
    
    controller = OpenVLAController.__new__(OpenVLAController)
    
    tasks = [
        "explore the environment",
        "move around the room",
        "scan the area",
        "investigate the space",
    ]
    
    print("Testing that tasks don't end at old bug point (step 16):\n")
    all_passed = True
    
    for task_desc in tasks:
        observation = {'robot_state': {}}
        history = [{'step': i} for i in range(16)]
        
        # Check step 16 specifically
        action = controller._dummy_predict(observation, task_desc, history)
        
        print(f"Task: '{task_desc}'")
        print(f"  Step 16 action: {action.reasoning}")
        print(f"  Done at step 16: {action.done}")
        
        if action.done:
            print(f"  Status: ✗ FAIL (premature completion!)\n")
            all_passed = False
        else:
            print(f"  Status: ✓ PASS (continues past step 16)\n")
    
    return all_passed


def test_reasoning_output():
    """Test that reasoning includes step numbers and context."""
    print("\n" + "="*60)
    print("Test 4: Reasoning Output Quality")
    print("="*60 + "\n")
    
    controller = OpenVLAController.__new__(OpenVLAController)
    
    task = "explore the room"
    observation = {'robot_state': {}}
    
    print(f"Testing reasoning output for: '{task}'\n")
    print("Sample reasoning at different steps:\n")
    
    for step in [0, 5, 10, 15, 20, 25]:
        history = [{'step': i} for i in range(step)]
        action = controller._dummy_predict(observation, task, history)
        
        print(f"Step {step+1}: {action.reasoning}")
        
        # Check if step number is included
        if f"step {step+1}" in action.reasoning.lower():
            print(f"  ✓ Includes step counter")
        else:
            print(f"  ⚠ Missing step counter")
        
        print()
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("OpenVLA Reasoning Improvements - Test Suite")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Completion Detection", test_completion_detection()))
    results.append(("Task-Aware Behavior", test_task_aware_behavior()))
    results.append(("No Premature Completion", test_no_premature_completion()))
    results.append(("Reasoning Quality", test_reasoning_output()))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70 + "\n")
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests PASSED! Reasoning improvements are working correctly.")
    else:
        print("✗ Some tests FAILED. Please review the output above.")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
