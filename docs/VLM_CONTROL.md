# Natural Language Control for Mobile ALOHA

This system enables natural language control of the Mobile ALOHA robot using Vision-Language-Action (VLA) models. The system implements closed-loop reasoning, allowing the robot to observe, reason, plan, and execute tasks based on natural language commands.

## Features

- **Multiple VLM Backends**: Support for OpenVLA, GPT-4V, Claude, and rule-based fallback
- **Closed-Loop Reasoning**: The robot continuously observes and adapts its behavior
- **Multi-Camera Integration**: Processes visual input from multiple camera views
- **Mobile Base & Arm Control**: Coordinates movement and manipulation
- **Interactive Mode**: Real-time command interface
- **Extensible Architecture**: Easy to add new VLM models

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Natural Language Command                │
│         "Pick up the red cup"                   │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│         Vision-Language Model                   │
│  ┌──────────────┐  ┌──────────────┐            │
│  │   OpenVLA    │  │   GPT-4V     │            │
│  │   7B model   │  │   Claude     │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│         Closed-Loop Reasoning                   │
│  1. Observe (cameras + state)                   │
│  2. Reason (understand situation)               │
│  3. Plan (decide action)                        │
│  4. Act (execute)                               │
│  5. Reflect (evaluate progress)                 │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│         Robot Control System                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Mobile   │  │ Left Arm │  │ Right    │     │
│  │ Base     │  │ & Gripper│  │ Arm      │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
```

## Supported Models

### 1. OpenVLA (Recommended)

OpenVLA is an open-source 7B parameter Vision-Language-Action model specifically trained for robot manipulation tasks.

**Pros:**
- Open source and free
- Trained on robot manipulation data
- Outputs direct robot actions
- Can run locally

**Cons:**
- Requires GPU with ~16GB VRAM (can use 8-bit/4-bit quantization)
- Initial model download (~14GB)

**Setup:**
```bash
pip install transformers accelerate bitsandbytes
```

**Usage:**
```bash
python scripts/natural_language_control.py \
    --model openvla \
    --task "move forward and explore the room" \
    --device cuda
```

**Quantization (for lower memory):**
```bash
# 8-bit (requires ~8GB VRAM)
python scripts/natural_language_control.py --model openvla --load-8bit

# 4-bit (requires ~4GB VRAM)
python scripts/natural_language_control.py --model openvla --load-4bit
```

### 2. GPT-4V (OpenAI)

Uses OpenAI's GPT-4 with vision for closed-loop reasoning and control.

**Pros:**
- Excellent reasoning capabilities
- No local GPU required
- Easy to get started

**Cons:**
- Requires API key and credits
- Network latency
- Cost per API call

**Setup:**
```bash
pip install openai
export OPENAI_API_KEY="your-api-key-here"
```

**Usage:**
```bash
python scripts/natural_language_control.py \
    --model gpt4v \
    --api-key YOUR_KEY \
    --task "navigate to the table and pick up the cup"
```

### 3. Claude (Anthropic)

Uses Anthropic's Claude vision models for reasoning.

**Pros:**
- Strong reasoning capabilities
- Good at complex instructions
- No local GPU required

**Cons:**
- Requires API key
- Network latency
- Cost per API call

**Setup:**
```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Usage:**
```bash
python scripts/natural_language_control.py \
    --model claude \
    --api-key YOUR_KEY \
    --task "explore the room and identify objects"
```

### 4. Test Mode (Rule-Based)

A simple rule-based fallback for testing the system without any VLM.

**Usage:**
```bash
python scripts/natural_language_control.py \
    --model test \
    --task "move forward"
```

## Installation

### Prerequisites

1. Mobile ALOHA robot with ROS2 setup
2. Python 3.8+
3. CUDA-capable GPU (for OpenVLA)

### Install Dependencies

```bash
cd /home/aloha/interbotix_ws/src/aloha

# Core dependencies
pip install -r requirements.txt

# For OpenVLA
pip install transformers>=4.40.0 accelerate bitsandbytes

# For GPT-4V
pip install openai>=1.0.0

# For Claude
pip install anthropic>=0.18.0
```

## Usage

### Basic Usage

```bash
# Single task with OpenVLA
python scripts/natural_language_control.py \
    --model openvla \
    --task "move forward 2 meters"

# Single task with GPT-4V
python scripts/natural_language_control.py \
    --model gpt4v \
    --api-key YOUR_KEY \
    --task "explore the room"
```

### Interactive Mode

```bash
# Interactive mode lets you issue commands in real-time
python scripts/natural_language_control.py \
    --model openvla \
    --interactive

# Then type commands:
# > move forward
# > turn left 90 degrees
# > pick up the cup
# > quit
```

### Advanced Options

```bash
# Enable arm control (WARNING: ensure safe position)
python scripts/natural_language_control.py \
    --model openvla \
    --enable-arms \
    --task "pick up the object"

# Adjust control frequency
python scripts/natural_language_control.py \
    --model openvla \
    --frequency 5.0 \
    --task "move slowly forward"

# Longer episodes
python scripts/natural_language_control.py \
    --model gpt4v \
    --max-steps 500 \
    --task "explore the entire room"
```

## Example Commands

### Navigation
- "move forward 2 meters"
- "turn left 90 degrees"
- "rotate right and then move forward"
- "navigate around the obstacle"
- "return to starting position"

### Exploration
- "explore the room"
- "scan the environment"
- "find the red object"

### Manipulation (requires --enable-arms)
- "pick up the cup"
- "grasp the red block"
- "place the object on the table"
- "open the drawer"

### Complex Tasks
- "go to the kitchen and pick up the cup"
- "explore the room and identify all objects"
- "navigate to the table, pick up the red cube, and place it in the box"

## How It Works

### Closed-Loop Control

The system implements a closed-loop control architecture:

1. **Observation**: 
   - Captures images from multiple cameras (cam_high, cam_left_wrist, cam_right_wrist)
   - Reads robot joint states and gripper positions
   - Tracks execution history

2. **Reasoning**:
   - VLM analyzes the visual scene
   - Understands the task goal
   - Assesses current progress
   - Identifies what needs to be done next

3. **Planning**:
   - Determines the next action (base movement, arm motion, grasp, etc.)
   - Generates confidence scores
   - Checks if task is complete

4. **Execution**:
   - Converts high-level actions to low-level robot commands
   - Publishes velocity commands to mobile base
   - Commands joint positions to arms
   - Controls gripper states

5. **Reflection**:
   - Evaluates action outcomes
   - Maintains history for context
   - Adapts based on observations

### Action Space

The VLM outputs actions in the following space:

- **Base Actions**: `[linear_velocity, angular_velocity]`
  - Linear: -0.7 to 0.7 m/s (forward/backward)
  - Angular: -1.0 to 1.0 rad/s (turn left/right)

- **Arm Actions**: `[6 joint angles]` for each arm
  - Shoulder, elbow, wrist, etc.

- **Gripper Actions**: `[gripper_position]`
  - Open: ~1.62
  - Close: ~0.62

## Extending the System

### Adding a New VLM Backend

1. Create a new file in `aloha/vlm_<name>.py`
2. Inherit from `VLMController` base class
3. Implement `predict_action()` and `reason_about_task()` methods
4. Add model selection in `natural_language_control.py`

Example:

```python
from aloha.vlm_controller import VLMController, VLMAction

class MyVLMController(VLMController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model
        
    def predict_action(self, observation, task_description, history):
        # Your model inference
        return VLMAction(...)
        
    def reason_about_task(self, observation, task_description, history):
        # Your reasoning logic
        return "reasoning text"
```

### Fine-Tuning for Your Tasks

To fine-tune OpenVLA on your specific tasks:

1. Collect demonstrations using `dual_side_teleop.py`
2. Record episodes with camera views
3. Fine-tune OpenVLA on your dataset
4. Load fine-tuned checkpoint:

```bash
python scripts/natural_language_control.py \
    --model openvla \
    --model-path /path/to/finetuned/model \
    --task "your custom task"
```

## Troubleshooting

### Common Issues

**OpenVLA model won't load:**
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Try 8-bit or 4-bit quantization to reduce memory
- Ensure transformers version >= 4.40.0

**API rate limits:**
- Use OpenVLA for high-frequency control
- Reduce control frequency: `--frequency 1.0`
- Implement response caching for repeated observations

**Robot not moving:**
- Check ROS2 topics: `ros2 topic list`
- Verify base controller: `ros2 topic echo /mobile_base/cmd_vel`
- Ensure robot is powered and initialized

**Poor performance:**
- Fine-tune on your specific environment
- Adjust confidence thresholds
- Provide more detailed task descriptions

## Safety Considerations

⚠️ **Important Safety Notes:**

1. **Start with base-only control**: Test with `--enable-arms=False` first
2. **Clear workspace**: Ensure area around robot is clear
3. **Emergency stop**: Keep E-stop accessible at all times
4. **Supervision**: Always supervise the robot during operation
5. **Test in simulation first**: If possible, test commands in simulation
6. **Gradual complexity**: Start with simple tasks before complex ones

## Performance Tips

1. **Use appropriate model**: OpenVLA for real-time, GPT-4V for complex reasoning
2. **Optimize frequency**: Lower frequency (1-5 Hz) for VLM inference
3. **Cache observations**: Reduce API calls for similar views
4. **Fine-tune models**: Better performance on your specific tasks
5. **Use quantization**: 8-bit/4-bit for memory-constrained systems

## Research & References

### OpenVLA
- Paper: "OpenVLA: An Open-Source Vision-Language-Action Model"
- Repository: https://github.com/openvla/openvla
- ArXiv: https://arxiv.org/abs/2406.09246

### Mobile ALOHA
- Website: https://mobile-aloha.github.io/
- Paper: "Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation"

### Related Work
- RT-1, RT-2 (Robotics Transformer)
- PaLM-E (Embodied Language Model)
- Octo (Open-Source Generalist Robot Policy)

## Contributing

To contribute new VLM backends or improvements:

1. Follow the existing code structure in `aloha/vlm_*.py`
2. Add comprehensive docstrings
3. Include usage examples
4. Test on real hardware
5. Submit detailed documentation

## Support

For issues and questions:
- Mobile ALOHA Docs: https://docs.trossenrobotics.com/aloha_docs/
- OpenVLA GitHub: https://github.com/openvla/openvla
- Create an issue in your project repository

## License

This software follows the same license as the Mobile ALOHA repository.
