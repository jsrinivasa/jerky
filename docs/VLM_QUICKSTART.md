# VLM Control Quick Start Guide

Get started with natural language control of Mobile ALOHA in 5 minutes!

## Quick Install

```bash
cd /home/aloha/interbotix_ws/src/aloha

# Install core dependencies
pip install pillow>=9.0.0

# Choose ONE of the following based on your preferred model:

# Option 1: OpenVLA (local, requires GPU)
pip install transformers>=4.40.0 accelerate

# Option 2: GPT-4V (API-based, requires API key)
pip install openai>=1.0.0

# Option 3: Claude (API-based, requires API key)
pip install anthropic>=0.18.0
```

## Test Without Models (Rule-Based)

The easiest way to test the system without downloading models or API keys:

```bash
# Make script executable
chmod +x scripts/natural_language_control.py

# Run test mode
python scripts/natural_language_control.py \
    --model test \
    --task "move forward" \
    --max-steps 20
```

This will move the robot in a simple pattern to verify the system works.

## Using OpenVLA (Recommended)

OpenVLA is the best option for real-time control:

```bash
# First run will download the model (~14GB)
python scripts/natural_language_control.py \
    --model openvla \
    --task "move forward and explore" \
    --max-steps 50

# If you have limited GPU memory, use quantization:
python scripts/natural_language_control.py \
    --model openvla \
    --load-8bit \
    --task "move forward"
```

## Using GPT-4V (Easiest to Start)

If you have an OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"

python scripts/natural_language_control.py \
    --model gpt4v \
    --task "explore the room" \
    --max-steps 30
```

## Interactive Mode

The most fun way to use the system:

```bash
python scripts/natural_language_control.py \
    --model openvla \
    --interactive
```

Then type commands:
```
Enter command > move forward 2 meters
Enter command > turn left 90 degrees
Enter command > explore the room
Enter command > quit
```

## Safety First! 🛡️

Before running:

1. ✅ Clear the area around the robot
2. ✅ Have E-stop ready
3. ✅ Start with `--enable-arms=False` (base only)
4. ✅ Use low `--max-steps` for first tests
5. ✅ Test simple commands before complex ones

## Example Commands

### Simple Navigation
```bash
# Move forward
python scripts/natural_language_control.py \
    --model test \
    --task "move forward"

# Turn in place
python scripts/natural_language_control.py \
    --model test \
    --task "turn left"

# Explore
python scripts/natural_language_control.py \
    --model openvla \
    --task "explore the room"
```

### With Vision Understanding (requires OpenVLA or API models)
```bash
# Task with reasoning
python scripts/natural_language_control.py \
    --model openvla \
    --task "find the red object" \
    --max-steps 100

# Complex navigation
python scripts/natural_language_control.py \
    --model gpt4v \
    --task "navigate to the table and stop"
```

## Troubleshooting

### "CUDA out of memory"
Use quantization:
```bash
python scripts/natural_language_control.py \
    --model openvla \
    --load-8bit  # or --load-4bit
```

### "No module named transformers"
Install OpenVLA dependencies:
```bash
pip install transformers accelerate
```

### "API key required"
Set environment variable:
```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Robot not moving
Check ROS2 setup:
```bash
ros2 topic list  # Should see /mobile_base/cmd_vel
ros2 topic echo /mobile_base/cmd_vel  # Should see messages when running
```

## Next Steps

1. Read full documentation: `docs/VLM_CONTROL.md`
2. Try interactive mode with different tasks
3. Fine-tune OpenVLA on your specific environment
4. Enable arm control (carefully!) with `--enable-arms`

## Performance Tips

- **Fast control**: Use OpenVLA with quantization
- **Best reasoning**: Use GPT-4V with lower frequency
- **No cost**: Use test mode for simple tasks
- **Real-time**: Set `--frequency 10.0` for OpenVLA

## Example Session

```bash
# Terminal 1: Start natural language control
cd /home/aloha/interbotix_ws/src/aloha
python scripts/natural_language_control.py --model openvla --interactive

# You'll see:
# Interactive Natural Language Control
# Enter natural language commands for the robot.
# Type 'quit' or 'exit' to stop.
#
# Enter command > move forward
# [Robot moves forward]
#
# Enter command > turn right 90 degrees
# [Robot turns]
#
# Enter command > explore
# [Robot explores autonomously]
#
# Enter command > quit
# Exiting...
```

## Help & Support

- Full docs: `docs/VLM_CONTROL.md`
- Mobile ALOHA docs: https://docs.trossenrobotics.com/aloha_docs/
- OpenVLA: https://github.com/openvla/openvla

Happy robot controlling! 🤖
