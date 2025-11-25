# Cisco Research - Mobile ALOHA

> Forked from Interbotix ALOHA repository

Project Websites:

* [ALOHA](https://tonyzhaozh.github.io/aloha/)
* [Mobile ALOHA](https://mobile-aloha.github.io/)

Trossen Robotics Documentation: https://docs.trossenrobotics.com/aloha_docs/

This codebase is forked from the [Mobile ALOHA repo](https://github.com/MarkFzp/mobile-aloha), and contains teleoperation and dataset collection and evaluation tools for the Stationary and Mobile ALOHA kits available from Trossen Robotics.

To get started with your ALOHA kit, follow the [ALOHA Getting Started Documentation](https://docs.trossenrobotics.com/aloha_docs/getting_started.html).

To train imitation learning algorithms, you would also need to install:

* [ACT for Stationary ALOHA](https://github.com/Interbotix/act).
* [ACT++ for Mobile ALOHA](https://github.com/Interbotix/act-plus-plus)

# Structure
- [``aloha``](./aloha/): Python package providing useful classes and constants for teleoperation and dataset collection.
- [``config``](./config/): a config for each robot, designating the port they should bind to, more details in quick start guide.
- [``launch``](./launch): a ROS 2 launch file for all cameras and manipulators.
- [``scripts``](./scripts/): Python scripts for teleop and data collection
- [``docs``](./docs/): Documentation for VLM control and other features

# Natural Language Control (NEW!)

Control your Mobile ALOHA robot using natural language commands powered by Vision-Language-Action models!

**Quick Start:**
```bash
# Test mode (no model required)
python scripts/natural_language_control.py --model test --task "move forward"

# With LLaVA (LOCAL MODEL - RECOMMENDED, no API needed)
python scripts/natural_language_control.py --model llava --task "explore the room"

# Interactive mode with local model
python scripts/natural_language_control.py --model llava --interactive
```

**Features:**
- 🤖 Multiple VLM backends: LLaVA (local), OpenVLA, GPT-4V, Claude
- 🔄 Closed-loop reasoning: observe, reason, plan, act
- 📹 Multi-camera vision integration
- 🎮 Interactive command interface
- 🧠 Autonomous task execution
- 🔓 No API keys required (with LLaVA)

**Documentation:**
- Quick Start: [docs/VLM_QUICKSTART.md](./docs/VLM_QUICKSTART.md)
- Full Guide: [docs/VLM_CONTROL.md](./docs/VLM_CONTROL.md)
- Demo Script: `scripts/vlm_demo.py`

**Supported Models:**
- **LLaVA**: LOCAL 7B vision-language model (RECOMMENDED, no API needed)
- **OpenVLA**: Open-source 7B VLA model for robot manipulation
- **GPT-4V**: OpenAI's vision model with reasoning
- **Claude**: Anthropic's vision model
- **Rule-based**: Simple fallback for testing
