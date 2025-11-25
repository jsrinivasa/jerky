# VLM Control System - Files Overview

This document provides an overview of all files added for the natural language control system.

## Core Modules (`aloha/`)

### `vlm_controller.py`
**Purpose**: Base class for all VLM controllers

**Key Components:**
- `VLMAction`: Data class representing robot actions (base, arms, grippers)
- `VLMController`: Abstract base class providing:
  - Camera integration via `ImageRecorder`
  - Robot state management
  - Action execution interface
  - Episode management with closed-loop control
  - Observation collection from multiple cameras

**Usage**: Inherit from this class to create custom VLM controllers

---

### `vlm_openvla.py`
**Purpose**: OpenVLA model integration for Mobile ALOHA

**Key Components:**
- `OpenVLAController`: Implements OpenVLA-based control
  - Loads 7B parameter vision-language-action model
  - Supports 8-bit/4-bit quantization for lower memory
  - Converts model outputs to robot actions
  - Includes dummy mode for testing without model

- `OpenVLAFineTunedController`: Variant for fine-tuned models
  - Loads Mobile ALOHA-specific fine-tuned weights
  - Optimized action parsing for ALOHA action space

**Models Supported:**
- `openvla/openvla-7b`: Base OpenVLA model
- `openvla/openvla-7b-mobile-aloha`: Fine-tuned variant (if available)
- Local checkpoints

**Requirements:**
- transformers >= 4.40.0
- accelerate
- bitsandbytes (for quantization)
- CUDA-capable GPU (or CPU with reduced performance)

---

### `vlm_reasoning.py`
**Purpose**: Closed-loop reasoning controller using API-based VLMs

**Key Components:**
- `ReasoningVLMController`: Implements reasoning loop
  - Supports GPT-4V (OpenAI) and Claude (Anthropic)
  - Structured reasoning: observe → reason → plan → act → reflect
  - JSON-based action parsing
  - Fallback rule-based mode when no API available

**Reasoning Process:**
1. Encode camera images to base64
2. Create detailed prompt with task and context
3. Call VLM API for reasoning
4. Parse response (JSON or text)
5. Convert to robot actions

**API Providers:**
- OpenAI (GPT-4V)
- Anthropic (Claude)
- Local (rule-based fallback)

**Requirements:**
- openai >= 1.0.0 (for GPT-4V)
- anthropic >= 0.18.0 (for Claude)

---

## Scripts (`scripts/`)

### `natural_language_control.py`
**Purpose**: Main command-line interface for natural language control

**Features:**
- Model selection (openvla, gpt4v, claude, test)
- Single task execution
- Interactive mode
- Configurable control parameters
- Safety options

**Usage Examples:**
```bash
# Single task
python scripts/natural_language_control.py --model openvla --task "explore"

# Interactive
python scripts/natural_language_control.py --model gpt4v --interactive

# Test mode
python scripts/natural_language_control.py --model test --task "move forward"
```

**Arguments:**
- `--model`: Choose VLM backend
- `--task`: Task description (single task mode)
- `--interactive`: Enable interactive mode
- `--enable-base`: Enable/disable mobile base
- `--enable-arms`: Enable/disable arm control
- `--max-steps`: Maximum episode length
- `--frequency`: Control loop frequency
- `--api-key`: API key for cloud models
- Quantization flags for OpenVLA

---

### `vlm_demo.py`
**Purpose**: Demonstration script showing programmatic usage

**Demos:**
1. **Basic Navigation**: Rule-based control (no model)
2. **OpenVLA Control**: Using OpenVLA for intelligent behavior
3. **Interactive Reasoning**: GPT-4V with advanced reasoning
4. **Custom Task**: Square pattern with observation monitoring

**Usage:**
```bash
# Run all demos interactively
python scripts/vlm_demo.py

# Run specific demo
python scripts/vlm_demo.py 1  # Basic navigation
python scripts/vlm_demo.py 2  # OpenVLA
python scripts/vlm_demo.py 3  # GPT-4V reasoning
python scripts/vlm_demo.py 4  # Custom pattern
```

---

## Documentation (`docs/`)

### `VLM_CONTROL.md`
**Purpose**: Comprehensive documentation for the VLM control system

**Sections:**
- Architecture overview with diagrams
- Detailed model comparisons
- Installation instructions
- Usage examples
- API reference
- Extending the system
- Troubleshooting
- Performance tips
- Safety considerations
- Research references

**Target Audience**: Users and developers

---

### `VLM_QUICKSTART.md`
**Purpose**: Fast-track guide to get started in 5 minutes

**Contents:**
- Minimal installation steps
- Quick test without models
- Getting started with each model type
- Common troubleshooting
- Example commands
- Next steps

**Target Audience**: New users wanting to try the system quickly

---

### `VLM_FILES_OVERVIEW.md`
**Purpose**: This file - overview of all components

---

## Configuration (`config/`)

### `vlm_config.yaml`
**Purpose**: Configuration file for VLM control settings

**Sections:**
- Model configuration (type, path, device, quantization)
- Control settings (enable base/arms, frequency, max steps)
- Visualization options
- API configuration (keys, model names)
- Safety settings (velocity limits, confidence thresholds)
- Camera configuration
- Task presets

**Usage**: Reference for creating custom configurations

---

## Updated Files

### `requirements.txt`
**Changes**: Added optional VLM dependencies with comments

**New Dependencies:**
- pillow >= 9.0.0 (core)
- transformers >= 4.40.0 (OpenVLA, commented)
- accelerate (OpenVLA, commented)
- bitsandbytes (quantization, commented)
- openai >= 1.0.0 (GPT-4V, commented)
- anthropic >= 0.18.0 (Claude, commented)

---

### `README.md`
**Changes**: Added "Natural Language Control" section

**New Content:**
- Overview of VLM control features
- Quick start commands
- Links to documentation
- List of supported models

---

## File Structure

```
aloha/
├── aloha/
│   ├── vlm_controller.py       # Base controller class
│   ├── vlm_openvla.py          # OpenVLA integration
│   └── vlm_reasoning.py        # Reasoning controller
├── scripts/
│   ├── natural_language_control.py  # Main CLI script
│   └── vlm_demo.py                  # Demo script
├── docs/
│   ├── VLM_CONTROL.md          # Full documentation
│   ├── VLM_QUICKSTART.md       # Quick start guide
│   └── VLM_FILES_OVERVIEW.md   # This file
├── config/
│   └── vlm_config.yaml         # Configuration template
├── requirements.txt            # Updated with VLM deps
└── README.md                   # Updated with VLM info
```

## Dependencies Graph

```
natural_language_control.py
    ├─> vlm_controller.py (base)
    ├─> vlm_openvla.py
    │   ├─> transformers
    │   ├─> torch
    │   └─> accelerate
    └─> vlm_reasoning.py
        ├─> openai (optional)
        └─> anthropic (optional)

All depend on:
    ├─> ROS2 (rclpy, geometry_msgs)
    ├─> interbotix packages
    ├─> aloha.robot_utils
    └─> aloha.constants
```

## Model Comparison

| Model | Type | Requirements | Latency | Best For |
|-------|------|--------------|---------|----------|
| OpenVLA | Local | GPU, 16GB VRAM | ~100ms | Real-time control |
| OpenVLA-8bit | Local | GPU, 8GB VRAM | ~150ms | Memory-constrained systems |
| GPT-4V | API | API key | ~1-2s | Complex reasoning |
| Claude | API | API key | ~1-2s | Detailed understanding |
| Rule-based | Local | None | <10ms | Testing, simple tasks |

## Quick Reference

### Starting the System
```bash
# Fastest way to test
python scripts/natural_language_control.py --model test --task "move forward"

# Best performance (requires GPU)
python scripts/natural_language_control.py --model openvla --task "explore"

# Best reasoning (requires API)
python scripts/natural_language_control.py --model gpt4v --api-key KEY --task "complex task"
```

### Interactive Mode
```bash
python scripts/natural_language_control.py --model openvla --interactive
```

### Programmatic Usage
```python
from aloha.vlm_openvla import OpenVLAController

controller = OpenVLAController(enable_base=True, enable_arms=False)
controller.run_episode("move forward", max_steps=50)
controller.shutdown()
```

## Safety Notes

⚠️ **Before Using:**
1. Clear workspace around robot
2. Have E-stop accessible
3. Start with `--enable-arms=False`
4. Test simple commands first
5. Monitor robot at all times

## Next Steps

1. Read [VLM_QUICKSTART.md](VLM_QUICKSTART.md) to get started
2. Try test mode: `--model test`
3. Install OpenVLA for real control
4. Read [VLM_CONTROL.md](VLM_CONTROL.md) for advanced usage
5. Explore [vlm_demo.py](../scripts/vlm_demo.py) examples

## Support

- Issues: Create GitHub issue
- Documentation: See `docs/` folder
- Mobile ALOHA: https://docs.trossenrobotics.com/aloha_docs/
- OpenVLA: https://github.com/openvla/openvla

---

**Created**: November 2024  
**Version**: 1.0  
**Status**: Ready for use ✅
