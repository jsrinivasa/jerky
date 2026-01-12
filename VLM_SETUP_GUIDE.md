# ALOHA VLM Environment Setup Guide

This guide explains how to set up and use the natural language control with LLaVA for Mobile ALOHA.

## Quick Setup

The conda environment `aloha_vlm` is being created with all necessary dependencies including:
- Python 3.10
- PyTorch with CUDA support
- Transformers (for LLaVA)
- Accelerate (for efficient model loading)
- Bitsandbytes (for 4-bit/8-bit quantization to save memory)
- All ALOHA dependencies

## After Environment Creation

### 1. Activate the environment
```bash
conda activate aloha_vlm
```

### 2. Run natural language control in interactive mode
```bash
cd /home/aloha/interbotix_ws/src/aloha
python scripts/natural_language_control.py --model llava --interactive
```

### 3. First run notes
- The first time you run, it will download the LLaVA model (~14GB)
- This download only happens once and is cached
- Model download location: `~/.cache/huggingface/hub/`

## Usage Examples

### Interactive mode (recommended for testing)
```bash
python scripts/natural_language_control.py --model llava --interactive
```

### Single task mode
```bash
python scripts/natural_language_control.py --model llava --task "move forward 2 meters"
```

### With 8-bit quantization (less memory)
```bash
python scripts/natural_language_control.py --model llava --load-8bit --interactive
```

### CPU mode (if no GPU)
```bash
python scripts/natural_language_control.py --model llava --device cpu --interactive
```

## Troubleshooting

### ImportError: cannot import name 'LlavaForConditionalGeneration'
This means transformers version is too old. Fix with:
```bash
conda activate aloha_vlm
pip install --upgrade transformers>=4.40.0
```

### CUDA out of memory
Try 4-bit quantization (enabled by default) or 8-bit:
```bash
python scripts/natural_language_control.py --model llava --load-8bit --interactive
```

### Model download fails
Check internet connection and try manual download:
```bash
python -c "from transformers import AutoProcessor, LlavaForConditionalGeneration; LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf')"
```

## Technical Details

### LLaVA Model
- **Model**: llava-hf/llava-1.5-7b-hf (7 billion parameters)
- **Size**: ~14GB download, ~4GB in memory with 4-bit quantization
- **Inference**: Runs locally, no API key needed
- **Speed**: ~1-2 seconds per inference on GPU

### Memory Requirements
- **4-bit quantization**: ~4-6GB GPU memory (recommended)
- **8-bit quantization**: ~7-8GB GPU memory
- **Full precision**: ~14GB GPU memory
- **CPU mode**: Works but very slow (30+ seconds per inference)

### Alternative Models

If you want to use other models instead of LLaVA:

#### Test mode (no model, rule-based)
```bash
python scripts/natural_language_control.py --model test --interactive
```

#### GPT-4V (requires OpenAI API key)
```bash
export OPENAI_API_KEY="your-key-here"
python scripts/natural_language_control.py --model gpt4v --interactive
```

#### Claude (requires Anthropic API key)
```bash
export ANTHROPIC_API_KEY="your-key-here"
python scripts/natural_language_control.py --model claude --interactive
```

## Files Created

- `environment_vlm.yml` - Conda environment specification
- `requirements_vlm.txt` - Pip requirements for VLM features
- `setup_vlm_env.sh` - Automated setup script (alternative to manual setup)
- `VLM_SETUP_GUIDE.md` - This guide

## Next Steps

1. Wait for conda environment creation to complete
2. Activate the environment: `conda activate aloha_vlm`
3. Run the script: `python scripts/natural_language_control.py --model llava --interactive`
4. Give the robot natural language commands!

## Example Commands to Try

Once in interactive mode, try these commands:
- "move forward"
- "turn left 90 degrees"
- "explore the room"
- "go back to start"
- "stop"

The LLaVA model will analyze the camera feed and decide on actions based on what it sees and your command.
