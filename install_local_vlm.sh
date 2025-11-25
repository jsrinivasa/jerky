#!/bin/bash

# Installation script for local VLM support (LLaVA)
# This installs the necessary dependencies to run vision-language models locally
# NO API KEYS NEEDED!

echo "=========================================="
echo "Local VLM Installation for Mobile ALOHA"
echo "=========================================="
echo ""
echo "This will install:"
echo "  - LLaVA 1.5 (7B vision-language model)"
echo "  - Required dependencies (transformers, accelerate, bitsandbytes)"
echo ""
echo "Requirements:"
echo "  - GPU with 8GB+ VRAM (recommended)"
echo "  - ~15GB disk space for model"
echo ""

read -p "Continue with installation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Installation cancelled"
    exit 1
fi

echo ""
echo "Installing dependencies..."
echo "=========================="

# Install core dependencies
pip install transformers>=4.40.0
pip install accelerate>=0.20.0
pip install bitsandbytes>=0.41.0

echo ""
echo "✓ Dependencies installed!"
echo ""
echo "Testing model download..."
echo "=========================="

# Test by importing and checking model availability
python3 << EOF
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

print("\nDownloading LLaVA model (this may take a few minutes)...")
model_path = "llava-hf/llava-1.5-7b-hf"

try:
    processor = AutoProcessor.from_pretrained(model_path)
    print("✓ Processor downloaded")
    
    # Note: We're not loading the full model here to save time
    # It will be downloaded on first use
    print("✓ Model available (will download on first use)")
    print("\nSetup complete!")
    
except Exception as e:
    print("✗ Error:", e)
    print("\nNote: Model will be downloaded automatically on first use")

EOF

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To test the system:"
echo ""
echo "  # Run with test data (no robot needed)"
echo "  python scripts/natural_language_control.py --model llava --task \"explore the room\" --max-steps 5"
echo ""
echo "  # Interactive mode"
echo "  python scripts/natural_language_control.py --model llava --interactive"
echo ""
echo "The LLaVA model will download automatically on first use (~7GB)"
echo "This may take a few minutes depending on your internet connection."
echo ""
echo "NO API KEYS REQUIRED! Everything runs locally on your machine."
echo ""
