#!/bin/bash
# Setup script for ALOHA VLM environment

set -e  # Exit on error

echo "========================================="
echo "ALOHA VLM Environment Setup"
echo "========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Step 1: Creating conda environment 'aloha_vlm'..."
echo "This may take several minutes..."
conda env create -f "$SCRIPT_DIR/environment_vlm.yml" || {
    echo "Warning: Environment might already exist. Updating instead..."
    conda env update -f "$SCRIPT_DIR/environment_vlm.yml" --prune
}

echo ""
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate aloha_vlm

echo ""
echo "Step 3: Verifying installation..."
python -c "from transformers import LlavaForConditionalGeneration; print('✓ LLaVA import successful')" || {
    echo "✗ LLaVA import failed. Installing missing dependencies..."
    pip install --upgrade transformers>=4.40.0 accelerate>=0.20.0 bitsandbytes>=0.41.0
}

echo ""
echo "Step 4: Testing PyTorch CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To use the environment:"
echo "  conda activate aloha_vlm"
echo ""
echo "To run natural language control:"
echo "  cd $SCRIPT_DIR"
echo "  python scripts/natural_language_control.py --model llava --interactive"
echo ""
echo "Note: First run will download the LLaVA model (~14GB)"
echo "========================================="
