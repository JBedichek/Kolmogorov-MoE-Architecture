#!/bin/bash
# Build custom CUDA kernels for MoE acceleration
#
# Usage: ./build_kernels.sh
#
# Requirements:
# - CUDA toolkit installed
# - PyTorch with CUDA support
# - ninja (for faster builds): pip install ninja

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/moe_arch/kernels"

echo "=========================================="
echo "Building MoE CUDA Kernels"
echo "=========================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Build kernels
cd "$KERNEL_DIR"

echo "Cleaning previous builds..."
rm -rf build/ *.so *.egg-info/ __pycache__/

echo "Building CUDA extension..."
python setup.py build_ext --inplace

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="

# Test import
echo ""
echo "Testing kernel import..."
python -c "
import torch
import moe_kernels
print('CUDA kernel imported successfully!')
print('Available functions:', dir(moe_kernels))
"

echo ""
echo "Running benchmark..."
python benchmark.py
