"""
Setup script for compiling custom CUDA kernels.

Usage:
    cd moe_arch/kernels
    python setup.py install

Or for development:
    python setup.py build_ext --inplace
"""

import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_version():
    """Get CUDA version from nvcc."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        output = result.stdout
        # Parse "release 12.0" or similar
        for line in output.split('\n'):
            if 'release' in line.lower():
                parts = line.split('release')[-1].strip()
                version = parts.split(',')[0].strip()
                major, minor = version.split('.')[:2]
                return int(major), int(minor)
    except:
        pass
    return 12, 0  # Default


def get_gpu_arch():
    """Detect the current GPU's compute capability."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.major, props.minor
    except:
        pass
    return None, None


def get_cuda_arch_flags():
    """Get CUDA architecture flags based on system nvcc capabilities."""
    cuda_major, cuda_minor = get_cuda_version()
    cuda_version = cuda_major * 10 + cuda_minor

    # Detect current GPU
    gpu_major, gpu_minor = get_gpu_arch()
    if gpu_major:
        print(f"Detected GPU: sm_{gpu_major}{gpu_minor}")

    flags = []

    # Always use system nvcc's capabilities (not PyTorch's version)
    # because nvcc is what actually compiles the code

    # For forward compatibility with newer GPUs, use PTX
    # PTX code can be JIT-compiled at runtime for newer architectures

    if gpu_major and gpu_major >= 12:
        # Blackwell GPU but system nvcc is older
        # Build PTX for the highest architecture nvcc supports
        if cuda_version >= 120:
            # CUDA 12.0+ supports sm_90
            print(f"Building PTX for compute_90 (forward compatible with Blackwell)")
            flags.extend(['-gencode', 'arch=compute_90,code=compute_90'])
        elif cuda_version >= 118:
            flags.extend(['-gencode', 'arch=compute_89,code=compute_89'])
        else:
            flags.extend(['-gencode', 'arch=compute_80,code=compute_80'])
    elif gpu_major == 10:
        # sm_100 (Blackwell variant)
        if cuda_version >= 124:
            flags.extend(['-gencode', 'arch=compute_100,code=sm_100'])
        else:
            flags.extend(['-gencode', 'arch=compute_90,code=compute_90'])
    elif gpu_major == 9:
        # Hopper
        if cuda_version >= 120:
            flags.extend(['-gencode', f'arch=compute_9{gpu_minor},code=sm_9{gpu_minor}'])
        else:
            flags.extend(['-gencode', 'arch=compute_89,code=compute_89'])
    elif gpu_major == 8:
        # Ampere (80) or Ada (89)
        if gpu_minor >= 9 and cuda_version >= 118:
            flags.extend(['-gencode', 'arch=compute_89,code=sm_89'])
        else:
            flags.extend(['-gencode', 'arch=compute_80,code=sm_80'])
    else:
        # Unknown or older GPU - build for common architectures
        if cuda_version >= 120:
            flags.extend(['-gencode', 'arch=compute_90,code=sm_90'])
        if cuda_version >= 118:
            flags.extend(['-gencode', 'arch=compute_89,code=sm_89'])
        if cuda_version >= 110:
            flags.extend(['-gencode', 'arch=compute_80,code=sm_80'])

    if not flags:
        # Last resort
        flags = ['-gencode', 'arch=compute_80,code=compute_80']

    print(f"System nvcc CUDA: {cuda_major}.{cuda_minor}")
    print(f"Architecture flags: {flags}")

    return flags


setup(
    name='moe_kernels',
    ext_modules=[
        CUDAExtension(
            name='moe_kernels',
            sources=['grouped_gemm.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',  # For profiling
                    '--ptxas-options=-v',  # Show register usage
                ] + get_cuda_arch_flags()
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
