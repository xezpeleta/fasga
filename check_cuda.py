#!/usr/bin/env python3
"""
Quick CUDA/cuDNN diagnostic script for FASGA.

Run this to check your GPU setup:
    uv run python check_cuda.py
"""

import sys

def main():
    print("="*60)
    print("FASGA CUDA/GPU Diagnostic")
    print("="*60)
    print()
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed!")
        sys.exit(1)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")
    
    if cuda_available:
        # CUDA details
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        
        # GPU details
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
        
        # cuDNN
        try:
            cudnn_available = torch.backends.cudnn.is_available()
            print(f"{'✓' if cudnn_available else '✗'} cuDNN available: {cudnn_available}")
            if cudnn_available:
                print(f"  - cuDNN version: {torch.backends.cudnn.version()}")
        except Exception as e:
            print(f"✗ cuDNN check failed: {e}")
        
        print()
        print("="*60)
        print("Status: ✓ GPU acceleration ready!")
        print("="*60)
    else:
        print()
        print("="*60)
        print("Status: ✗ GPU acceleration NOT available")
        print("="*60)
        print()
        print("Possible causes:")
        print("  1. Missing cuDNN library")
        print("     → Install cuDNN for CUDA 12.x")
        print("     → See: CUDA_FIX.md for instructions")
        print()
        print("  2. CUDA toolkit not properly installed")
        print("     → Install NVIDIA CUDA Toolkit")
        print()
        print("  3. No compatible GPU detected")
        print("     → Check: nvidia-smi")
        print()
        print("System commands to try:")
        print("  $ nvidia-smi")
        print("  $ ldconfig -p | grep libcudnn")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()

