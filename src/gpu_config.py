"""
GPU Configuration for Face Recognition
Forces use of GPU acceleration with PyTorch backend

Usage:
    Import this at the top of train_model.py and test_model.py:
    
    import gpu_config
    
This will:
1. Force DeepFace to use PyTorch backend (which has GPU support)
2. Verify GPU is available
3. Display GPU information
"""
import os
import sys

# Force PyTorch backend for DeepFace
os.environ['DEEPFACE_BACKEND'] = 'pytorch'

# Verify GPU availability
try:
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("\n" + "="*60)
        print("GPU ACCELERATION ENABLED")
        print("="*60)
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ VRAM: {gpu_memory:.1f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ Backend: PyTorch (GPU-accelerated)")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("WARNING: GPU NOT AVAILABLE")
        print("="*60)
        print("⚠ Running on CPU - will be slower")
        print("⚠ Check GPU drivers and CUDA installation")
        print("="*60 + "\n")
        
except ImportError:
    print("\n" + "="*60)
    print("ERROR: PyTorch not installed")
    print("="*60)
    print("Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("="*60 + "\n")
    sys.exit(1)


def get_gpu_info():
    """Get detailed GPU information."""
    import torch
    
    if not torch.cuda.is_available():
        return None
    
    info = {
        'available': True,
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(0),
        'total_memory': torch.cuda.get_device_properties(0).total_memory,
        'cuda_version': torch.version.cuda
    }
    
    return info


def optimize_gpu_memory():
    """Optimize GPU memory usage."""
    import torch
    
    if torch.cuda.is_available():
        # Enable TF32 for faster training on Ampere GPUs (RTX 4090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        
        print("✓ GPU optimizations enabled (TF32, cuDNN benchmark)")


# Apply optimizations
optimize_gpu_memory()