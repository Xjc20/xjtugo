"""检查CUDA是否可用"""
import sys

try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n可能的原因:")
        print("1. 没有安装NVIDIA显卡驱动")
        print("2. 安装的PyTorch是CPU版本")
        print("3. CUDA toolkit没有正确安装")
        print("\n解决方案:")
        print("1. 安装GPU版PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("2. 确保已安装NVIDIA驱动和CUDA toolkit")
except ImportError as e:
    print(f"导入错误: {e}")
    print("PyTorch可能没有安装")
    sys.exit(1)
