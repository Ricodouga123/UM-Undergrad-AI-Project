import torch

print(f"PyTorch version:   {torch.__version__}")
print(f"CUDA available:    {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version:      {torch.version.cuda}")
    print(f"Device name:       {torch.cuda.get_device_name(0)}")
    print(f"Device memory:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Simple tensor operation on GPU
    print("\nRunning a quick tensor test on GPU...")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"Matrix multiply result shape: {z.shape}")
    print("GPU tensor test passed!")
else:
    print("\nWARNING: CUDA is not available.")
    print("If you just installed PyTorch from PyPI, that is likely the issue.")
    print("On Jetson, install PyTorch from NVIDIA's wheel instead:")
    print("  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048")
