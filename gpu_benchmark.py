import os
import time
import numpy as np
import torch



def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def test_pytorch_gpu():
    """Test PyTorch GPU capabilities with increasing workloads."""
    print_separator("PyTorch GPU Test")

    # Basic GPU info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

    # Matrix multiplication benchmark
    sizes = [1000, 2000, 4000, 6000, 8000]
    print("\nMatrix multiplication benchmark:")
    for size in sizes:
        # Create random matrices on GPU
        a = torch.rand(size, size, device='cuda')
        b = torch.rand(size, size, device='cuda')

        # Ensure GPU operations are done
        torch.cuda.synchronize()

        # Time matrix multiplication
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        print(f"Size: {size}x{size}, Time: {elapsed:.4f} seconds, "
              f"Memory used: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

        # Clear GPU memory
        del a, b, c
        torch.cuda.empty_cache()

    # Run a small CNN to test compute capability
    print("\nConvolutional network test:")
    batch_size = 64
    input_size = 224

    # Create a simple convolutional network
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(256, 10)
    ).cuda()

    # Create random input
    inputs = torch.rand(batch_size, 3, input_size, input_size, device='cuda')

    # Time forward pass
    torch.cuda.synchronize()
    start_time = time.time()
    outputs = model(inputs)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    print(f"CNN forward pass with batch size {batch_size}: {elapsed:.4f} seconds")
    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB")

    # Clear GPU memory
    del model, inputs, outputs
    torch.cuda.empty_cache()



def main():
    """Run all GPU tests."""
    # Print system info
    print_separator("System Information")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")

    # Run PyTorch tests
    test_pytorch_gpu()

    print_separator("Tests Completed")


if __name__ == "__main__":
    main()