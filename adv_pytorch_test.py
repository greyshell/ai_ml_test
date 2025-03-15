import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    print_separator("Basic Tensor Operations Test")

    # Create tensors on GPU
    a = torch.randn(10000, 10000, device='cuda')
    b = torch.randn(10000, 10000, device='cuda')

    # Element-wise operations
    torch.cuda.synchronize()
    start_time = time.time()
    c = a + b
    torch.cuda.synchronize()
    print(f"Element-wise addition (10k x 10k): {time.time() - start_time:.6f} seconds")

    # Matrix multiplication
    torch.cuda.synchronize()
    start_time = time.time()
    d = torch.mm(a, b)
    torch.cuda.synchronize()
    print(f"Matrix multiplication (10k x 10k): {time.time() - start_time:.6f} seconds")

    # Reduction operations
    torch.cuda.synchronize()
    start_time = time.time()
    e = torch.sum(a, dim=1)
    torch.cuda.synchronize()
    print(f"Sum reduction (10k x 10k): {time.time() - start_time:.6f} seconds")

    # Clear GPU memory
    del a, b, c, d, e
    torch.cuda.empty_cache()


def test_training_loop():
    """Test a training loop with backpropagation."""
    print_separator("Training Loop Test")

    # Generate synthetic data
    n_samples = 10000
    n_features = 1000

    # Create random data on CPU (transfer to GPU inside loop is realistic)
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, 1)

    # Create dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(n_features, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).cuda()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 5
    times = []

    print(f"Training for {n_epochs} epochs with {n_samples} samples, {n_features} features")
    for epoch in range(n_epochs):
        torch.cuda.synchronize()
        epoch_start = time.time()

        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data to GPU
            data, target = data.cuda(), target.cuda()

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print(f"Epoch {epoch + 1}/{n_epochs}: {epoch_time:.4f} seconds")

    print(f"Average epoch time: {np.mean(times):.4f} seconds")

    # Clear GPU memory
    del model, X, y
    torch.cuda.empty_cache()


def test_memory_bandwidth():
    """Test GPU memory bandwidth."""
    print_separator("Memory Bandwidth Test")

    # Test different tensor sizes
    sizes = [1000, 2000, 4000, 8000, 16000]

    for size in sizes:
        # Create a large tensor
        tensor = torch.randn(size, size, device='cuda')

        # Measure read bandwidth
        torch.cuda.synchronize()
        start_time = time.time()
        result = tensor * 2  # Simple operation to force memory read/write
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Calculate bandwidth
        bytes_processed = tensor.nelement() * tensor.element_size() * 2  # Read and write
        bandwidth = bytes_processed / elapsed / (1024 ** 3)  # GB/s

        print(f"Size: {size}x{size}, Bandwidth: {bandwidth:.2f} GB/s")

        # Clear memory
        del tensor, result
        torch.cuda.empty_cache()


def test_multi_gpu_transfer():
    """Test data transfer between CPU and GPU."""
    print_separator("CPU-GPU Transfer Test")

    # Create tensors of different sizes
    sizes = [1000, 2000, 4000, 8000, 16000]

    for size in sizes:
        # Create CPU tensor
        cpu_tensor = torch.randn(size, size)

        # Measure CPU to GPU transfer time
        start_time = time.time()
        gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()
        cpu_to_gpu = time.time() - start_time

        # Measure GPU to CPU transfer time
        start_time = time.time()
        new_cpu_tensor = gpu_tensor.cpu()
        torch.cuda.synchronize()
        gpu_to_cpu = time.time() - start_time

        # Calculate transfer rates
        bytes_transferred = cpu_tensor.nelement() * cpu_tensor.element_size()
        cpu_to_gpu_rate = bytes_transferred / cpu_to_gpu / (1024 ** 3)  # GB/s
        gpu_to_cpu_rate = bytes_transferred / gpu_to_cpu / (1024 ** 3)  # GB/s

        print(f"Size: {size}x{size}")
        print(f"  CPU to GPU: {cpu_to_gpu:.6f} s ({cpu_to_gpu_rate:.2f} GB/s)")
        print(f"  GPU to CPU: {gpu_to_cpu:.6f} s ({gpu_to_cpu_rate:.2f} GB/s)")

        # Clear memory
        del cpu_tensor, gpu_tensor, new_cpu_tensor
        torch.cuda.empty_cache()


def test_stream_execution():
    """Test concurrent execution with CUDA streams."""
    print_separator("CUDA Streams Test")

    # Create two streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    size = 8000

    # Create tensors
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    c = torch.randn(size, size, device='cuda')
    d = torch.randn(size, size, device='cuda')

    # Synchronize before starting timing
    torch.cuda.synchronize()

    # Sequential execution
    start_time = time.time()
    e = torch.mm(a, b)
    f = torch.mm(c, d)
    torch.cuda.synchronize()
    sequential_time = time.time() - start_time

    # Concurrent execution with streams
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.cuda.stream(stream1):
        g = torch.mm(a, b)

    with torch.cuda.stream(stream2):
        h = torch.mm(c, d)

    torch.cuda.synchronize()
    concurrent_time = time.time() - start_time

    print(f"Sequential execution time: {sequential_time:.6f} seconds")
    print(f"Concurrent execution time: {concurrent_time:.6f} seconds")
    print(f"Speedup: {sequential_time / concurrent_time:.2f}x")

    # Clear memory
    del a, b, c, d, e, f, g, h
    torch.cuda.empty_cache()


def test_mixed_precision():
    """Test FP16 (mixed precision) vs FP32 performance."""
    print_separator("Mixed Precision Test")

    size = 8000

    # FP32 test
    a_fp32 = torch.randn(size, size, device='cuda', dtype=torch.float32)
    b_fp32 = torch.randn(size, size, device='cuda', dtype=torch.float32)

    torch.cuda.synchronize()
    start_time = time.time()
    c_fp32 = torch.mm(a_fp32, b_fp32)
    torch.cuda.synchronize()
    fp32_time = time.time() - start_time

    # FP16 test
    a_fp16 = torch.randn(size, size, device='cuda', dtype=torch.float16)
    b_fp16 = torch.randn(size, size, device='cuda', dtype=torch.float16)

    torch.cuda.synchronize()
    start_time = time.time()
    c_fp16 = torch.mm(a_fp16, b_fp16)
    torch.cuda.synchronize()
    fp16_time = time.time() - start_time

    print(f"FP32 matrix multiplication time: {fp32_time:.6f} seconds")
    print(f"FP16 matrix multiplication time: {fp16_time:.6f} seconds")
    print(f"Speedup from mixed precision: {fp32_time / fp16_time:.2f}x")

    # Clear memory
    del a_fp32, b_fp32, c_fp32, a_fp16, b_fp16, c_fp16
    torch.cuda.empty_cache()


def main():
    """Run all tests."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation.")
        return

    # Print GPU information
    print_separator("GPU Information")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Run tests
    test_tensor_operations()
    test_memory_bandwidth()
    test_multi_gpu_transfer()
    test_stream_execution()
    test_mixed_precision()
    test_training_loop()

    print_separator("All Tests Completed")
    print("Your GPU is functioning properly with PyTorch.")


if __name__ == "__main__":
    main()