import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend, size=16) -> None:
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


def plot_timing_results(times):
    plt.figure(figsize=(10, 6))
    sizes = list(times.keys())
    
    # Extract timing data for each backend
    fast_times = [times[size]['fast'] for size in sizes]
    gpu_times = [times[size]['gpu'] for size in sizes]
    
    # Create the plot
    plt.plot(sizes, fast_times, 'b-o', label='FastTensor Backend', linewidth=2)
    plt.plot(sizes, gpu_times, 'g-o', label='GPU Backend', linewidth=2)
    
    # Customize the plot
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Use logarithmic scale for better visualization
    plt.yscale('log')
    
    # Add size labels on x-axis
    plt.xticks(sizes, [f'{size}x{size}' for size in sizes])
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add value annotations
    for size, fast_time, gpu_time in zip(sizes, fast_times, gpu_times):
        plt.annotate(f'{fast_time:.3f}s', 
                    (size, fast_time),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
        plt.annotate(f'{gpu_time:.3f}s', 
                    (size, gpu_time),
                    textcoords="offset points",
                    xytext=(0,-15),
                    ha='center')
    
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")
            
    # Create and show the plot
    plt = plot_timing_results(times)
    plt.show()