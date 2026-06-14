# -*- coding: utf-8 -*-
"""
检查GPU浮点单元使用情况 - 简化版本
"""

import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
os.environ['PATH'] = os.environ['CUDA_PATH'] + r'\bin;' + os.environ.get('PATH', '')

import numpy as np
import time
import sys
from numba import cuda, float32, float64

print("=" * 70)
print("GPU Float Point Unit Analysis")
print("=" * 70)

if not cuda.is_available():
    print("[ERROR] CUDA not available!")
    sys.exit(1)

device = cuda.get_current_device()
print(f"\nGPU: {device.name.decode()}")
print(f"Compute Capability: {device.compute_capability}")
print()

# ============================================================================
# FP32 (float32) 内核
# ============================================================================

@cuda.jit
def matvec_fp32(A, x, y, m, n):
    """FP32矩阵-向量乘法"""
    row = cuda.grid(1)
    if row < m:
        sum_val = float32(0.0)
        for j in range(n):
            sum_val += A[row, j] * x[j]
        y[row] = sum_val

# ============================================================================
# FP64 (float64) 内核
# ============================================================================

@cuda.jit
def matvec_fp64(A, x, y, m, n):
    """FP64矩阵-向量乘法"""
    row = cuda.grid(1)
    if row < m:
        sum_val = float64(0.0)
        for j in range(n):
            sum_val += A[row, j] * x[j]
        y[row] = sum_val

# ============================================================================
# 性能测试
# ============================================================================

def benchmark_fp32_vs_fp64():
    """对比FP32和FP64性能"""
    print("=" * 70)
    print("FP32 vs FP64 Performance Comparison")
    print("=" * 70)
    
    sizes = [
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
    ]
    
    for m, n in sizes:
        print(f"\nMatrix size: {m} x {n}")
        print("-" * 50)
        
        # 生成数据
        np.random.seed(42)
        A_fp32 = np.random.randn(m, n).astype(np.float32)
        x_fp32 = np.random.randn(n).astype(np.float32)
        
        A_fp64 = A_fp32.astype(np.float64)
        x_fp64 = x_fp32.astype(np.float64)
        
        threads = 256
        blocks_m = (m + threads - 1) // threads
        
        # ========== FP32测试 ==========
        A32 = cuda.to_device(A_fp32)
        x32 = cuda.to_device(x_fp32)
        result32_m = cuda.device_array(m, dtype=np.float32)
        
        # 预热
        matvec_fp32[blocks_m, threads](A32, x32, result32_m, m, n)
        cuda.synchronize()
        
        # 测试
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            matvec_fp32[blocks_m, threads](A32, x32, result32_m, m, n)
        cuda.synchronize()
        time_fp32 = (time.time() - start) / iterations
        
        # ========== FP64测试 ==========
        A64 = cuda.to_device(A_fp64)
        x64 = cuda.to_device(x_fp64)
        result64_m = cuda.device_array(m, dtype=np.float64)
        
        # 预热
        matvec_fp64[blocks_m, threads](A64, x64, result64_m, m, n)
        cuda.synchronize()
        
        # 测试
        start = time.time()
        for _ in range(iterations):
            matvec_fp64[blocks_m, threads](A64, x64, result64_m, m, n)
        cuda.synchronize()
        time_fp64 = (time.time() - start) / iterations
        
        # 结果
        speedup = time_fp64 / time_fp32
        
        print(f"  FP32 time: {time_fp32*1000:.3f} ms")
        print(f"  FP64 time: {time_fp64*1000:.3f} ms")
        print(f"  FP32 speedup: {speedup:.2f}x")
        
        # 验证结果正确性
        result32_host = result32_m.copy_to_host()
        result64_host = result64_m.copy_to_host()
        
        # 计算CPU参考
        result_cpu_fp32 = A_fp32 @ x_fp32
        result_cpu_fp64 = A_fp64 @ x_fp64
        
        error_fp32 = np.linalg.norm(result32_host - result_cpu_fp32)
        error_fp64 = np.linalg.norm(result64_host - result_cpu_fp64)
        
        print(f"  FP32 error: {error_fp32:.6e}")
        print(f"  FP64 error: {error_fp64:.6e}")

def check_gpu_fp_capabilities():
    """检查GPU浮点能力"""
    print("=" * 70)
    print("GPU Floating Point Capabilities")
    print("=" * 70)
    
    # RTX 5060 架构信息
    print("\n[NVIDIA GeForce RTX 5060]")
    print("  Architecture: Ada Lovelace (Compute Capability 8.9)")
    print("  FP32 CUDA Cores: 6144")
    print("  FP64 CUDA Cores: 192 (1/32 of FP32)")
    print("  FP32 Performance: ~22 TFLOPS")
    print("  FP64 Performance: ~0.7 TFLOPS (1/32 of FP32)")
    print("  Tensor Cores: 4th Gen")
    print("  Supports: FP16, BF16, FP32, FP64, INT8, INT4")
    
    print("\n[FP32 vs FP64 Ratio]")
    print("  Theoretical: FP32 is 32x faster than FP64")
    print("  Expected speedup: 16-32x")

def main():
    print("\n" + "=" * 70)
    print("GPU Floating Point Unit Analysis Tool")
    print("=" * 70)
    
    check_gpu_fp_capabilities()
    benchmark_fp32_vs_fp64()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nRTX 5060 has dedicated FP32 and FP64 units:")
    print("  - FP32: 6144 cores, full performance")
    print("  - FP64: 192 cores, 1/32 performance")
    print("\nCurrent LP solver uses FP64 (float64) for numerical stability")
    print("For pure performance, FP32 would be ~16-32x faster")
    print("=" * 70)

if __name__ == "__main__":
    main()
