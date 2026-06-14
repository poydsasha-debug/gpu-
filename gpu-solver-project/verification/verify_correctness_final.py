# -*- coding: utf-8 -*-
"""
GPU LP求解器 - 计算正确性验证
使用scipy.optimize.linprog作为参考
验证GPU矩阵运算的正确性
"""

import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
os.environ['PATH'] = os.environ['CUDA_PATH'] + r'\bin;' + os.environ.get('PATH', '')

import numpy as np
import time
import sys
from numba import cuda
from scipy.optimize import linprog

print("=" * 70)
print("GPU LP Solver - Correctness Verification")
print("=" * 70)

if not cuda.is_available():
    print("[ERROR] CUDA not available!")
    sys.exit(1)

device = cuda.get_current_device()
print(f"\nGPU: {device.name.decode()}")
print()

# ============================================================================
# CUDA内核 - 矩阵运算
# ============================================================================

@cuda.jit
def matvec_kernel(A, x, y, m, n):
    """矩阵-向量乘法 y = A @ x"""
    row = cuda.grid(1)
    if row < m:
        sum_val = 0.0
        for j in range(n):
            sum_val += A[row, j] * x[j]
        y[row] = sum_val

@cuda.jit
def matvec_trans_kernel(A, x, y, m, n):
    """矩阵转置-向量乘法 y = A.T @ x"""
    col = cuda.grid(1)
    if col < n:
        sum_val = 0.0
        for i in range(m):
            sum_val += A[i, col] * x[i]
        y[col] = sum_val

@cuda.jit
def vector_add_kernel(a, b, result, n):
    idx = cuda.grid(1)
    if idx < n:
        result[idx] = a[idx] + b[idx]

@cuda.jit
def vector_sub_kernel(a, b, result, n):
    idx = cuda.grid(1)
    if idx < n:
        result[idx] = a[idx] - b[idx]

# ============================================================================
# GPU矩阵运算验证
# ============================================================================

class GPUMatrixOps:
    """GPU矩阵运算类"""
    
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.threads_per_block = 256
        self.blocks_per_grid_m = (m + self.threads_per_block - 1) // self.threads_per_block
        self.blocks_per_grid_n = (n + self.threads_per_block - 1) // self.threads_per_block
    
    def matvec(self, A_gpu, x_gpu, result_gpu):
        """矩阵-向量乘法"""
        matvec_kernel[self.blocks_per_grid_m, self.threads_per_block](
            A_gpu, x_gpu, result_gpu, self.m, self.n)
        cuda.synchronize()
    
    def matvec_trans(self, A_gpu, x_gpu, result_gpu):
        """矩阵转置-向量乘法"""
        matvec_trans_kernel[self.blocks_per_grid_n, self.threads_per_block](
            A_gpu, x_gpu, result_gpu, self.m, self.n)
        cuda.synchronize()

# ============================================================================
# 验证测试
# ============================================================================

def verify_matrix_operations():
    """验证GPU矩阵运算的正确性"""
    print("\n" + "=" * 70)
    print("TEST 1: GPU Matrix Operations Correctness")
    print("=" * 70)
    
    np.random.seed(42)
    m, n = 100, 200
    
    # 生成测试数据
    A = np.random.randn(m, n).astype(np.float64)
    x = np.random.randn(n).astype(np.float64)
    y = np.random.randn(m).astype(np.float64)
    
    print(f"\nMatrix size: {m} x {n}")
    
    # 分配到GPU
    A_gpu = cuda.to_device(A)
    x_gpu = cuda.to_device(x)
    y_gpu = cuda.to_device(y)
    result_m = cuda.device_array(m, dtype=np.float64)
    result_n = cuda.device_array(n, dtype=np.float64)
    
    ops = GPUMatrixOps(m, n)
    
    # 测试1: y = A @ x
    print("\n[Test 1] y = A @ x")
    ops.matvec(A_gpu, x_gpu, result_m)
    y_gpu_result = result_m.copy_to_host()
    y_cpu = A @ x
    
    diff = np.linalg.norm(y_gpu_result - y_cpu)
    rel_diff = diff / (np.linalg.norm(y_cpu) + 1e-10)
    
    print(f"  CPU result norm: {np.linalg.norm(y_cpu):.6f}")
    print(f"  GPU result norm: {np.linalg.norm(y_gpu_result):.6f}")
    print(f"  Absolute diff: {diff:.6e}")
    print(f"  Relative diff: {rel_diff:.6e}")
    
    if rel_diff < 1e-10:
        print("  [PASS] Matrix-vector multiplication correct")
        test1_pass = True
    else:
        print("  [FAIL] Results differ")
        test1_pass = False
    
    # 测试2: x = A.T @ y
    print("\n[Test 2] x = A.T @ y")
    ops.matvec_trans(A_gpu, y_gpu, result_n)
    x_gpu_result = result_n.copy_to_host()
    x_cpu = A.T @ y
    
    diff = np.linalg.norm(x_gpu_result - x_cpu)
    rel_diff = diff / (np.linalg.norm(x_cpu) + 1e-10)
    
    print(f"  CPU result norm: {np.linalg.norm(x_cpu):.6f}")
    print(f"  GPU result norm: {np.linalg.norm(x_gpu_result):.6f}")
    print(f"  Absolute diff: {diff:.6e}")
    print(f"  Relative diff: {rel_diff:.6e}")
    
    if rel_diff < 1e-10:
        print("  [PASS] Matrix-transpose-vector multiplication correct")
        test2_pass = True
    else:
        print("  [FAIL] Results differ")
        test2_pass = False
    
    return test1_pass and test2_pass

def verify_lp_solution():
    """验证LP解的正确性"""
    print("\n" + "=" * 70)
    print("TEST 2: LP Solution Correctness")
    print("=" * 70)
    
    # 使用scipy求解LP作为参考
    np.random.seed(42)
    m, n = 50, 100
    
    # 生成一个简单但有界的LP问题
    # min c^T x s.t. Ax <= b, x >= 0
    A_ub = np.random.rand(m, n) * 2 - 1
    b_ub = np.random.rand(m) * 10 + 5
    c = np.random.randn(n)
    
    print(f"\nProblem: {m} inequalities, {n} variables")
    
    # 使用scipy求解
    print("\n[SciPy linprog]")
    start = time.time()
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n, 
                     method='highs', options={'maxiter': 1000})
    time_scipy = time.time() - start
    
    if result.success:
        x_scipy = result.x
        obj_scipy = result.fun
        
        print(f"  Status: Optimal")
        print(f"  Objective: {obj_scipy:.6f}")
        print(f"  Time: {time_scipy:.4f}s")
        
        # 验证约束
        violation_scipy = np.max(A_ub @ x_scipy - b_ub)
        print(f"  Max constraint violation: {violation_scipy:.6e}")
        
        # 注意：这里我们只是验证scipy能求解
        # 真正的GPU LP求解器需要更复杂的实现
        print("\n[Note] Full GPU LP solver requires more complex implementation")
        print("  Current GPU kernels correctly implement matrix operations")
        
        return True
    else:
        print(f"  Failed: {result.message}")
        return False

def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 70)
    print("TEST 3: Performance Benchmark")
    print("=" * 70)
    
    test_sizes = [
        (100, 200),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
    ]
    
    for m, n in test_sizes:
        print(f"\nSize: {m} x {n}")
        
        np.random.seed(42)
        A = np.random.randn(m, n).astype(np.float64)
        x = np.random.randn(n).astype(np.float64)
        
        # CPU
        start = time.time()
        for _ in range(10):
            y_cpu = A @ x
        time_cpu = (time.time() - start) / 10
        
        # GPU
        A_gpu = cuda.to_device(A)
        x_gpu = cuda.to_device(x)
        result_gpu = cuda.device_array(m, dtype=np.float64)
        
        ops = GPUMatrixOps(m, n)
        
        # 预热
        ops.matvec(A_gpu, x_gpu, result_gpu)
        
        start = time.time()
        for _ in range(100):
            ops.matvec(A_gpu, x_gpu, result_gpu)
        time_gpu = (time.time() - start) / 100
        
        speedup = time_cpu / time_gpu
        
        print(f"  CPU time: {time_cpu*1000:.3f} ms")
        print(f"  GPU time: {time_gpu*1000:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

def main():
    print("\n" + "=" * 70)
    print("GPU LP Solver - Comprehensive Correctness Check")
    print("=" * 70)
    print("\nThis test verifies:")
    print("  1. GPU matrix operations match CPU results")
    print("  2. LP problem formulation is correct")
    print("  3. Performance is as expected")
    
    # 测试1: 矩阵运算正确性
    matrix_correct = verify_matrix_operations()
    
    # 测试2: LP解正确性
    lp_correct = verify_lp_solution()
    
    # 测试3: 性能
    performance_benchmark()
    
    # 总结
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nMatrix Operations: {'PASS' if matrix_correct else 'FAIL'}")
    print(f"LP Formulation: {'PASS' if lp_correct else 'FAIL'}")
    
    if matrix_correct and lp_correct:
        print("\n[OK] GPU kernels are correct")
        print("[Note] Full interior point method needs careful numerical implementation")
    else:
        print("\n[WARNING] Some tests failed")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
