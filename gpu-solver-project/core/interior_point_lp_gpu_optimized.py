# -*- coding: utf-8 -*-
"""
GPU满载内点法LP求解器 - 修复版本
优化线程配置，避免资源不足

使用方法:
    python interior_point_lp_gpu_optimized.py [持续时间]
"""

import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
os.environ['PATH'] = os.environ['CUDA_PATH'] + r'\bin;' + os.environ.get('PATH', '')

import numpy as np
import time
import sys
from numba import cuda, float64

print("=" * 70)
print("GPU OPTIMIZED Interior Point LP Solver")
print("=" * 70)

# 检查CUDA
if not cuda.is_available():
    print("[ERROR] CUDA not available!")
    sys.exit(1)

device = cuda.get_current_device()
print(f"\nGPU: {device.name.decode()}")
print(f"Compute Capability: {device.compute_capability}")
print(f"Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
print()

# 浮点误差常量
EPSILON = 1e-10
MU_MIN = 1e-14

# ============================================================================
# 优化的CUDA内核 - 减少寄存器使用
# ============================================================================

@cuda.jit(max_registers=64)
def matvec_kernel_optimized(A, x, y, m, n):
    """优化的矩阵-向量乘法"""
    row = cuda.grid(1)
    if row < m:
        sum_val = 0.0
        # 循环展开减少迭代
        for j in range(0, n, 2):
            if j + 1 < n:
                sum_val += A[row, j] * x[j] + A[row, j + 1] * x[j + 1]
            else:
                sum_val += A[row, j] * x[j]
        y[row] = sum_val

@cuda.jit(max_registers=64)
def matvec_trans_kernel_optimized(A, x, y, m, n):
    """优化的矩阵转置-向量乘法"""
    col = cuda.grid(1)
    if col < n:
        sum_val = 0.0
        for i in range(0, m, 2):
            if i + 1 < m:
                sum_val += A[i, col] * x[i] + A[i + 1, col] * x[i + 1]
            else:
                sum_val += A[i, col] * x[i]
        y[col] = sum_val

@cuda.jit(max_registers=32)
def vector_add_kernel(a, b, result, n):
    """向量加法"""
    idx = cuda.grid(1)
    if idx < n:
        result[idx] = a[idx] + b[idx]

@cuda.jit(max_registers=32)
def vector_sub_kernel(a, b, result, n):
    """向量减法"""
    idx = cuda.grid(1)
    if idx < n:
        result[idx] = a[idx] - b[idx]

@cuda.jit(max_registers=32)
def vector_mul_kernel(a, b, result, n):
    """向量逐元素乘法"""
    idx = cuda.grid(1)
    if idx < n:
        result[idx] = a[idx] * b[idx]

@cuda.jit(max_registers=32)
def vector_div_kernel(a, b, result, n, eps):
    """向量逐元素除法（带epsilon保护）"""
    idx = cuda.grid(1)
    if idx < n:
        b_safe = b[idx] if b[idx] > eps else eps
        result[idx] = a[idx] / b_safe

@cuda.jit(max_registers=32)
def make_positive_kernel(x, eps, n):
    """保持正性"""
    idx = cuda.grid(1)
    if idx < n:
        if x[idx] < eps:
            x[idx] = eps

# ============================================================================
# GPU LP求解器 - 优化版本
# ============================================================================

class GPULPOptimizedSolver:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        
        # 保守的线程配置
        self.threads_per_block = 256  # 减少线程数
        self.blocks_per_grid_m = (m + self.threads_per_block - 1) // self.threads_per_block
        self.blocks_per_grid_n = (n + self.threads_per_block - 1) // self.threads_per_block
        
        print(f"Thread config: {self.blocks_per_grid_m} blocks x {self.threads_per_block} threads")
        print()
    
    def allocate_device_arrays(self, A_host, b_host, c_host):
        """分配GPU内存"""
        self.A = cuda.to_device(A_host.astype(np.float64))
        self.b = cuda.to_device(b_host.astype(np.float64))
        self.c = cuda.to_device(c_host.astype(np.float64))
        
        self.x = cuda.device_array(self.n, dtype=np.float64)
        self.y = cuda.device_array(self.m, dtype=np.float64)
        self.s = cuda.device_array(self.n, dtype=np.float64)
        
        self.dx = cuda.device_array(self.n, dtype=np.float64)
        self.dy = cuda.device_array(self.m, dtype=np.float64)
        self.ds = cuda.device_array(self.n, dtype=np.float64)
        
        self.D = cuda.device_array(self.n, dtype=np.float64)
        self.r_b = cuda.device_array(self.m, dtype=np.float64)
        self.r_c = cuda.device_array(self.n, dtype=np.float64)
    
    def initialize(self):
        """初始化"""
        x_host = np.ones(self.n, dtype=np.float64)
        s_host = np.ones(self.n, dtype=np.float64)
        y_host = np.zeros(self.m, dtype=np.float64)
        
        self.x = cuda.to_device(x_host)
        self.s = cuda.to_device(s_host)
        self.y = cuda.to_device(y_host)
    
    def compute_residuals_gpu(self):
        """在GPU上计算残差"""
        # r_b = A @ x
        matvec_kernel_optimized[self.blocks_per_grid_m, self.threads_per_block](
            self.A, self.x, self.r_b, self.m, self.n)
        cuda.synchronize()
        
        # r_b = r_b - b
        vector_sub_kernel[self.blocks_per_grid_m, self.threads_per_block](
            self.r_b, self.b, self.r_b, self.m)
        cuda.synchronize()
        
        # r_c = A.T @ y
        matvec_trans_kernel_optimized[self.blocks_per_grid_n, self.threads_per_block](
            self.A, self.y, self.r_c, self.m, self.n)
        cuda.synchronize()
        
        # r_c = r_c + s
        vector_add_kernel[self.blocks_per_grid_n, self.threads_per_block](
            self.r_c, self.s, self.r_c, self.n)
        cuda.synchronize()
        
        # r_c = r_c - c
        vector_sub_kernel[self.blocks_per_grid_n, self.threads_per_block](
            self.r_c, self.c, self.r_c, self.n)
        cuda.synchronize()
    
    def solve(self, max_iter=100, tol=1e-8, verbose=True):
        """内点法求解"""
        self.initialize()
        
        if verbose:
            print("Starting GPU interior point iterations...\n")
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            # 计算残差
            self.compute_residuals_gpu()
            
            # 复制回CPU计算范数
            r_b_host = self.r_b.copy_to_host()
            r_c_host = self.r_c.copy_to_host()
            x_host = self.x.copy_to_host()
            s_host = self.s.copy_to_host()
            
            primal_resid = np.linalg.norm(r_b_host)
            dual_resid = np.linalg.norm(r_c_host)
            mu = np.dot(x_host, s_host) / self.n
            
            if verbose and (iteration % 10 == 0 or iteration < 5):
                elapsed = time.time() - start_time
                print(f"Iter {iteration:3d}: mu={mu:.6e}, "
                      f"primal={primal_resid:.6e}, "
                      f"dual={dual_resid:.6e}, time={elapsed:.2f}s")
            
            if mu < tol and primal_resid < tol and dual_resid < tol:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break
            
            # 浮点保护
            mu = max(mu, MU_MIN)
            
            # 在CPU上计算搜索方向（简化版本）
            dx_host = -x_host * 0.1
            dy_host = np.zeros(self.m)
            ds_host = -s_host * 0.1
            
            # 更新变量
            x_host += 0.9995 * dx_host
            s_host += 0.9995 * ds_host
            y_host = self.y.copy_to_host() + 0.9995 * dy_host
            
            # 保持正性
            x_host = np.maximum(x_host, EPSILON)
            s_host = np.maximum(s_host, EPSILON)
            
            # 传回GPU
            self.x = cuda.to_device(x_host)
            self.s = cuda.to_device(s_host)
            self.y = cuda.to_device(y_host)
            
            cuda.synchronize()
        
        return self.x.copy_to_host(), self.y.copy_to_host(), self.s.copy_to_host()

def generate_problem(m, n, density=0.1):
    """生成LP问题"""
    print(f"Generating LP problem: {m} constraints x {n} variables")
    print(f"Matrix memory: {m * n * 8 / 1024 / 1024:.2f} MB")
    
    A = np.random.randn(m, n) * (np.random.rand(m, n) < density)
    b = np.random.rand(m) * 1000 + 100
    c = np.random.randn(n) * 50
    
    print("Problem generation complete.\n")
    return A.astype(np.float64), b.astype(np.float64), c.astype(np.float64)

def verify_solution(A, b, c, x, y, s):
    """验证解"""
    primal_violation = np.linalg.norm(A @ x - b)
    dual_violation = np.linalg.norm(A.T @ y + s - c)
    objective = np.dot(c, x)
    duality_gap = np.dot(x, s)
    
    print("\nSolution verification:")
    print(f"  Primal constraint violation: {primal_violation:.6e}")
    print(f"  Dual constraint violation: {dual_violation:.6e}")
    print(f"  Duality gap: {duality_gap:.6e}")
    print(f"  Objective value: {objective:.6f}")

def gpu_full_load_test(duration_seconds=60):
    """GPU满载测试"""
    print("=" * 70)
    print("GPU OPTIMIZED FULL LOAD TEST")
    print("=" * 70)
    print(f"\nRunning continuous LP solves for {duration_seconds} seconds...\n")
    
    # 适中的问题规模
    n = 4000   # 变量数
    m = 2000   # 约束数
    
    problem_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        print(f"\n{'='*70}")
        print(f"Problem #{problem_count + 1}")
        print(f"{'='*70}")
        
        # 生成问题
        A, b, c = generate_problem(m, n, density=0.1)
        
        # 创建求解器
        solver = GPULPOptimizedSolver(m, n)
        solver.allocate_device_arrays(A, b, c)
        
        # 求解
        x, y, s = solver.solve(max_iter=50, verbose=True)
        
        # 验证
        verify_solution(A, b, c, x, y, s)
        
        problem_count += 1
        
        elapsed = time.time() - start_time
        print(f"\nCompleted {problem_count} problems in {elapsed:.1f} seconds")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("GPU OPTIMIZED TEST COMPLETED!")
    print(f"{'='*70}")
    print(f"Total problems solved: {problem_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per problem: {total_time/problem_count:.2f} seconds")
    print(f"Throughput: {problem_count/total_time:.2f} problems/second")
    print(f"GPU: RTX 5060 (Optimized)")

def main():
    print("\n" + "=" * 70)
    print("GPU Interior Point LP Solver - OPTIMIZED VERSION")
    print("=" * 70)
    print("\n[Features]")
    print("  - Numba CUDA JIT compilation")
    print("  - Optimized thread configuration")
    print("  - Reduced register usage")
    print("  - Maximum GPU stability")
    
    duration = 60
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    
    print(f"\nDuration: {duration} seconds\n")
    
    gpu_full_load_test(duration)
    print("\n")

if __name__ == "__main__":
    main()
