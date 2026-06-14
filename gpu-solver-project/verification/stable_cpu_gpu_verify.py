#!/usr/bin/env python3
"""
================================================================================
CPU-GPU Precision Verification - Stable Version
使用经过验证的方法进行精度对比
================================================================================
"""

import numpy as np
import numba
from numba import cuda
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CPU-GPU Precision Verification (Stable Version)")
print("=" * 70)

# 检查GPU
has_gpu = cuda.is_available()
if has_gpu:
    gpu = cuda.get_current_device()
    print(f"GPU: {gpu.name}")
    print(f"Compute Capability: {gpu.compute_capability}")
else:
    print("GPU: Not available")
print("=" * 70)


# CUDA核函数
@cuda.jit
def gpu_vec_add(a, b, c, n):
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] + b[i]

@cuda.jit
def gpu_vec_scale(a, scale, b, n):
    i = cuda.grid(1)
    if i < n:
        b[i] = a[i] * scale

@cuda.jit
def gpu_matvec(A, x, y, m, n):
    i = cuda.grid(1)
    if i < m:
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        y[i] = s


class StableCPU_GPU_Verifier:
    """
    稳定的CPU-GPU精度验证
    使用简单但可靠的算法
    """
    
    def __init__(self):
        self.threads = 256
    
    def solve_simple_gradient_cpu(self, H, c, max_iter=1000, lr=0.01):
        """
        CPU梯度下降求解 min 1/2 x^T H x + c^T x
        简单但稳定
        """
        n = H.shape[0]
        x = np.zeros(n, dtype=np.float64)
        
        print(f"\n[CPU] Gradient descent (FP64)")
        print(f"  Problem size: n={n}")
        
        start = time.time()
        
        for k in range(max_iter):
            # 梯度: Hx + c
            grad = H @ x + c
            
            # 更新
            x = x - lr * grad
            
            # 收敛检查
            grad_norm = np.linalg.norm(grad)
            if k % 100 == 0:
                obj = 0.5 * x @ (H @ x) + c @ x
                print(f"    Iter {k}: obj={obj:.6f}, grad_norm={grad_norm:.2e}")
            
            if grad_norm < 1e-6:
                break
        
        cpu_time = time.time() - start
        obj = 0.5 * x @ (H @ x) + c @ x
        
        return {
            'x': x,
            'obj': obj,
            'time': cpu_time,
            'iter': k,
            'grad_norm': grad_norm
        }
    
    def solve_simple_gradient_gpu(self, H, c, max_iter=1000, lr=0.01):
        """
        GPU梯度下降求解
        使用CUDA核函数
        """
        if not has_gpu:
            return None
        
        n = H.shape[0]
        blocks = (n + self.threads - 1) // self.threads
        
        # FP32 for GPU computation
        H_fp32 = H.astype(np.float32)
        c_fp32 = c.astype(np.float32)
        
        # GPU内存
        H_gpu = cuda.to_device(H_fp32)
        c_gpu = cuda.to_device(c_fp32)
        x_gpu = cuda.device_array(n, dtype=np.float32)
        grad_gpu = cuda.device_array(n, dtype=np.float32)
        temp_gpu = cuda.device_array(n, dtype=np.float32)
        
        # 初始化x=0
        x = np.zeros(n, dtype=np.float32)
        x_gpu = cuda.to_device(x)
        
        print(f"\n[GPU] Gradient descent (FP32)")
        print(f"  Problem size: n={n}")
        print(f"  GPU config: {blocks} blocks x {self.threads} threads")
        
        start = time.time()
        
        for k in range(max_iter):
            # 计算梯度: grad = H @ x + c
            # H @ x
            gpu_matvec[blocks, self.threads](H_gpu, x_gpu, grad_gpu, n, n)
            cuda.synchronize()
            
            # grad + c
            gpu_vec_add[blocks, self.threads](grad_gpu, c_gpu, grad_gpu, n)
            cuda.synchronize()
            
            # 更新: x = x - lr * grad
            # temp = -lr * grad
            gpu_vec_scale[blocks, self.threads](grad_gpu, -lr, temp_gpu, n)
            cuda.synchronize()
            
            # x = x + temp
            gpu_vec_add[blocks, self.threads](x_gpu, temp_gpu, x_gpu, n)
            cuda.synchronize()
            
            # 每100次迭代检查收敛 (复制到CPU)
            if k % 100 == 0:
                x_host = x_gpu.copy_to_host().astype(np.float64)
                grad_host = H @ x_host + c
                grad_norm = np.linalg.norm(grad_host)
                obj = 0.5 * x_host @ (H @ x_host) + c @ x_host
                print(f"    Iter {k}: obj={obj:.6f}, grad_norm={grad_norm:.2e}")
                
                if grad_norm < 1e-5:
                    break
        
        gpu_time = time.time() - start
        
        # 最终结果
        x_final = x_gpu.copy_to_host().astype(np.float64)
        obj = 0.5 * x_final @ (H @ x_final) + c @ x_final
        
        return {
            'x': x_final,
            'obj': obj,
            'time': gpu_time,
            'iter': k
        }
    
    def verify(self, cpu_result, gpu_result):
        """验证精度"""
        print("\n" + "=" * 70)
        print("Precision Verification")
        print("=" * 70)
        
        x_cpu = cpu_result['x']
        x_gpu = gpu_result['x']
        
        # 误差分析
        abs_error = np.abs(x_cpu - x_gpu)
        rel_error = abs_error / (np.abs(x_cpu) + 1e-10)
        
        print(f"Solution Error Analysis:")
        print(f"  Max absolute error: {np.max(abs_error):.2e}")
        print(f"  Mean absolute error: {np.mean(abs_error):.2e}")
        print(f"  Max relative error: {np.max(rel_error):.2e}")
        print(f"  Mean relative error: {np.mean(rel_error):.2e}")
        
        print(f"\nObjective Comparison:")
        print(f"  CPU: {cpu_result['obj']:.10f}")
        print(f"  GPU: {gpu_result['obj']:.10f}")
        obj_error = abs(cpu_result['obj'] - gpu_result['obj'])
        obj_rel_error = obj_error / (abs(cpu_result['obj']) + 1e-10)
        print(f"  Absolute diff: {obj_error:.2e}")
        print(f"  Relative diff: {obj_rel_error:.2e}")
        
        print(f"\nPerformance:")
        print(f"  CPU time: {cpu_result['time']:.3f}s")
        print(f"  GPU time: {gpu_result['time']:.3f}s")
        speedup = cpu_result['time'] / gpu_result['time']
        print(f"  Speedup: {speedup:.2f}x")
        
        # 验证标准
        passed = np.max(rel_error) < 1e-2 and obj_rel_error < 1e-2
        
        if passed:
            print(f"\n[OK] Verification PASSED")
            print(f"  GPU results match CPU within 1% tolerance")
        else:
            print(f"\n[WARN] Large errors detected")
            print(f"  This may be due to FP32 vs FP64 precision difference")
        
        return passed


def run_stable_verification():
    """运行稳定的验证测试"""
    print("\n" + "=" * 70)
    print("Stable CPU-GPU Precision Verification")
    print("=" * 70)
    
    # 生成测试问题
    np.random.seed(42)
    n = 1000
    
    print(f"\nGenerating test problem: n={n}")
    
    # 正定Hessian
    M = np.random.randn(n, n)
    H = M.T @ M + 0.1 * np.eye(n)
    c = np.random.randn(n)
    
    print(f"  H condition number: {np.linalg.cond(H):.2e}")
    
    # 创建验证器
    verifier = StableCPU_GPU_Verifier()
    
    # CPU求解
    cpu_result = verifier.solve_simple_gradient_cpu(H, c, max_iter=1000, lr=0.001)
    
    print(f"\n[CPU] Summary:")
    print(f"  Objective: {cpu_result['obj']:.10f}")
    print(f"  Time: {cpu_result['time']:.3f}s")
    print(f"  Iterations: {cpu_result['iter']}")
    
    # GPU求解
    if has_gpu:
        gpu_result = verifier.solve_simple_gradient_gpu(H, c, max_iter=1000, lr=0.001)
        
        if gpu_result:
            print(f"\n[GPU] Summary:")
            print(f"  Objective: {gpu_result['obj']:.10f}")
            print(f"  Time: {gpu_result['time']:.3f}s")
            print(f"  Iterations: {gpu_result['iter']}")
            
            # 验证
            verifier.verify(cpu_result, gpu_result)
    else:
        print("\n[SKIP] GPU not available")
    
    print("\n" + "=" * 70)
    print("Verification Complete")
    print("=" * 70)
    print("\nKey Findings:")
    print("1. CPU uses FP64 (double precision)")
    print("2. GPU uses FP32 (single precision)")
    print("3. Small differences expected due to precision")
    print("4. Both should converge to similar solutions")


if __name__ == "__main__":
    run_stable_verification()
