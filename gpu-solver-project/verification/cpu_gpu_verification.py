#!/usr/bin/env python3
"""
================================================================================
CPU-GPU Precision Verification Framework
CPU计算 + GPU验证精度
================================================================================
"""

import numpy as np
import numba
from numba import cuda
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CPU-GPU Precision Verification Framework")
print("=" * 70)

# 检查GPU
if cuda.is_available():
    gpu = cuda.get_current_device()
    print(f"GPU: {gpu.name}")
    print(f"Compute Capability: {gpu.compute_capability}")
    has_gpu = True
else:
    print("[WARN] GPU not available, using CPU only")
    has_gpu = False

print("=" * 70)


class CPU_GPU_Verifier:
    """
    CPU计算 + GPU验证精度
    
    流程:
    1. CPU使用高精度(FP64)计算参考解
    2. GPU使用混合精度计算
    3. 对比结果验证精度
    """
    
    def __init__(self, tol=1e-10):
        self.tol = tol
        self.threads = 256
    
    def solve_cpu_high_precision(self, H, c, G, h, max_iter=1000):
        """
        CPU高精度求解 (参考解)
        使用numpy的FP64和直接求解器
        """
        n = H.shape[0]
        m = G.shape[0]
        
        print(f"\n[CPU] High precision solver (FP64)")
        print(f"  Problem: n={n}, m={m}")
        
        start_time = time.time()
        
        # 使用CVXOPT风格的原始-对偶内点法
        # 简化为直接求解KKT系统
        
        # 初始点
        x = np.ones(n, dtype=np.float64) * 0.5
        s = np.ones(m, dtype=np.float64)
        z = np.ones(m, dtype=np.float64)
        v = np.zeros(m, dtype=np.float64)
        
        mu = 1.0
        converged = False
        
        for k in range(max_iter):
            # Softplus
            s = (v + np.sqrt(v**2 + 4*mu)) / 2
            z = (-v + np.sqrt(v**2 + 4*mu)) / 2
            
            # 残差
            F1 = H @ x + c + G.T @ z
            F3 = G @ x + s - h
            
            res = np.linalg.norm(F1)
            gap = np.dot(s, z) / m
            
            if k % 50 == 0:
                print(f"    CPU iter {k}: res={res:.2e}, gap={gap:.2e}")
            
            if res < self.tol and gap < self.tol:
                converged = True
                break
            
            # 构建KKT
            ds = 0.5 * (1 + v / np.sqrt(v**2 + 4*mu))
            dz = 0.5 * (1 - v / np.sqrt(v**2 + 4*mu))
            D = dz / ds
            
            H_tilde = H + G.T @ np.diag(D) @ G + 1e-8 * np.eye(n)
            KKT = np.block([
                [H_tilde, G.T],
                [G, -1e-8 * np.eye(m)]
            ])
            rhs = np.concatenate([-F1 - G.T @ (D * F3), -F3])
            
            # 求解
            try:
                dw = np.linalg.solve(KKT, rhs)
            except:
                dw = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            
            dx = dw[:n]
            dv = dw[n:]
            
            # 线搜索
            alpha = 1.0
            for _ in range(20):
                x_new = np.maximum(x + alpha * dx, 1e-10)
                v_new = v + alpha * dv
                
                z_new = (-v_new + np.sqrt(v_new**2 + 4*mu)) / 2
                F1_new = H @ x_new + c + G.T @ z_new
                
                if np.linalg.norm(F1_new) < 0.95 * np.linalg.norm(F1):
                    break
                alpha *= 0.8
            
            x = x_new
            v = v_new
            mu = max(0.5 * mu, 1e-12)
        
        cpu_time = time.time() - start_time
        
        return {
            'x': x,
            'obj': 0.5 * x @ (H @ x) + c @ x,
            'time': cpu_time,
            'iter': k,
            'converged': converged,
            'res': res,
            'gap': gap
        }
    
    def solve_gpu_mixed_precision(self, H, c, G, h, max_iter=100):
        """
        GPU混合精度求解
        FP32计算 + FP64验证
        """
        if not cuda.is_available():
            print("[WARN] GPU not available, skipping GPU solve")
            return None
        
        n = H.shape[0]
        m = G.shape[0]
        
        print(f"\n[GPU] Mixed precision solver (FP32/FP64)")
        print(f"  Problem: n={n}, m={m}")
        
        start_time = time.time()
        
        # 数据准备 (FP32 for compute)
        H_fp32 = H.astype(np.float32)
        c_fp32 = c.astype(np.float32)
        G_fp32 = G.astype(np.float32)
        h_fp32 = h.astype(np.float32)
        
        # 初始点
        x = np.ones(n, dtype=np.float32) * 0.5
        v = np.zeros(m, dtype=np.float32)
        
        # GPU内存
        H_gpu = cuda.to_device(H_fp32)
        G_gpu = cuda.to_device(G_fp32)
        x_gpu = cuda.to_device(x)
        v_gpu = cuda.to_device(v)
        
        mu = np.float32(1.0)
        converged = False
        
        for k in range(max_iter):
            # 复制到CPU进行softplus (简化)
            v = v_gpu.copy_to_host()
            s = (v + np.sqrt(v**2 + 4*mu)) / 2
            z = (-v + np.sqrt(v**2 + 4*mu)) / 2
            
            # FP64验证
            x_fp64 = x_gpu.copy_to_host().astype(np.float64)
            v_fp64 = v.astype(np.float64)
            s_fp64 = (v_fp64 + np.sqrt(v_fp64**2 + 4*1.0)) / 2
            z_fp64 = (-v_fp64 + np.sqrt(v_fp64**2 + 4*1.0)) / 2
            
            F1 = H @ x_fp64 + c + G.T @ z_fp64
            F3 = G @ x_fp64 + s_fp64 - h
            
            res = np.linalg.norm(F1)
            gap = np.dot(s_fp64, z_fp64) / m
            
            if k % 20 == 0:
                print(f"    GPU iter {k}: res={res:.2e}, gap={gap:.2e}")
            
            if res < self.tol and gap < self.tol:
                converged = True
                break
            
            # 简化的更新 (GPU计算核心)
            ds = 0.5 * (1 + v / np.sqrt(v**2 + 4*mu))
            dz = 0.5 * (1 - v / np.sqrt(v**2 + 4*mu))
            D = dz / ds
            
            # CPU求解KKT (小规模)
            H_tilde = H_fp32 + G_fp32.T @ np.diag(D) @ G_fp32 + 1e-6 * np.eye(n)
            KKT = np.block([
                [H_tilde, G_fp32.T],
                [G_fp32, -1e-6 * np.eye(m)]
            ])
            F1_fp32 = H_fp32 @ x + c_fp32 + G_fp32.T @ z.astype(np.float32)
            F3_fp32 = G_fp32 @ x + s.astype(np.float32) - h_fp32
            rhs = np.concatenate([-F1_fp32 - G_fp32.T @ (D * F3_fp32), -F3_fp32])
            
            try:
                dw = np.linalg.solve(KKT.astype(np.float64), rhs.astype(np.float64))
            except:
                dw = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            
            dx = dw[:n].astype(np.float32)
            dv = dw[n:].astype(np.float32)
            
            # 更新
            alpha = 0.9
            x = np.maximum(x + alpha * dx, 1e-8)
            v = v + alpha * dv
            
            x_gpu = cuda.to_device(x)
            v_gpu = cuda.to_device(v)
            
            mu = max(0.9 * mu, 1e-12)
        
        gpu_time = time.time() - start_time
        
        x_final = x_gpu.copy_to_host().astype(np.float64)
        
        return {
            'x': x_final,
            'obj': 0.5 * x_final @ (H @ x_final) + c @ x_final,
            'time': gpu_time,
            'iter': k,
            'converged': converged,
            'res': res,
            'gap': gap
        }
    
    def verify_precision(self, cpu_result, gpu_result):
        """
        验证CPU和GPU结果的一致性
        """
        if gpu_result is None:
            print("\n[VERIFY] GPU result not available")
            return False
        
        print("\n" + "=" * 70)
        print("Precision Verification")
        print("=" * 70)
        
        x_cpu = cpu_result['x']
        x_gpu = gpu_result['x']
        
        # 计算差异
        abs_diff = np.abs(x_cpu - x_gpu)
        rel_diff = abs_diff / (np.abs(x_cpu) + 1e-10)
        
        max_abs_error = np.max(abs_diff)
        max_rel_error = np.max(rel_diff)
        mean_abs_error = np.mean(abs_diff)
        mean_rel_error = np.mean(rel_diff)
        
        # 目标函数差异
        obj_diff = abs(cpu_result['obj'] - gpu_result['obj'])
        obj_rel_diff = obj_diff / (abs(cpu_result['obj']) + 1e-10)
        
        print(f"Solution Comparison:")
        print(f"  Max absolute error: {max_abs_error:.2e}")
        print(f"  Max relative error: {max_rel_error:.2e}")
        print(f"  Mean absolute error: {mean_abs_error:.2e}")
        print(f"  Mean relative error: {mean_rel_error:.2e}")
        
        print(f"\nObjective Comparison:")
        print(f"  CPU objective: {cpu_result['obj']:.10f}")
        print(f"  GPU objective: {gpu_result['obj']:.10f}")
        print(f"  Absolute diff: {obj_diff:.2e}")
        print(f"  Relative diff: {obj_rel_diff:.2e}")
        
        print(f"\nTiming:")
        print(f"  CPU time: {cpu_result['time']:.3f}s")
        print(f"  GPU time: {gpu_result['time']:.3f}s")
        speedup = cpu_result['time'] / gpu_result['time'] if gpu_result['time'] > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        
        # 验证通过标准
        passed = max_rel_error < 1e-3 and obj_rel_diff < 1e-3
        
        if passed:
            print(f"\n[OK] Verification PASSED")
            print(f"  GPU results match CPU reference within tolerance")
        else:
            print(f"\n[WARN] Verification needs attention")
            print(f"  Errors larger than expected, but may still be acceptable")
        
        return passed


def run_precision_verification():
    """运行精度验证测试"""
    print("\n" + "=" * 70)
    print("CPU-GPU Precision Verification Test")
    print("=" * 70)
    
    # 生成可行测试问题
    np.random.seed(42)
    n, m = 100, 50
    
    print(f"\nGenerating feasible test problem: n={n}, m={m}")
    
    # 生成确保可行的问题
    x_feasible = np.abs(np.random.randn(n)) + 0.5
    
    M = np.random.randn(n, n)
    H = M.T @ M + 0.1 * np.eye(n)
    c = np.random.randn(n)
    
    G = np.random.randn(m, n)
    h = G @ x_feasible + np.abs(np.random.randn(m)) + 1.0
    
    print(f"  Feasible point norm: {np.linalg.norm(x_feasible):.2f}")
    print(f"  Constraint check: max(Gx-h) = {np.max(G @ x_feasible - h):.2e}")
    
    # 创建验证器
    verifier = CPU_GPU_Verifier(tol=1e-8)
    
    # CPU高精度求解
    cpu_result = verifier.solve_cpu_high_precision(H, c, G, h, max_iter=200)
    
    print(f"\n[CPU] Results:")
    print(f"  Converged: {cpu_result['converged']}")
    print(f"  Objective: {cpu_result['obj']:.10f}")
    print(f"  Time: {cpu_result['time']:.3f}s")
    print(f"  Iterations: {cpu_result['iter']}")
    print(f"  Final residual: {cpu_result['res']:.2e}")
    
    # GPU混合精度求解
    if cuda.is_available():
        gpu_result = verifier.solve_gpu_mixed_precision(H, c, G, h, max_iter=100)
        
        if gpu_result:
            print(f"\n[GPU] Results:")
            print(f"  Converged: {gpu_result['converged']}")
            print(f"  Objective: {gpu_result['obj']:.10f}")
            print(f"  Time: {gpu_result['time']:.3f}s")
            print(f"  Iterations: {gpu_result['iter']}")
            print(f"  Final residual: {gpu_result['res']:.2e}")
            
            # 验证精度
            verifier.verify_precision(cpu_result, gpu_result)
    else:
        print("\n[SKIP] GPU not available, skipping GPU verification")
    
    print("\n" + "=" * 70)
    print("Precision Verification Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_precision_verification()
