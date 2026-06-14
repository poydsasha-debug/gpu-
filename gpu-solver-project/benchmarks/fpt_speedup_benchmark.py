# -*- coding: utf-8 -*-
"""
FPT算法加速测试 - 对比纯CPU vs 异构调度

测试内容：
1. Vertex Cover (不同规模)
2. 批量LP求解 (FPT子程序)
3. 综合加速比分析
"""

import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
os.environ['PATH'] = os.environ['CUDA_PATH'] + r'\bin;' + os.environ.get('PATH', '')

import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
from numba import cuda

# 导入调度器
from fpt_scheduler_production import (
    HeterogeneousScheduler, TaskProfile, TaskType, 
    DeviceType, CPULPSolver, GPULPSolver
)

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    problem_size: str
    cpu_time_ms: float
    gpu_time_ms: float
    heterogeneous_time_ms: float
    cpu_only_speedup: float
    gpu_only_speedup: float
    heterogeneous_speedup: float
    gpu_utilization: float

class FPTBenchmark:
    """FPT算法基准测试"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.cpu_scheduler = HeterogeneousScheduler(num_cpu_workers=8, enable_gpu=False)
        self.gpu_scheduler = HeterogeneousScheduler(num_cpu_workers=1, enable_gpu=True)
        self.heterogeneous_scheduler = HeterogeneousScheduler(num_cpu_workers=8, enable_gpu=True)
    
    def benchmark_lp_batch(self, sizes: List[Tuple[int, int]], num_problems_per_size: int = 5):
        """测试批量LP求解"""
        print("=" * 70)
        print("FPT LP子程序加速测试")
        print("=" * 70)
        
        for m, n in sizes:
            print(f"\n问题规模: {m}×{n} (内存: {m*n*8/1024/1024:.1f} MB)")
            print("-" * 50)
            
            # 生成问题
            tasks = []
            for i in range(num_problems_per_size):
                A = np.random.randn(m, n) * (np.random.rand(m, n) < 0.1)
                b = np.random.rand(m) * 100 + 50
                c = np.random.randn(n) * 10
                
                profile = TaskProfile(
                    task_id=i,
                    task_type=TaskType.LP_SOLVE,
                    problem_size=(m, n),
                    data_bytes=m * n * 8
                )
                tasks.append((profile, A.astype(np.float64), b.astype(np.float64), c.astype(np.float64)))
            
            # 纯CPU测试
            print("  测试纯CPU...", end=" ")
            start = time.time()
            cpu_results = self.cpu_scheduler.run_batch(tasks)
            cpu_time = (time.time() - start) * 1000
            print(f"{cpu_time:.1f} ms")
            
            # 纯GPU测试（强制所有任务GPU）
            print("  测试纯GPU...", end=" ")
            start = time.time()
            gpu_results = []
            for profile, A, b, c in tasks:
                solver = GPULPSolver(m, n)
                solver.solve(A, b, c)
            gpu_time = (time.time() - start) * 1000
            print(f"{gpu_time:.1f} ms")
            
            # 异构调度测试
            print("  测试异构调度...", end=" ")
            start = time.time()
            het_results = self.heterogeneous_scheduler.run_batch(tasks)
            het_time = (time.time() - start) * 1000
            print(f"{het_time:.1f} ms")
            
            # 统计GPU利用率
            gpu_count = sum(1 for r in het_results if r.device_used == DeviceType.GPU)
            gpu_util = gpu_count / len(het_results) * 100
            
            # 计算加速比
            baseline = cpu_time
            result = BenchmarkResult(
                test_name="LP_Batch",
                problem_size=f"{m}×{n}",
                cpu_time_ms=cpu_time,
                gpu_time_ms=gpu_time,
                heterogeneous_time_ms=het_time,
                cpu_only_speedup=1.0,
                gpu_only_speedup=baseline / max(gpu_time, 0.001),
                heterogeneous_speedup=baseline / max(het_time, 0.001),
                gpu_utilization=gpu_util
            )
            self.results.append(result)
            
            print(f"\n  加速比:")
            print(f"    纯GPU vs CPU: {result.gpu_only_speedup:.2f}x")
            print(f"    异构 vs CPU: {result.heterogeneous_speedup:.2f}x")
            print(f"    GPU利用率: {gpu_util:.1f}%")
    
    def benchmark_mixed_workload(self):
        """测试混合工作负载（模拟真实FPT场景）"""
        print("\n" + "=" * 70)
        print("混合工作负载测试 (模拟FPT实际场景)")
        print("=" * 70)
        
        # 模拟FPT求解过程中的LP调用
        # 包含：小核化问题(多) + 大松弛问题(少)
        tasks = []
        task_id = 0
        
        # 10个小问题 (核化后的子问题)
        for _ in range(10):
            m, n = 300, 600
            A = np.random.randn(m, n) * (np.random.rand(m, n) < 0.1)
            b = np.random.rand(m) * 100 + 50
            c = np.random.randn(n) * 10
            profile = TaskProfile(task_id=task_id, task_type=TaskType.LP_SOLVE,
                                problem_size=(m, n), data_bytes=m*n*8)
            tasks.append((profile, A.astype(np.float64), b.astype(np.float64), c.astype(np.float64)))
            task_id += 1
        
        # 3个中等问题 (LP松弛)
        for _ in range(3):
            m, n = 1000, 2000
            A = np.random.randn(m, n) * (np.random.rand(m, n) < 0.1)
            b = np.random.rand(m) * 100 + 50
            c = np.random.randn(n) * 10
            profile = TaskProfile(task_id=task_id, task_type=TaskType.LP_SOLVE,
                                problem_size=(m, n), data_bytes=m*n*8)
            tasks.append((profile, A.astype(np.float64), b.astype(np.float64), c.astype(np.float64)))
            task_id += 1
        
        # 1个大问题 (最终验证)
        m, n = 2000, 4000
        A = np.random.randn(m, n) * (np.random.rand(m, n) < 0.1)
        b = np.random.rand(m) * 100 + 50
        c = np.random.randn(n) * 10
        profile = TaskProfile(task_id=task_id, task_type=TaskType.LP_SOLVE,
                            problem_size=(m, n), data_bytes=m*n*8)
        tasks.append((profile, A.astype(np.float64), b.astype(np.float64), c.astype(np.float64)))
        
        print(f"\n工作负载: 10小 + 3中 + 1大 = 14个任务")
        print("-" * 50)
        
        # 纯CPU
        print("  纯CPU...", end=" ")
        start = time.time()
        cpu_results = self.cpu_scheduler.run_batch(tasks)
        cpu_time = (time.time() - start) * 1000
        print(f"{cpu_time:.1f} ms")
        
        # 纯GPU
        print("  纯GPU...", end=" ")
        start = time.time()
        for profile, A, b, c in tasks:
            m, n = profile.problem_size
            solver = GPULPSolver(m, n)
            solver.solve(A, b, c)
        gpu_time = (time.time() - start) * 1000
        print(f"{gpu_time:.1f} ms")
        
        # 异构
        print("  异构调度...", end=" ")
        start = time.time()
        het_results = self.heterogeneous_scheduler.run_batch(tasks)
        het_time = (time.time() - start) * 1000
        print(f"{het_time:.1f} ms")
        
        # 统计
        gpu_count = sum(1 for r in het_results if r.device_used == DeviceType.GPU)
        cpu_count = sum(1 for r in het_results if r.device_used == DeviceType.CPU)
        
        result = BenchmarkResult(
            test_name="Mixed_Workload",
            problem_size="10小+3中+1大",
            cpu_time_ms=cpu_time,
            gpu_time_ms=gpu_time,
            heterogeneous_time_ms=het_time,
            cpu_only_speedup=1.0,
            gpu_only_speedup=cpu_time / max(gpu_time, 0.001),
            heterogeneous_speedup=cpu_time / max(het_time, 0.001),
            gpu_utilization=gpu_count / len(het_results) * 100
        )
        self.results.append(result)
        
        print(f"\n  分配: CPU={cpu_count}, GPU={gpu_count}")
        print(f"  加速比:")
        print(f"    纯GPU vs CPU: {result.gpu_only_speedup:.2f}x")
        print(f"    异构 vs CPU: {result.heterogeneous_speedup:.2f}x")
    
    def benchmark_single_problem_scaling(self):
        """测试单问题规模扩展性"""
        print("\n" + "=" * 70)
        print("单问题规模扩展性测试")
        print("=" * 70)
        
        sizes = [
            (500, 1000, "小型"),
            (1000, 2000, "中型"),
            (2000, 4000, "大型"),
        ]
        
        for m, n, label in sizes:
            print(f"\n{label}问题: {m}×{n}")
            print("-" * 30)
            
            A = np.random.randn(m, n) * (np.random.rand(m, n) < 0.1)
            b = np.random.rand(m) * 100 + 50
            c = np.random.randn(n) * 10
            
            # CPU
            start = time.time()
            cpu_solver = CPULPSolver(m, n)
            cpu_solver.solve(A.astype(np.float64), b.astype(np.float64), c.astype(np.float64))
            cpu_time = (time.time() - start) * 1000
            
            # GPU
            start = time.time()
            gpu_solver = GPULPSolver(m, n)
            gpu_solver.solve(A.astype(np.float64), b.astype(np.float64), c.astype(np.float64))
            gpu_time = (time.time() - start) * 1000
            
            speedup = cpu_time / max(gpu_time, 0.001)
            
            print(f"  CPU: {cpu_time:.1f} ms")
            print(f"  GPU: {gpu_time:.1f} ms")
            print(f"  加速比: {speedup:.2f}x")
    
    def print_summary(self):
        """打印汇总报告"""
        print("\n" + "=" * 70)
        print("FPT加速测试汇总报告")
        print("=" * 70)
        
        print("\n{:<20} {:<15} {:>10} {:>10} {:>12} {:>10}".format(
            "测试", "规模", "CPU(ms)", "GPU(ms)", "异构(ms)", "加速比"
        ))
        print("-" * 70)
        
        for r in self.results:
            print("{:<20} {:<15} {:>10.1f} {:>10.1f} {:>12.1f} {:>9.2f}x".format(
                r.test_name, r.problem_size, r.cpu_time_ms, r.gpu_time_ms,
                r.heterogeneous_time_ms, r.heterogeneous_speedup
            ))
        
        # 计算平均加速比
        if self.results:
            avg_gpu_speedup = np.mean([r.gpu_only_speedup for r in self.results])
            avg_het_speedup = np.mean([r.heterogeneous_speedup for r in self.results])
            avg_gpu_util = np.mean([r.gpu_utilization for r in self.results])
            
            print("-" * 70)
            print(f"\n平均加速比:")
            print(f"  纯GPU vs CPU: {avg_gpu_speedup:.2f}x")
            print(f"  异构 vs CPU: {avg_het_speedup:.2f}x")
            print(f"  平均GPU利用率: {avg_gpu_util:.1f}%")
        
        # 导出JSON
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": {
                "cpu_cores": 32,
                "gpu": "RTX 5060 Laptop",
                "cuda_available": cuda.is_available()
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "problem_size": r.problem_size,
                    "cpu_time_ms": r.cpu_time_ms,
                    "gpu_time_ms": r.gpu_time_ms,
                    "heterogeneous_time_ms": r.heterogeneous_time_ms,
                    "speedup": r.heterogeneous_speedup,
                    "gpu_utilization": r.gpu_utilization
                }
                for r in self.results
            ]
        }
        
        with open("fpt_speedup_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n详细报告已保存: fpt_speedup_report.json")

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("FPT算法异构加速全面测试")
    print("=" * 70)
    print("\n系统配置:")
    print(f"  CPU: 32 cores")
    if cuda.is_available():
        device = cuda.get_current_device()
        print(f"  GPU: {device.name.decode()}")
    print()
    
    benchmark = FPTBenchmark()
    
    # 运行测试
    benchmark.benchmark_lp_batch([
        (400, 800),     # 小型
        (1000, 2000),   # 中型
        (2000, 4000),   # 大型
    ], num_problems_per_size=5)
    
    benchmark.benchmark_mixed_workload()
    benchmark.benchmark_single_problem_scaling()
    
    # 汇总
    benchmark.print_summary()
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
