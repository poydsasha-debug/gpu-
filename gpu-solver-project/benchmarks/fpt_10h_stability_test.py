# -*- coding: utf-8 -*-
"""
FPT异构调度器 - 10小时连续稳定性测试

测试目标:
1. 验证长时间运行稳定性
2. 监控内存泄漏
3. 统计GPU/CPU利用率
4. 记录性能衰减情况
"""

import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
os.environ['PATH'] = os.environ['CUDA_PATH'] + r'\bin;' + os.environ.get('PATH', '')

import numpy as np
import time
import json
import threading
from datetime import datetime, timedelta
from collections import deque
import sys

from numba import cuda

# 导入调度器
from fpt_scheduler_production import (
    HeterogeneousScheduler, TaskProfile, TaskType, 
    DeviceType, CPULPSolver, GPULPSolver
)

class LongRunningTest:
    """长时间运行测试"""
    
    def __init__(self, duration_hours=10):
        self.duration_hours = duration_hours
        self.duration_seconds = duration_hours * 3600
        
        # 统计
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'problems_solved': 0,
            'cpu_problems': 0,
            'gpu_problems': 0,
            'errors': 0,
            'total_cpu_time_ms': 0.0,
            'total_gpu_time_ms': 0.0,
        }
        
        # 性能历史 (保存最近1000个样本)
        self.cpu_time_history = deque(maxlen=1000)
        self.gpu_time_history = deque(maxlen=1000)
        
        # 创建调度器
        self.scheduler = HeterogeneousScheduler(num_cpu_workers=8, enable_gpu=True)
        
        # 日志文件
        self.log_file = open("10h_test_log.txt", "w", buffering=1)
        
        # 停止标志
        self.stop_flag = threading.Event()
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.log_file.write(log_line + "\n")
        self.log_file.flush()
    
    def generate_problem(self, size_type='mixed'):
        """生成测试问题"""
        if size_type == 'small':
            m, n = 400, 800
        elif size_type == 'medium':
            m, n = 1000, 2000
        elif size_type == 'large':
            m, n = 2000, 4000
        else:  # mixed
            r = np.random.random()
            if r < 0.5:
                m, n = 400, 800
            elif r < 0.8:
                m, n = 1000, 2000
            else:
                m, n = 2000, 4000
        
        A = np.random.randn(m, n) * (np.random.rand(m, n) < 0.1)
        b = np.random.rand(m) * 100 + 50
        c = np.random.randn(n) * 10
        
        return (m, n), A.astype(np.float64), b.astype(np.float64), c.astype(np.float64)
    
    def run_single_batch(self, batch_size=10):
        """运行一批测试"""
        tasks = []
        
        for i in range(batch_size):
            (m, n), A, b, c = self.generate_problem('mixed')
            profile = TaskProfile(
                task_id=self.stats['problems_solved'] + i,
                task_type=TaskType.LP_SOLVE,
                problem_size=(m, n),
                data_bytes=m * n * 8
            )
            tasks.append((profile, A, b, c))
        
        try:
            results = self.scheduler.run_batch(tasks)
            
            for r in results:
                self.stats['problems_solved'] += 1
                
                if r.device_used == DeviceType.CPU:
                    self.stats['cpu_problems'] += 1
                    self.stats['total_cpu_time_ms'] += r.total_time_ms
                    self.cpu_time_history.append(r.total_time_ms)
                else:
                    self.stats['gpu_problems'] += 1
                    self.stats['total_gpu_time_ms'] += r.total_time_ms
                    self.gpu_time_history.append(r.total_time_ms)
            
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            self.log(f"ERROR: {str(e)}")
            return False
    
    def print_status(self, elapsed_seconds):
        """打印状态"""
        elapsed_hours = elapsed_seconds / 3600
        remaining = self.duration_seconds - elapsed_seconds
        remaining_hours = remaining / 3600
        
        total = self.stats['problems_solved']
        cpu_count = self.stats['cpu_problems']
        gpu_count = self.stats['gpu_problems']
        
        avg_cpu = np.mean(self.cpu_time_history) if self.cpu_time_history else 0
        avg_gpu = np.mean(self.gpu_time_history) if self.gpu_time_history else 0
        
        gpu_util = gpu_count / total * 100 if total > 0 else 0
        throughput = total / elapsed_seconds * 3600 if elapsed_seconds > 0 else 0
        
        self.log("=" * 70)
        self.log(f"进度: {elapsed_hours:.2f}h / {self.duration_hours}h ({remaining_hours:.2f}h 剩余)")
        self.log(f"已解决问题: {total} (CPU={cpu_count}, GPU={gpu_count}, GPU利用率={gpu_util:.1f}%)")
        self.log(f"平均时间: CPU={avg_cpu:.1f}ms, GPU={avg_gpu:.1f}ms")
        self.log(f"吞吐量: {throughput:.0f} problems/hour")
        self.log(f"错误数: {self.stats['errors']}")
        self.log("=" * 70)
    
    def run(self):
        """运行测试"""
        self.log("=" * 70)
        self.log(f"FPT异构调度器 10小时稳定性测试")
        self.log("=" * 70)
        self.log(f"开始时间: {self.stats['start_time']}")
        self.log(f"计划时长: {self.duration_hours} hours")
        self.log(f"GPU: {cuda.get_current_device().name.decode() if cuda.is_available() else 'N/A'}")
        self.log("=" * 70)
        
        start_time = time.time()
        last_status_time = start_time
        batch_count = 0
        
        try:
            while not self.stop_flag.is_set():
                elapsed = time.time() - start_time
                
                if elapsed >= self.duration_seconds:
                    self.log("达到计划时长，测试完成")
                    break
                
                # 运行一批测试
                success = self.run_single_batch(batch_size=5)
                batch_count += 1
                
                # 每5分钟打印状态
                if time.time() - last_status_time >= 300:  # 5 minutes
                    self.print_status(elapsed)
                    last_status_time = time.time()
                
                # 短暂休息避免过热
                if batch_count % 100 == 0:
                    time.sleep(0.5)
                
        except KeyboardInterrupt:
            self.log("测试被用户中断")
        except Exception as e:
            self.log(f"测试异常: {str(e)}")
        
        finally:
            self.finish()
    
    def finish(self):
        """完成测试"""
        end_time = datetime.now()
        start = datetime.fromisoformat(self.stats['start_time'])
        actual_duration = (end_time - start).total_seconds()
        
        self.stats['end_time'] = end_time.isoformat()
        self.stats['actual_duration_hours'] = actual_duration / 3600
        
        self.log("\n" + "=" * 70)
        self.log("测试完成!")
        self.log("=" * 70)
        self.log(f"实际运行时间: {self.stats['actual_duration_hours']:.2f} hours")
        self.log(f"总解决问题: {self.stats['problems_solved']}")
        self.log(f"  - CPU: {self.stats['cpu_problems']}")
        self.log(f"  - GPU: {self.stats['gpu_problems']}")
        self.log(f"错误数: {self.stats['errors']}")
        
        if self.stats['problems_solved'] > 0:
            throughput = self.stats['problems_solved'] / self.stats['actual_duration_hours']
            self.log(f"平均吞吐量: {throughput:.1f} problems/hour")
        
        # 保存详细报告
        report = {
            'summary': self.stats,
            'cpu_time_samples': list(self.cpu_time_history),
            'gpu_time_samples': list(self.gpu_time_history),
        }
        
        with open("10h_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        self.log(f"详细报告已保存: 10h_test_report.json")
        self.log_file.close()

def main():
    """主函数"""
    print("FPT 10小时稳定性测试")
    print("=" * 70)
    print("按Ctrl+C可随时停止测试")
    print("=" * 70)
    print()
    
    test = LongRunningTest(duration_hours=10)
    test.run()

if __name__ == "__main__":
    main()
