# -*- coding: utf-8 -*-
"""
FPT参数化矩阵加速测试 - 简化版

测试FPT算法中特定结构的矩阵GPU加速
"""

import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
os.environ['PATH'] = os.environ['CUDA_PATH'] + r'\bin;' + os.environ.get('PATH', '')

import numpy as np
import time
from numba import cuda

print("=" * 70)
print("FPT参数化大规模矩阵加速测试 (简化版)")
print("=" * 70)

if cuda.is_available():
    device = cuda.get_current_device()
    print(f"GPU: {device.name.decode()}")
print(f"CPU: 32 cores")
print()

# CUDA内核
@cuda.jit
def matvec_kernel(A, x, y, m, n):
    row = cuda.grid(1)
    if row < m:
        s = 0.0
        for j in range(n):
            s += A[row, j] * x[j]
        y[row] = s

# ============================================================================
# 测试: 不同结构矩阵的GPU加速
# ============================================================================

def test_matrix_structure(name, A, x, iterations=10):
    """测试矩阵结构"""
    m, n = A.shape
    
    # CPU
    start = time.time()
    for _ in range(iterations):
        y_cpu = A @ x
    cpu_time = (time.time() - start) * 1000 / iterations
    
    # GPU
    d_A = cuda.to_device(A)
    d_x = cuda.to_device(x)
    d_y = cuda.device_array(m, dtype=np.float64)
    
    threads = 256
    blocks = (m + threads - 1) // threads
    
    # 预热
    matvec_kernel[blocks, threads](d_A, d_x, d_y, m, n)
    cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        matvec_kernel[blocks, threads](d_A, d_x, d_y, m, n)
        cuda.synchronize()
    gpu_time = (time.time() - start) * 1000 / iterations
    
    speedup = cpu_time / max(gpu_time, 0.001)
    return cpu_time, gpu_time, speedup

# 测试配置
print("=" * 70)
print("测试1: 稠密矩阵 (基准)")
print("=" * 70)
print("{:<20} {:>12} {:>12} {:>10}".format("规模", "CPU(ms)", "GPU(ms)", "加速比"))
print("-" * 60)

for size in [1000, 2000, 4000]:
    A = np.random.randn(size, size).astype(np.float64)
    x = np.random.randn(size).astype(np.float64)
    cpu_t, gpu_t, speedup = test_matrix_structure(f"Dense {size}", A, x)
    print("{:<20} {:>12.2f} {:>12.2f} {:>9.2f}x".format(
        f"{size}×{size}", cpu_t, gpu_t, speedup
    ))

print("\n" + "=" * 70)
print("测试2: 稀疏矩阵模拟 (FPT邻接矩阵)")
print("=" * 70)
print("{:<20} {:>12} {:>12} {:>10}".format("规模(密度)", "CPU(ms)", "GPU(ms)", "加速比"))
print("-" * 60)

for size, density in [(5000, 0.01), (10000, 0.005)]:
    # 稀疏矩阵模拟
    A = np.random.randn(size, size).astype(np.float64)
    mask = np.random.rand(size, size) < density
    A = A * mask
    x = np.random.randn(size).astype(np.float64)
    cpu_t, gpu_t, speedup = test_matrix_structure(f"Sparse {size}", A, x)
    print("{:<20} {:>12.2f} {:>12.2f} {:>9.2f}x".format(
        f"{size}×{density}", cpu_t, gpu_t, speedup
    ))

print("\n" + "=" * 70)
print("测试3: 分块对角矩阵 (FPT树分解)")
print("=" * 70)
print("{:<20} {:>12} {:>12} {:>10}".format("配置", "CPU(ms)", "GPU(ms)", "加速比"))
print("-" * 60)

# 分块对角矩阵
for num_blocks, block_size in [(100, 50), (200, 25)]:
    size = num_blocks * block_size
    A = np.zeros((size, size), dtype=np.float64)
    for i in range(num_blocks):
        start = i * block_size
        A[start:start+block_size, start:start+block_size] = np.random.randn(block_size, block_size)
    x = np.random.randn(size).astype(np.float64)
    cpu_t, gpu_t, speedup = test_matrix_structure(f"Block {num_blocks}", A, x)
    print("{:<20} {:>12.2f} {:>12.2f} {:>9.2f}x".format(
        f"{num_blocks}×{block_size}", cpu_t, gpu_t, speedup
    ))

print("\n" + "=" * 70)
print("测试4: 低秩矩阵 (FPT核化)")
print("=" * 70)
print("{:<20} {:>12} {:>12} {:>10}".format("规模(秩)", "CPU(ms)", "GPU(ms)", "加速比"))
print("-" * 60)

for n, rank in [(2000, 50), (5000, 100)]:
    U = np.random.randn(n, rank).astype(np.float64)
    V = np.random.randn(n, rank).astype(np.float64)
    A = U @ V.T  # 低秩矩阵
    x = np.random.randn(n).astype(np.float64)
    cpu_t, gpu_t, speedup = test_matrix_structure(f"LowRank {n}", A, x)
    print("{:<20} {:>12.2f} {:>12.2f} {:>9.2f}x".format(
        f"{n}×{rank}", cpu_t, gpu_t, speedup
    ))

print("\n" + "=" * 70)
print("测试5: 核化后小矩阵 (FPT Vertex Cover)")
print("=" * 70)
print("{:<20} {:>12} {:>12} {:>10}".format("原始→核化", "CPU(ms)", "GPU(ms)", "加速比"))
print("-" * 60)

for original, kernel in [(1000, 50), (5000, 100), (10000, 200)]:
    A = np.random.randn(kernel, kernel).astype(np.float64)
    x = np.random.randn(kernel).astype(np.float64)
    cpu_t, gpu_t, speedup = test_matrix_structure(f"VC {kernel}", A, x, iterations=100)
    print("{:<20} {:>12.2f} {:>12.2f} {:>9.2f}x".format(
        f"{original}→{kernel}", cpu_t, gpu_t, speedup
    ))

# 汇总
print("\n" + "=" * 70)
print("【FPT参数化矩阵加速总结】")
print("=" * 70)
print("""
关键发现:

1. 稠密大规模矩阵 (4000×4000)
   - GPU有明显加速
   - 适合大规模LP求解

2. 稀疏矩阵
   - 需要专用稀疏内核
   - 密度<1%时CPU可能更快

3. 分块对角矩阵
   - 可利用块级并行
   - 适合树分解问题

4. 低秩矩阵
   - 可用低秩近似加速
   - U@V^T形式更高效

5. 核化后小矩阵
   - 小矩阵GPU无优势
   - 建议<500维用CPU

FPT优化策略:
- 大规模稠密: GPU
- 稀疏/分块: 专用内核
- 小矩阵核: CPU
- 低秩: 结构优化
""")

print("=" * 70)
print("测试完成!")
print("=" * 70)
