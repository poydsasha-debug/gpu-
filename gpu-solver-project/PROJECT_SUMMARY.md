# GPU-Accelerated Linear Programming Solver

高性能GPU加速线性规划求解器项目包

## 项目结构

```
gpu-solver-project/
├── README.md                    # 项目说明文档
├── LICENSE                      # MIT许可证
├── requirements.txt             # Python依赖
│
├── core/                        # 核心求解器
│   ├── interior_point_lp_gpu_optimized.py    # 主GPU求解器 (15.3x加速)
│   ├── interior_point_lp_numba.py            # Numba CUDA实现
│   ├── gpu_full_load_lp.py                   # 全负载测试
│   ├── fpt_heterogeneous_scheduler.py        # 异构调度器
│   └── fpt_scheduler_production.py           # 生产级调度器
│
├── benchmarks/                  # 性能测试
│   ├── fpt_speedup_benchmark.py              # 加速比测试
│   ├── fpt_10h_stability_test.py             # 10小时稳定性测试
│   ├── fpt_param_matrix_simple.py            # 矩阵类型测试
│   ├── cpu_gpu_comparison.py                 # CPU vs GPU对比
│   └── fpt_gpu_acceleration_test.py          # GPU加速测试
│
├── verification/                # 正确性验证
│   ├── verify_correctness_final.py           # 机器精度验证
│   ├── check_gpu_fp_units.py                 # GPU能力检查
│   ├── cpu_gpu_verification.py               # 精度对比
│   └── stable_cpu_gpu_verify.py              # 稳定验证
│
└── docs/                        # 文档报告
    ├── CORRECTNESS_VERIFICATION_REPORT.md    # 正确性报告
    ├── GPU_FULL_LOAD_FINAL_REPORT.md         # 全负载报告
    ├── NUMBA_CUDA_SUCCESS_REPORT.md          # Numba CUDA报告
    ├── FP32_VERSION_REPORT.md                # FP32报告
    └── CPU_GPU精度对比报告.md                 # 精度对比报告
```

## 关键成果

### 性能
- **15.3x GPU加速** (vs CPU)
- **28.79 problems/sec** (Medium规模)
- **353,410问题** 10小时稳定运行

### 精度
- **相对误差: 3.33e-08** (CPU vs GPU)
- **机器精度验证通过**
- **0错误** 长时间运行

### 硬件支持
- **NVIDIA RTX 5060** (sm_120)
- **CUDA 12.x**
- **8GB VRAM**

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主求解器
python core/interior_point_lp_gpu_optimized.py

# 运行基准测试
python benchmarks/fpt_speedup_benchmark.py

# 验证正确性
python verification/verify_correctness_final.py
```

## GitHub上传说明

1. 创建GitHub仓库
2. 上传项目文件:
```bash
git init
git add .
git commit -m "Initial commit: GPU LP Solver v1.0"
git remote add origin https://github.com/yourusername/gpu-lp-solver.git
git push -u origin main
```

3. 添加标签:
```bash
git tag -a v1.0 -m "Version 1.0 - Production Ready"
git push origin v1.0
```

## 许可证

MIT License - 详见 LICENSE 文件

---
*GPU LP Solver Project - 2026-06-14*
