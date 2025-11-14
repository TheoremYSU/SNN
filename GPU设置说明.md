# GPU设置说明

## 问题描述

如果出现以下错误:
```
ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!
```

这表示PyTorch无法检测到可用的GPU。这通常是由于 `CUDA_VISIBLE_DEVICES` 环境变量设置不当导致的。

## 原因分析

### 常见原因

1. **CUDA_VISIBLE_DEVICES未设置或设置错误**
   - 环境变量指定的GPU编号不存在
   - 例如: 系统只有GPU 0-6,但设置了 `CUDA_VISIBLE_DEVICES=2,3,6,7`

2. **硬编码GPU设置冲突**
   - 代码中写死了GPU编号,与实际环境不符
   - 已在改进版代码中移除硬编码设置

3. **多进程启动顺序问题**
   - 在参数解析前就设置了环境变量
   - 导致进程spawn时GPU检测失败

## 解决方案

### 方案1: 在shell脚本中设置 (推荐)

**在 `train.sh` 中添加:**

```bash
# 在脚本开头添加(第2-3行)
export CUDA_VISIBLE_DEVICES=2,3,6,7  # 根据你的GPU实际情况修改
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
```

**完整示例:**

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,6,7  # 使用GPU 2,3,6,7
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# TET训练脚本 - 改进版
DATASET="cifar100"
NUM_CLASSES=100
DATA_PATH="/your/data/path"
# ... 其他配置
```

### 方案2: 在命令行中设置

**临时设置 (仅当前会话):**

```bash
export CUDA_VISIBLE_DEVICES=2,3,6,7
bash train.sh
```

**一行命令:**

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 bash train.sh
```

### 方案3: 系统级设置 (不推荐)

**在 `~/.bashrc` 中添加:**

```bash
export CUDA_VISIBLE_DEVICES=2,3,6,7
```

**生效:**

```bash
source ~/.bashrc
```

⚠️ 注意: 这会影响所有使用GPU的程序

## 快速诊断工具

### 使用诊断脚本 (推荐)

我们提供了一个诊断脚本来快速检测GPU问题:

```bash
cd temporal_efficient_training
python test_gpu_multiprocessing.py
```

**这个脚本会检查:**
- ✓ 主进程中CUDA是否可用
- ✓ 子进程中CUDA是否可用
- ✓ NCCL后端是否能正常初始化
- ✓ 所有GPU是否在multiprocessing中可见

**如果全部通过,会显示:**
```
✓ 所有测试通过! GPU在multiprocessing中正常工作
```

**如果有问题,会显示详细的错误信息和解决建议。**

## 如何检查你的GPU

### 1. 查看可用GPU

```bash
nvidia-smi
```

输出示例:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05    Driver Version: 580.95.05    CUDA Version: 13.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce ...  Off  | 00000000:02:00.0 Off |                  N/A |
|   1  NVIDIA GeForce ...  Off  | 00000000:03:00.0 Off |                  N/A |
|   2  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
|   ...
```

从左侧 **GPU列** 可以看到可用GPU编号 (0, 1, 2, ...)

### 2. 选择要使用的GPU

**示例场景:**

- 系统有7个GPU (编号 0-6)
- 你想使用其中4个

**选择方式:**

```bash
# 使用前4个GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 或使用GPU 2,3,6,7 (如果这些GPU空闲)
export CUDA_VISIBLE_DEVICES=2,3,6,7

# 或只使用1个GPU
export CUDA_VISIBLE_DEVICES=0
```

⚠️ **重要提示:** 
- 设置的GPU数量必须 >= `train.sh` 中的 `NPROCS` 参数
- 例如: `NPROCS=4` 需要至少4个GPU

## 改进版代码的GPU检测功能

### 自动诊断

改进版代码添加了 `check_cuda_availability()` 函数,会自动检查:

1. ✓ torch.cuda.is_available() - CUDA是否可用
2. ✓ CUDA版本信息
3. ✓ 可用GPU数量
4. ✓ 每个GPU的名称
5. ✓ CUDA_VISIBLE_DEVICES环境变量值

### 运行时输出

**正常情况:**

```
=== CUDA Availability Check ===
✓ CUDA is available
CUDA Version: 11.3
Available GPUs: 4
GPU 0: NVIDIA GeForce RTX 2080 Ti
GPU 1: NVIDIA GeForce RTX 2080 Ti
GPU 2: NVIDIA GeForce RTX 2080 Ti
GPU 3: NVIDIA GeForce RTX 2080 Ti
CUDA_VISIBLE_DEVICES: 2,3,6,7
================================
使用 4 个GPU进行分布式训练
```

**异常情况:**

```
=== CUDA Availability Check ===
❌ CUDA is not available!
Possible reasons:
  1. PyTorch not compiled with CUDA support
  2. No NVIDIA GPUs detected
  3. CUDA drivers not installed
  4. CUDA_VISIBLE_DEVICES set incorrectly
================================
❌ 错误: 没有检测到可用的GPU!
解决方案:
  1. 检查CUDA_VISIBLE_DEVICES环境变量
  2. 运行: export CUDA_VISIBLE_DEVICES=2,3,6,7
  3. 或在脚本中设置: export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 完整训练流程

### 步骤1: 检查GPU可用性

```bash
nvidia-smi
```

### 步骤2: 设置CUDA_VISIBLE_DEVICES

```bash
# 根据nvidia-smi输出,选择要使用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 步骤3: 修改train.sh配置

```bash
vim train.sh
```

修改以下参数:

```bash
DATASET="cifar100"           # 数据集类型
NUM_CLASSES=100              # 类别数
DATA_PATH="/your/path"       # 数据路径
NPROCS=4                     # GPU数量(必须 <= CUDA_VISIBLE_DEVICES中的GPU数)
```

### 步骤4: 运行训练

```bash
bash train.sh
```

或一行命令:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh
```

## 常见问题

### Q1: 我的GPU编号不连续,可以使用吗?

**A:** 可以! PyTorch会自动重映射。

例如:
```bash
export CUDA_VISIBLE_DEVICES=2,5,7  # 使用不连续的GPU编号
```

PyTorch内部会将它们映射为: 0,1,2

### Q2: 如何使用所有可用GPU?

**A:** 不设置 `CUDA_VISIBLE_DEVICES`,或设置为所有GPU:

```bash
# 方法1: 不设置(使用所有GPU)
bash train.sh

# 方法2: 明确指定所有GPU(假设有7个)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
bash train.sh
```

### Q3: 训练时GPU内存不足怎么办?

**A:** 减少batch size或减少GPU数量:

```bash
# 在train.sh中修改
BATCH_SIZE=8     # 减小batch size
NPROCS=2         # 减少GPU数量
```

### Q4: 如何验证GPU设置是否生效?

**A:** 运行训练脚本,查看诊断输出:

```bash
bash train.sh | head -20
```

应该看到:

```
=== CUDA Availability Check ===
✓ CUDA is available
Available GPUs: 4
...
使用 4 个GPU进行分布式训练
```

## 推荐配置

### 配置1: 使用4个GPU训练CIFAR-100

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET="cifar100"
NUM_CLASSES=100
DATA_PATH="/path/to/cifar100"
MODEL="VGGSNN"
T=4
NPROCS=4        # 使用4个GPU
BATCH_SIZE=16
```

### 配置2: 使用2个GPU训练DVS-CIFAR10

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

DATASET="dvscifar10"
NUM_CLASSES=10
DATA_PATH="/path/to/dvscifar10"
MODEL="VGGSNN"
T=4
NPROCS=2        # 使用2个GPU
BATCH_SIZE=16
```

### 配置3: 单GPU训练

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

DATASET="cifar10"
NUM_CLASSES=10
DATA_PATH="/path/to/cifar10"
MODEL="VGGSNN"
T=4
NPROCS=1        # 使用1个GPU
BATCH_SIZE=32   # 单GPU可以用更大的batch size
```

## 总结

1. **优先使用方案1**: 在 `train.sh` 开头设置 `CUDA_VISIBLE_DEVICES`
2. **确保GPU编号有效**: 使用 `nvidia-smi` 查看可用GPU
3. **NPROCS必须匹配**: NPROCS <= 设置的GPU数量
4. **查看诊断输出**: 改进版代码会自动检查并提示问题
5. **遇到问题**: 查看终端输出的错误提示和解决建议

---

**最后更新:** 根据用户实际GPU配置 (7个RTX 2080 Ti) 修复了硬编码GPU选择问题
