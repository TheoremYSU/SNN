# ProcessGroupNCCL错误排查指南

## 错误信息

```
ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!
```

## 快速诊断 (3步解决)

### 第1步: 运行诊断脚本

```bash
cd temporal_efficient_training
bash diagnose_gpu.sh
```

这个脚本会自动检查所有问题并给出建议。

### 第2步: 检查PyTorch CUDA版本

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

**如果显示 `CUDA available: False`:**

你的PyTorch是CPU版本,需要重新安装:

```bash
# 检查你的CUDA版本
nvidia-smi  # 查看Driver Version和CUDA Version

# CUDA 11.x
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.x
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 第3步: 设置CUDA_VISIBLE_DEVICES

```bash
# 在train.sh开头添加 (第2行)
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据你的GPU数量调整

# 或者在命令行设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash train.sh
```

## 常见原因和解决方案

### 原因1: PyTorch是CPU版本

**症状:**
- `torch.cuda.is_available()` 返回 `False`
- `torch.version.cuda` 返回 `None`

**解决:**

```bash
# 卸载CPU版本
pip uninstall torch torchvision

# 安装CUDA版本 (根据你的CUDA版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 原因2: CUDA版本不匹配

**症状:**
- `nvidia-smi` 显示 CUDA 12.x
- 但 `torch.version.cuda` 显示 11.x 或 `None`

**解决:**

```bash
# 检查系统CUDA版本
nvidia-smi | grep "CUDA Version"

# 安装匹配的PyTorch版本
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 原因3: CUDA_VISIBLE_DEVICES未设置

**症状:**
- GPU检测工具显示GPU可用
- 但训练时报错 "no GPUs found"

**解决:**

**方法1: 在train.sh中设置**

编辑 `train.sh`,在第2行添加:

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 添加这行
```

**方法2: 命令行设置**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash train.sh
```

### 原因4: NCCL库问题

**症状:**
- CUDA可用
- 但NCCL初始化失败

**解决:**

```bash
# 重新安装PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 检查NCCL
python -c "import torch.distributed as dist; print('NCCL OK')"
```

### 原因5: multiprocessing环境变量丢失

**症状:**
- 主进程能检测到GPU
- 子进程检测不到GPU

**解决:**

在 `train.sh` **最开始**设置环境变量:

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 必须在最开始
export OMP_NUM_THREADS=1              # 可选,提高性能

# 然后是其他配置...
```

## 完整诊断流程

### 1. 检查硬件

```bash
nvidia-smi
```

应该看到你的GPU列表。

### 2. 检查PyTorch

```bash
python -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
print('CUDA版本:', torch.version.cuda)
print('GPU数量:', torch.cuda.device_count())
"
```

期望输出:
```
PyTorch版本: 2.x.x
CUDA可用: True
CUDA版本: 11.8 (或12.1)
GPU数量: 4 (或更多)
```

### 3. 测试multiprocessing

```bash
python test_gpu_multiprocessing.py
```

应该显示:
```
✓ 所有测试通过! GPU在multiprocessing中正常工作
```

### 4. 运行训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash train.sh
```

## 推荐配置

### train.sh 推荐配置

```bash
#!/bin/bash

# ============ GPU设置 (必须放在最前面) ============
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据可用GPU调整
export OMP_NUM_THREADS=1

# 检查GPU
echo "检查GPU环境..."
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"

# ============ 训练参数 ============
DATASET="cifar100"
NUM_CLASSES=100
DATA_PATH="/your/data/path"
MODEL="VGGSNN"
T=4
NPROCS=4  # 必须 <= CUDA_VISIBLE_DEVICES中的GPU数
BATCH_SIZE=16
LR=0.1
EPOCHS=320

# ============ 启动训练 ============
python -u main_training_distribute_improved.py \
    --dataset $DATASET \
    --num-classes $NUM_CLASSES \
    --data-path $DATA_PATH \
    --model $MODEL \
    --T $T \
    --nprocs $NPROCS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS
```

## 最后检查清单

运行训练前,确保:

- [ ] `nvidia-smi` 能看到GPU
- [ ] `torch.cuda.is_available()` 返回 `True`
- [ ] `torch.version.cuda` 不是 `None`
- [ ] `CUDA_VISIBLE_DEVICES` 已设置
- [ ] `test_gpu_multiprocessing.py` 通过
- [ ] `train.sh` 中的 `NPROCS` <= GPU数量

全部通过后,运行:

```bash
bash train.sh
```

## 仍然有问题?

### 详细诊断输出

运行改进版训练脚本,会看到详细的GPU检测信息:

```bash
python main_training_distribute_improved.py --help
```

训练时会自动显示:

```
================================================================================
GPU环境检查
================================================================================
CUDA是否可用: True
CUDA版本: 11.8
可用GPU数量: 4
  GPU 0: NVIDIA GeForce RTX 2080 Ti
  GPU 1: NVIDIA GeForce RTX 2080 Ti
  GPU 2: NVIDIA GeForce RTX 2080 Ti
  GPU 3: NVIDIA GeForce RTX 2080 Ti
CUDA_VISIBLE_DEVICES: 0,1,2,3
================================================================================
```

如果子进程失败,会显示:

```
[Rank 2] ❌ 错误: 子进程中CUDA不可用!
[Rank 2] CUDA_VISIBLE_DEVICES = 未设置
[Rank 2] 这可能是因为:
  1. PyTorch没有正确编译CUDA支持
  2. CUDA驱动版本与PyTorch不兼容
  3. 环境变量在multiprocessing spawn中丢失
```

### 联系信息

如果以上方法都无法解决,请提供以下信息:

```bash
# 系统信息
uname -a
nvidia-smi
python --version

# PyTorch信息
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDNN:', torch.backends.cudnn.version())"

# 环境变量
echo $CUDA_VISIBLE_DEVICES
echo $LD_LIBRARY_PATH

# 测试结果
python test_gpu_multiprocessing.py
```

---

**最后更新:** 2025年11月15日
**适用版本:** PyTorch 1.x/2.x with CUDA support
