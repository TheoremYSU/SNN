#!/bin/bash

# GPU问题紧急诊断脚本
# 用于快速排查 "ProcessGroupNCCL is only supported with GPUs, no GPUs found!" 错误

echo "========================================"
echo "  GPU环境诊断"
echo "========================================"
echo ""

# 1. 检查nvidia-smi
echo "1. 检查NVIDIA驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    echo "✓ NVIDIA驱动正常"
else
    echo "❌ nvidia-smi不可用!"
    echo "   请安装NVIDIA驱动"
    exit 1
fi
echo ""

# 2. 检查CUDA_VISIBLE_DEVICES
echo "2. 检查CUDA_VISIBLE_DEVICES环境变量..."
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "⚠️  CUDA_VISIBLE_DEVICES未设置"
    echo "   建议: export CUDA_VISIBLE_DEVICES=0,1,2,3"
    echo ""
    echo "你想现在设置吗? (输入GPU编号,例如: 0,1,2,3)"
    read -p "GPU编号 (直接回车跳过): " gpu_ids
    if [ -n "$gpu_ids" ]; then
        export CUDA_VISIBLE_DEVICES=$gpu_ids
        echo "✓ 已设置: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
else
    echo "✓ CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
fi
echo ""

# 3. 检查PyTorch CUDA支持
echo "3. 检查PyTorch CUDA支持..."
python -c "
import torch
import sys

print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA是否可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print('✓ PyTorch CUDA支持正常')
else:
    print('❌ PyTorch CUDA不可用!')
    print('可能原因:')
    print('  1. PyTorch是CPU版本')
    print('  2. CUDA版本不兼容')
    print('  3. CUDA_VISIBLE_DEVICES设置错误')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "解决方案:"
    echo "  1. 重新安装PyTorch CUDA版本:"
    echo "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo "  2. 或安装CUDA 12.x版本:"
    echo "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi
echo ""

# 4. 检查NCCL
echo "4. 检查NCCL后端..."
python -c "
import torch.distributed as dist
print('✓ NCCL后端可导入')
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ NCCL后端不可用!"
    echo "   请重新安装PyTorch"
    exit 1
fi
echo ""

# 5. 测试GPU multiprocessing
echo "5. 测试GPU在multiprocessing中的可用性..."
echo "   运行: python test_gpu_multiprocessing.py"
python test_gpu_multiprocessing.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  ✓ 所有检查通过!"
    echo "========================================"
    echo ""
    echo "你的环境配置正确,可以运行训练了:"
    echo "  bash train.sh"
    echo ""
else
    echo ""
    echo "========================================"
    echo "  ❌ 发现问题"
    echo "========================================"
    echo ""
    echo "请根据上述错误信息修复问题"
    echo ""
fi
