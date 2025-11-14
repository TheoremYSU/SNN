#!/bin/bash

# 快速测试脚本 - 用于验证改进版训练代码

# ===================== 配置参数 =====================
DATA_PATH="/home/liuwei/lzx/DVS-CIFAR10"  # 根据实际情况修改
OUTPUT_DIR="./test_runs"                   # 测试输出目录

# 使用小的参数快速测试
MODEL="vgg_snn"
T=2                    # 减少时间步
EPOCHS=2               # 只训练2个epoch测试
BATCH_SIZE=8           # 小batch size
LR=0.1
WORKERS=2
NPROCS=2               # 只用2个GPU测试
SAVE_FREQ=1            # 每个epoch都保存

# ===================== 启动测试 =====================
echo "======================================"
echo "  快速测试 - 改进版训练代码"
echo "======================================"

mkdir -p "$OUTPUT_DIR"

python -u main_training_distribute_improved.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --T $T \
    --lamb 1e-4 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --workers $WORKERS \
    --nprocs $NPROCS \
    --save-freq $SAVE_FREQ

echo ""
echo "======================================"
echo "  测试完成!"
echo "======================================"
echo "检查输出目录: $OUTPUT_DIR"
echo "查看TensorBoard: tensorboard --logdir=$OUTPUT_DIR"
