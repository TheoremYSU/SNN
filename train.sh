#!/bin/bash

# DVS-CIFAR10 训练脚本
# 使用改进版训练代码,支持TensorBoard和完整checkpoint管理

# ===================== 配置参数 =====================
# 数据路径(根据实际情况修改)
DATA_PATH="/home/liuwei/lzx/DVS-CIFAR10"

# 输出目录(使用当前目录下的runs文件夹,自动创建)
OUTPUT_DIR="./runs"

# 实验名称(留空则自动生成,格式: model_T{T}_lr{lr}_lamb{lamb}_{timestamp})
EXP_NAME=""

# 模型参数
MODEL="vgg_snn"          # 模型: vgg_snn / resnet19
T=4                      # 时间步数
LAMBDA=1e-4              # MSE损失权重

# 训练参数
EPOCHS=320
BATCH_SIZE=16
LR=0.1
WORKERS=4

# GPU设置
NPROCS=4                 # GPU数量(根据可用GPU数量设置)

# Checkpoint设置
SAVE_FREQ=10             # 每N个epoch保存一次checkpoint
RESUME=""                # 恢复训练路径,留空则从头训练

# ===================== 启动训练 =====================
echo "======================================"
echo "  TET训练 - 改进版"
echo "======================================"
echo "模型: $MODEL"
echo "时间步: T=$T"
echo "Lambda: $LAMBDA"
echo "学习率: $LR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "GPU数量: $NPROCS"
echo "输出目录: $OUTPUT_DIR"
echo "======================================"

# 创建输出目录(如果不存在)
mkdir -p "$OUTPUT_DIR"

# 构建训练命令
CMD="python -u main_training_distribute_improved.py \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --model $MODEL \
    --T $T \
    --lamb $LAMBDA \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --workers $WORKERS \
    --nprocs $NPROCS \
    --save-freq $SAVE_FREQ"

# 添加可选参数
if [ -n "$EXP_NAME" ]; then
    CMD="$CMD --exp-name $EXP_NAME"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# 执行训练
echo "执行命令:"
echo "$CMD"
echo ""

eval $CMD

# ===================== 训练完成 =====================
echo ""
echo "======================================"
echo "  训练完成!"
echo "======================================"
echo "查看训练日志:"
echo "  tensorboard --logdir=$OUTPUT_DIR"
echo ""
echo "实验目录位于: $OUTPUT_DIR"
echo "======================================"
