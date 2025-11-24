#!/bin/bash

# TSE训练示例脚本
# 使用说明: bash train_tse_example.sh

# 配置GPU数量
NUM_GPUS=4

# 数据集路径(请修改为你的实际路径)
DATA_PATH="/path/to/your/dataset"

# 数据集选择: CIFAR-10, CIFAR-100, DVS-CIFAR10
DATASET="CIFAR-10"

# 网络架构
ARCH="resnet19"

# TSE超参数
TSE_ENABLED="--tse"          # 启用TSE
TAU_F=0.5                    # 固定阈值
KAPPA=1.0                    # 动态阈值系数

# 训练超参数
BATCH_SIZE=128
EPOCHS=320
LR=0.1
TIME_STEPS=4

# 其他参数
WORKERS=8
PRINT_FREQ=50

echo "=========================================="
echo "TSE训练脚本"
echo "=========================================="
echo "数据集: $DATASET"
echo "架构: $ARCH"
echo "GPU数量: $NUM_GPUS"
echo "TSE参数: tau_f=$TAU_F, kappa=$KAPPA"
echo "时间步: $TIME_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo "=========================================="

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    main_training_distribute_improved.py \
    --data-path $DATA_PATH \
    --dataset $DATASET \
    --arch $ARCH \
    --T $TIME_STEPS \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    $TSE_ENABLED \
    --tau-f $TAU_F \
    --kappa $KAPPA \
    --workers $WORKERS \
    --print-freq $PRINT_FREQ

echo ""
echo "训练完成!"
