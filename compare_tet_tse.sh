#!/bin/bash

# TET vs TSE 对比实验脚本
# 目的: 在相同设置下对比TET和TSE的性能

# 配置
NUM_GPUS=4
DATA_PATH="/path/to/cifar100"  # 修改为实际路径
DATASET="CIFAR-100"
ARCH="resnet19"
BATCH_SIZE=128
EPOCHS=320
LR=0.1
TIME_STEPS=4
WORKERS=8

echo "=========================================="
echo "TET vs TSE 对比实验"
echo "=========================================="
echo "数据集: $DATASET"
echo "架构: $ARCH (T=$TIME_STEPS)"
echo "训练设置: Epochs=$EPOCHS, LR=$LR, BS=$BATCH_SIZE"
echo ""

# 实验1: Baseline (标准训练,无TET无TSE)
echo "实验1: Baseline (标准训练)"
echo "=========================================="
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
    --no-tet \
    --no-tse \
    --workers $WORKERS \
    --output-dir ./results/baseline

echo ""
echo "实验1完成!"
echo ""

# 实验2: TET训练
echo "实验2: TET训练"
echo "=========================================="
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
    --tet \
    --means 1.0 \
    --lamb 0.05 \
    --workers $WORKERS \
    --output-dir ./results/tet

echo ""
echo "实验2完成!"
echo ""

# 实验3: TSE训练 (默认参数)
echo "实验3: TSE训练 (tau_f=0.5, kappa=1.0)"
echo "=========================================="
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
    --tse \
    --tau-f 0.5 \
    --kappa 1.0 \
    --workers $WORKERS \
    --output-dir ./results/tse_default

echo ""
echo "实验3完成!"
echo ""

# 实验4: TSE训练 (更激进的擦除)
echo "实验4: TSE训练 (tau_f=0.3, kappa=0.5 - 更激进)"
echo "=========================================="
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
    --tse \
    --tau-f 0.3 \
    --kappa 0.5 \
    --workers $WORKERS \
    --output-dir ./results/tse_aggressive

echo ""
echo "实验4完成!"
echo ""

# 实验5: TSE训练 (更保守的擦除)
echo "实验5: TSE训练 (tau_f=0.7, kappa=2.0 - 更保守)"
echo "=========================================="
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
    --tse \
    --tau-f 0.7 \
    --kappa 2.0 \
    --workers $WORKERS \
    --output-dir ./results/tse_conservative

echo ""
echo "实验5完成!"
echo ""

# 生成总结
echo "=========================================="
echo "所有实验完成!"
echo "=========================================="
echo ""
echo "实验结果目录:"
echo "  1. Baseline:        ./results/baseline"
echo "  2. TET:             ./results/tet"
echo "  3. TSE (默认):      ./results/tse_default"
echo "  4. TSE (激进):      ./results/tse_aggressive"
echo "  5. TSE (保守):      ./results/tse_conservative"
echo ""
echo "请查看各目录下的训练日志和checkpoint文件"
echo ""

# 如果安装了tensorboard,可以可视化对比
if command -v tensorboard &> /dev/null; then
    echo "使用Tensorboard查看结果:"
    echo "  tensorboard --logdir ./results"
else
    echo "提示: 安装tensorboard可以可视化对比 (pip install tensorboard)"
fi
