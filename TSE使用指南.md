# TSE (Temporal-Self-Erasing) 使用指南

## 方法简介

TSE (Temporal-Self-Erasing) 是AAAI 2025的一种新型SNN监督方法,通过动态擦除之前时间步的高激活区域,迫使网络在不同时间步探索新的判别性区域,从而解决SNN中反向传播梯度在时间步上相同导致特征表示相似的问题。

### 核心思想

**问题**: SNN在反向传播时,所有时间步接收相同的梯度,导致学到的特征表示高度相似。

**解决方案**: 
1. 第一个时间步 (t=0): 使用原始特征正常训练
2. 后续时间步 (t>0): 
   - 计算前面时间步的平均预测图
   - 找出高置信度的区域(高于固定阈值τ_f或动态阈值τ_d)
   - 擦除(抑制)这些区域
   - 强制网络关注新的区域

### 算法流程

对于每个时间步t:

1. **生成分类预测图** (Eq.7):
   ```
   P_t(c, i, j) = FC(F_t(i,j))  # 对每个空间位置(i,j)进行分类
   ```

2. **计算平均预测** (t>0时):
   ```
   P̄_{t-1} = Softmax(mean(P_0, P_1, ..., P_{t-1}))
   ```

3. **提取真实类别的概率图**:
   ```
   P_{t-1}_y(i,j) = P̄_{t-1}(y, i, j)  # y是真实标签
   ```

4. **计算动态阈值** (Eq.9):
   ```
   τ_d = mean(P_{t-1}_y) + κ × std(P_{t-1}_y)
   ```

5. **构建擦除掩码** (Eq.10):
   ```
   M_t(i,j) = {
       0,  if P_{t-1}_y(i,j) >= max(τ_f, τ_d)  # 擦除高置信区域
       1,  otherwise                            # 保留其他区域
   }
   ```

6. **特征调制** (Eq.11):
   ```
   F̃_t = F_t ⊙ M_t  # 元素级乘法
   ```

7. **计算损失** (Eq.12):
   ```
   L = L_CE(p_1, y) + Σ_{t=2}^T L_CE(p̃_t, y)
   ```
   其中: p̃_t = FC(GAP(F̃_t))

## 代码实现

### 1. 已实现的功能

#### functions.py
新增 `TSE_loss()` 函数:
```python
def TSE_loss(feature_maps, fc_layer, labels, criterion, tau_f=0.5, kappa=1.0):
    """
    参数:
        feature_maps: [B, T, C, H, W] - GAP之前的特征图
        fc_layer: 分类层(nn.Linear)
        labels: [B] - 真实标签
        criterion: 损失函数(CrossEntropyLoss)
        tau_f: 固定阈值(默认0.5)
        kappa: 动态阈值的标准差倍数(默认1.0)
    
    返回:
        total_loss: 所有时间步的总损失
    """
```

#### models/resnet_models.py
修改 `forward()` 方法支持返回GAP之前的特征:
```python
def forward(self, x, return_features=False):
    output, features_before_gap = self._forward_impl(x)
    if return_features:
        return output, features_before_gap
    else:
        return output
```

#### main_training_distribute_improved.py
1. 添加TSE相关参数:
```python
--tse / --no-tse  # 启用/禁用TSE (默认: False)
--tau-f           # 固定阈值 (默认: 0.5)
--kappa           # 动态阈值的κ参数 (默认: 1.0)
```

2. 修改train()函数支持TSE训练

### 2. 使用方法

#### 基本用法

使用TSE训练ResNet19:
```bash
python main_training_distribute_improved.py \
    --data-path /path/to/cifar10 \
    --dataset CIFAR10 \
    --arch resnet19 \
    --T 4 \
    --batch-size 128 \
    --epochs 320 \
    --lr 0.1 \
    --tse \
    --tau-f 0.5 \
    --kappa 1.0
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tse` | flag | False | 启用TSE训练 |
| `--no-tse` | flag | - | 禁用TSE训练(显式) |
| `--tau-f` | float | 0.5 | 固定阈值τ_f,控制擦除的最低置信度 |
| `--kappa` | float | 1.0 | 动态阈值的标准差倍数κ |

#### 与其他方法组合

**不要同时使用TET和TSE!** 它们是互斥的训练方法:

✅ **正确**:
```bash
# 只用TSE
python main_training_distribute_improved.py --tse --tau-f 0.5 ...

# 只用TET
python main_training_distribute_improved.py --tet --means 1.0 --lamb 0.05 ...

# 都不用(标准训练)
python main_training_distribute_improved.py --no-tet --no-tse ...
```

❌ **错误**:
```bash
# 不要同时启用TET和TSE!
python main_training_distribute_improved.py --tse --tet ...  # 错误!
```

### 3. 超参数调优建议

#### τ_f (固定阈值)
- **推荐值**: 0.5 (论文默认)
- **范围**: 0.3 ~ 0.7
- **作用**: 控制擦除的"硬性"下限
  - 较小值(0.3): 擦除更多区域,探索性更强,可能损失有用信息
  - 较大值(0.7): 只擦除极高置信区域,更保守

#### κ (标准差倍数)
- **推荐值**: 1.0 (论文默认)
- **范围**: 0.5 ~ 2.0
- **作用**: 控制动态阈值的自适应性
  - 较小值(0.5): 动态阈值更低,擦除更多
  - 较大值(2.0): 动态阈值更高,更谨慎擦除

#### 调优策略

1. **先用默认值** (τ_f=0.5, κ=1.0) 训练baseline
2. **数据集相关调整**:
   - **复杂数据集**(ImageNet): 减小τ_f到0.4,增加探索
   - **简单数据集**(CIFAR-10): 可保持默认或略微增加τ_f到0.6
3. **观察训练曲线**:
   - **过拟合**: 减小τ_f或κ,增加擦除
   - **欠拟合**: 增加τ_f或κ,减少擦除

### 4. 完整训练示例

#### CIFAR-10 (DVS)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main_training_distribute_improved.py \
    --data-path /data/dvscifar10 \
    --dataset DVS-CIFAR10 \
    --arch resnet19 \
    --T 10 \
    --batch-size 20 \
    --epochs 320 \
    --lr 0.1 \
    --tse \
    --tau-f 0.5 \
    --kappa 1.0 \
    --workers 4
```

#### CIFAR-100
```bash
python -m torch.distributed.launch \
    --nproc_per_node=7 \
    main_training_distribute_improved.py \
    --data-path /data/cifar100 \
    --dataset CIFAR-100 \
    --arch resnet19 \
    --T 4 \
    --batch-size 128 \
    --epochs 320 \
    --lr 0.1 \
    --tse \
    --tau-f 0.45 \
    --kappa 1.0 \
    --workers 8
```

### 5. 模型兼容性

#### 当前支持
- ✅ **ResNet19**: 完全支持,已修改返回features_before_gap
- ✅ **ResNet18/34/50**: 理论上支持(继承自相同基类)

#### 需要适配
- ⚠️ **VGG_SNN**: 需要类似修改,返回GAP之前的特征
  
如需VGG支持,需修改 `models/VGG_models.py`:
```python
def forward(self, x, return_features=False):
    # ... 卷积层 ...
    features_before_gap = x  # 保存GAP前的特征
    x = self.avgpool(x)
    x = torch.flatten(x, 2)
    x = self.classifier(x)
    
    if return_features:
        return x, features_before_gap
    else:
        return x
```

### 6. 预期性能

根据AAAI 2025论文:

| 数据集 | 基线(标准训练) | TSE | 提升 |
|--------|----------------|-----|------|
| CIFAR-10 | 93.x% | 94.x% | ~1% |
| CIFAR-100 | 70.x% | 72.x% | ~2% |
| DVS-CIFAR10 | 76.x% | 78.x% | ~2% |

*注: 具体数值取决于网络架构和训练超参数*

### 7. 调试和验证

#### 检查TSE是否生效

在train()函数中,TSE启用时会:
1. 调用 `model(images, return_features=True)`
2. 获取 `features_before_gap` 张量
3. 调用 `TSE_loss()` 而非标准loss或TET_loss

可以添加打印验证:
```python
if args.TSE:
    print(f"TSE enabled: tau_f={args.tau_f}, kappa={args.kappa}")
    output, features = model(images, return_features=True)
    print(f"Features shape: {features.shape}")  # 应该是 [B,T,C,H,W]
```

#### 常见问题

1. **AttributeError: 'ResNet' object has no attribute 'fc2'**
   - 原因: 使用的模型不是ResNet19或没有fc2层
   - 解决: 检查 `--arch` 参数,确保是resnet19

2. **RuntimeError: Expected 5D tensor, got 4D**
   - 原因: features_before_gap维度不对
   - 解决: 确保模型正确返回 [B,T,C,H,W] 格式的特征

3. **训练速度变慢**
   - 原因: TSE需要额外计算预测图和掩码
   - 预期: 比标准训练慢10-20%
   - 优化: 增加 `--workers` 或减小batch size

### 8. 论文引用

如果使用TSE方法,请引用:

```bibtex
@inproceedings{tse2025,
  title={Temporal-Self-Erasing Supervision for Spiking Neural Networks},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## 实现细节

### TSE_loss函数工作流程

```python
# 伪代码
for t in range(T):
    # 步骤1: 生成空间分类预测图
    P_t = classify_each_location(features[t])  # [B,num_classes,H,W]
    
    if t == 0:
        # 第一个时间步: 直接计算损失
        loss_t = CE_loss(GAP(P_t), labels)
    else:
        # 步骤2: 平均之前的预测
        P_avg = mean(P_0, ..., P_{t-1})
        P_avg = Softmax(P_avg)
        
        # 步骤3: 提取真实类别的概率图
        P_y = P_avg[labels]  # [B,H,W]
        
        # 步骤4: 计算动态阈值
        tau_d = mean(P_y) + kappa * std(P_y)
        
        # 步骤5: 构建掩码
        threshold = max(tau_f, tau_d)
        mask = (P_y < threshold).float()  # 0擦除,1保留
        
        # 步骤6: 调制特征
        F_erased = features[t] * mask
        
        # 步骤7: 计算损失
        P_erased = classify(GAP(F_erased))
        loss_t = CE_loss(P_erased, labels)
    
    total_loss += loss_t

return total_loss
```

### 与TET的区别

| 方面 | TET | TSE |
|------|-----|-----|
| **监督方式** | 所有时间步使用相同标签 | 每个时间步独立监督 |
| **特征调制** | 无 | 动态擦除高置信区域 |
| **损失函数** | L_CE + λ·L_MSE | Σ L_CE(p̃_t, y) |
| **时间依赖** | 无时间步间交互 | 后续步依赖前面步的预测 |
| **正则化** | MSE正则化膜电位 | 空间掩码正则化特征 |

## 总结

TSE是一种创新的SNN训练方法,通过空间-时间联合监督,解决了梯度相同导致的特征相似问题。集成到本代码库后,可以方便地与现有TET方法切换使用,为SNN训练提供了新的选择。

**建议的实验顺序**:
1. 先用默认参数 (τ_f=0.5, κ=1.0) 在小数据集(CIFAR-10)上测试
2. 与baseline(标准训练)和TET对比性能
3. 在目标数据集上调优超参数
4. 记录训练曲线和最终精度

祝训练顺利! 🚀
