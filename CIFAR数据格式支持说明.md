# CIFAR数据格式支持说明

## 支持的数据格式

代码现在支持**3种**CIFAR数据格式:

### 格式1: 标准二进制格式 ✅

**CIFAR-10结构:**
```
data_path/
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
└── test_batch
```

**CIFAR-100结构:**
```
data_path/
├── meta
├── train        # 二进制文件(非文件夹)
├── test         # 二进制文件(非文件夹)
└── file.txt     # 可选
```

**使用方法:**
```python
from data_loaders import build_cifar

# CIFAR-10
train_dataset, val_dataset = build_cifar(
    use_cifar10=True,
    data_path='/path/to/cifar10'
)

# CIFAR-100
train_dataset, val_dataset = build_cifar(
    use_cifar10=False,
    data_path='/path/to/cifar100'
)
```

---

### 格式2: 图片文件夹格式 ✅ (新增)

**结构要求:**
```
data_path/
├── train/
│   ├── class0/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   ├── class1/
│   │   └── ...
│   └── class99/
│       └── ...
└── test/
    ├── class0/
    ├── class1/
    └── class99/
```

**使用方法:**
```python
from data_loaders import build_cifar_from_images

# CIFAR-100图片格式
train_dataset, val_dataset = build_cifar_from_images(
    data_path='/path/to/cifar100_images',
    use_cifar10=False  # CIFAR-100使用False
)
```

---

### 格式3: 自动检测 ✅ (推荐)

**使用方法:**
```python
from data_loaders import auto_build_cifar

# 自动检测格式并加载
train_dataset, val_dataset = auto_build_cifar(
    data_path='/path/to/cifar100',
    use_cifar10=False,
    download=False
)
```

代码会自动:
1. 检查是否为图片文件夹格式
2. 检查是否为标准二进制格式
3. 选择合适的加载方式

---

## 数据格式检测工具

### 1. 检测数据格式

```bash
cd TET_improve/temporal_efficient_training
python check_data_format.py /path/to/your/cifar100
```

**示例输出:**
```
================================================================================
检查数据路径: /path/to/cifar100
================================================================================

根目录内容: ['train', 'test', 'meta', 'file.txt']

✅ 检测到格式1: 标准二进制格式
   数据集: CIFAR-100
   训练文件: train (二进制)
   测试文件: test (二进制)
   元数据: meta

使用方法:
   train_dataset, val_dataset = build_cifar(
       use_cifar10=False,
       data_path='/path/to/cifar100'
   )
```

### 2. 转换图片为类别文件夹

如果你的图片直接在train/test文件夹中,没有类别子文件夹:

```bash
python check_data_format.py /path/to/cifar100_flat \
    --convert \
    --label-file /path/to/labels.txt
```

标签文件格式 (labels.txt):
```
img1.png 0
img2.png 1
img3.png 0
...
```

---

## 在训练脚本中使用

### 修改 main_training_distribute_improved.py

目前DVS-CIFAR10硬编码,需要修改支持CIFAR-100:

```python
# 添加参数
parser.add_argument('--dataset',
                    default='dvscifar10',
                    type=str,
                    choices=['dvscifar10', 'cifar10', 'cifar100'],
                    help='dataset name')

# 在main_worker中修改数据加载
if args.dataset == 'dvscifar10':
    train_dataset, val_dataset = data_loaders.build_dvscifar(args.data_path)
elif args.dataset == 'cifar10':
    train_dataset, val_dataset = data_loaders.auto_build_cifar(
        args.data_path, use_cifar10=True
    )
elif args.dataset == 'cifar100':
    train_dataset, val_dataset = data_loaders.auto_build_cifar(
        args.data_path, use_cifar10=False
    )
```

### 修改 train.sh

```bash
# 添加数据集参数
DATASET="cifar100"           # dvscifar10 / cifar10 / cifar100
DATA_PATH="/path/to/cifar100"

# 在训练命令中添加
CMD="python -u main_training_distribute_improved.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    ..."
```

---

## 你的数据格式对应

### 格式1: file.txt, meta, test, train (二进制)
- ✅ **已支持**
- 使用: `build_cifar(use_cifar10=False, data_path='...')`
- 或: `auto_build_cifar(data_path='...', use_cifar10=False)`

### 格式2: train/, test/ 文件夹包含图片
- ✅ **已支持**
- 使用: `build_cifar_from_images(data_path='...', use_cifar10=False)`
- 或: `auto_build_cifar(data_path='...', use_cifar10=False)` (自动检测)

---

## 完整使用示例

### 示例1: 检测并使用标准二进制格式

```python
# 1. 检测格式
python check_data_format.py /data/cifar100

# 输出: ✅ 检测到格式1: 标准二进制格式

# 2. 在代码中使用
from data_loaders import build_cifar
train_dataset, val_dataset = build_cifar(
    use_cifar10=False,
    data_path='/data/cifar100'
)
```

### 示例2: 使用图片文件夹格式

```python
# 1. 检测格式
python check_data_format.py /data/cifar100_images

# 输出: ✅ 检测到格式2: 图片文件夹格式

# 2. 在代码中使用
from data_loaders import build_cifar_from_images
train_dataset, val_dataset = build_cifar_from_images(
    data_path='/data/cifar100_images',
    use_cifar10=False
)
```

### 示例3: 自动检测(推荐)

```python
# 适用于任何格式,自动检测
from data_loaders import auto_build_cifar
train_dataset, val_dataset = auto_build_cifar(
    data_path='/data/cifar100',  # 任意格式
    use_cifar10=False
)
```

---

## 常见问题

### Q1: 我的数据是CIFAR-100标准格式,但代码报错?

**检查:**
```bash
python check_data_format.py /path/to/your/data
```

**可能原因:**
- 路径错误
- 文件名不匹配(train文件是否为二进制文件而非文件夹)

### Q2: 图片格式但没有类别子文件夹怎么办?

**方案1:** 手动创建类别文件夹
```bash
# 在train/test下创建class_0, class_1, ... class_99
# 然后将图片移动到对应文件夹
```

**方案2:** 使用转换工具
```bash
python check_data_format.py /path/to/data \
    --convert \
    --label-file labels.txt
```

### Q3: 类别数不是100怎么办?

**修改模型:**
```python
# 在main_training_distribute_improved.py中
if args.model == 'VGGSNN':
    model = VGGSNN(num_classes=100)  # 修改类别数
elif args.model == 'resnet19':
    model = resnet19(num_classes=100)
```

---

## 测试数据加载

```python
# test_dataloader.py
from data_loaders import auto_build_cifar
import torch

# 加载数据
train_dataset, val_dataset = auto_build_cifar(
    data_path='/path/to/cifar100',
    use_cifar10=False
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(val_dataset)}")

# 测试加载一个batch
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True
)

images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")
print(f"Labels: {labels}")
```

运行测试:
```bash
python test_dataloader.py
```

预期输出:
```
检测到图片文件夹格式 (带类别子文件夹): /path/to/cifar100
加载图片数据集:
  训练集: 50000 张图片
  测试集: 10000 张图片
  类别数: 100
训练集大小: 50000
测试集大小: 10000
Batch shape: torch.Size([4, 3, 32, 32])
Labels: tensor([23, 45, 67, 89])
```
