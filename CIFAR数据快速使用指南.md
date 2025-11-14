# CIFAR数据快速使用指南

## 🚀 快速开始

### 步骤1: 检测你的数据格式

```bash
cd TET_improve/temporal_efficient_training
python check_data_format.py /path/to/your/cifar100
```

**可能的输出:**

#### 情况A: 标准二进制格式
```
✅ 检测到格式1: 标准二进制格式
   数据集: CIFAR-100
   训练文件: train (二进制)
   测试文件: test (二进制)
```
**→ 你的格式已支持!** 继续步骤2

#### 情况B: 图片文件夹格式
```
✅ 检测到格式2: 图片文件夹格式
   训练集: 100 个类别文件夹
   训练图片数: 50000
```
**→ 你的格式已支持!** 继续步骤2

---

### 步骤2: 测试数据加载

```bash
python test_dataloader.py /path/to/your/cifar100
```

**预期输出:**
```
================================================================================
测试数据加载
================================================================================
数据路径: /path/to/your/cifar100
数据集: CIFAR-100
加载方式: auto
================================================================================

使用自动检测格式...
检测到标准CIFAR-100二进制格式: /path/to/your/cifar100

✅ 数据集加载成功!
================================================================================
训练集大小: 50000
测试集大小: 10000

测试加载单个样本...
  数据形状: torch.Size([3, 32, 32])
  数据范围: [-2.429, 2.754]
  标签: 19

✅ 所有测试通过!
```

**如果测试失败,会显示详细错误和建议**

---

### 步骤3: 在训练代码中使用

#### 方式1: 使用自动检测(推荐)

```python
from data_loaders import auto_build_cifar

# 自动检测并加载
train_dataset, val_dataset = auto_build_cifar(
    data_path='/path/to/your/cifar100',
    use_cifar10=False  # CIFAR-100用False, CIFAR-10用True
)
```

#### 方式2: 指定格式类型

```python
from data_loaders import build_cifar, build_cifar_from_images

# 如果是标准二进制格式
train_dataset, val_dataset = build_cifar(
    use_cifar10=False,
    data_path='/path/to/your/cifar100'
)

# 如果是图片文件夹格式
train_dataset, val_dataset = build_cifar_from_images(
    data_path='/path/to/your/cifar100',
    use_cifar10=False
)
```

---

## 📋 你的两种格式对应

### 格式1: file.txt, meta, test, train (二进制文件)

**检测命令:**
```bash
python check_data_format.py /path/to/format1
```

**使用代码:**
```python
from data_loaders import build_cifar

train_dataset, val_dataset = build_cifar(
    use_cifar10=False,
    data_path='/path/to/format1'
)
```

**或使用自动检测:**
```python
from data_loaders import auto_build_cifar

train_dataset, val_dataset = auto_build_cifar(
    data_path='/path/to/format1',
    use_cifar10=False
)
```

---

### 格式2: train/, test/ 文件夹包含图片

**目录结构要求:**
```
your_data_path/
├── train/
│   ├── class_0/
│   │   ├── img1.png
│   │   └── img2.png
│   ├── class_1/
│   └── ... (class_0 到 class_99)
└── test/
    ├── class_0/
    └── ... (class_0 到 class_99)
```

**检测命令:**
```bash
python check_data_format.py /path/to/format2
```

**使用代码:**
```python
from data_loaders import build_cifar_from_images

train_dataset, val_dataset = build_cifar_from_images(
    data_path='/path/to/format2',
    use_cifar10=False
)
```

**或使用自动检测:**
```python
from data_loaders import auto_build_cifar

train_dataset, val_dataset = auto_build_cifar(
    data_path='/path/to/format2',
    use_cifar10=False
)
```

---

## ⚠️ 常见问题

### Q1: 图片直接在train/test文件夹中,没有类别子文件夹?

**问题:** 你的结构是这样的:
```
data/
├── train/
│   ├── img1.png
│   ├── img2.png
│   └── ...
└── test/
    └── ...
```

**解决方案A: 手动整理(推荐)**
```bash
# 创建类别文件夹
mkdir -p train/class_0 train/class_1 ... train/class_99
mkdir -p test/class_0 test/class_1 ... test/class_99

# 根据标签文件移动图片到对应文件夹
```

**解决方案B: 使用转换工具**

首先创建标签文件 `labels.txt`:
```
img1.png 0
img2.png 23
img3.png 45
...
```

然后运行:
```bash
python check_data_format.py /path/to/data \
    --convert \
    --label-file labels.txt
```

---

### Q2: 类别数不是100怎么办?

**检测实际类别数:**
```bash
python test_dataloader.py /path/to/your/data
```

输出会显示:
```
检测到 XX 个不同类别
```

**修改模型类别数:**

编辑 `main_training_distribute_improved.py`:
```python
# 在创建模型的地方
if args.model == 'VGGSNN':
    model = VGGSNN(num_classes=XX)  # 改为实际类别数
elif args.model == 'resnet19':
    model = resnet19(num_classes=XX)
```

---

### Q3: 测试数据加载报错?

**运行完整测试:**
```bash
python test_dataloader.py /path/to/your/data --format auto
```

**查看详细错误信息,按提示操作:**
1. 检查路径是否正确
2. 检查文件权限
3. 查看格式检测结果: `python check_data_format.py /path/to/data`
4. 阅读完整文档: `CIFAR数据格式支持说明.md`

---

## 📝 完整测试流程示例

### 测试格式1 (二进制)

```bash
# 1. 检测格式
python check_data_format.py /data/cifar100_binary

# 输出: ✅ 检测到格式1: 标准二进制格式

# 2. 测试加载
python test_dataloader.py /data/cifar100_binary

# 输出: ✅ 所有测试通过!

# 3. 在代码中使用
# 见上面"步骤3"
```

### 测试格式2 (图片)

```bash
# 1. 检测格式
python check_data_format.py /data/cifar100_images

# 输出: ✅ 检测到格式2: 图片文件夹格式

# 2. 测试加载
python test_dataloader.py /data/cifar100_images --format images

# 输出: ✅ 所有测试通过!

# 3. 在代码中使用
# 见上面"步骤3"
```

---

## 🎯 命令速查表

| 操作 | 命令 |
|------|------|
| 检测数据格式 | `python check_data_format.py /path/to/data` |
| 测试加载(自动) | `python test_dataloader.py /path/to/data` |
| 测试加载(二进制) | `python test_dataloader.py /path/to/data --format binary` |
| 测试加载(图片) | `python test_dataloader.py /path/to/data --format images` |
| 测试CIFAR-10 | `python test_dataloader.py /path/to/data --cifar10` |
| 转换平铺图片 | `python check_data_format.py /path --convert --label-file labels.txt` |

---

## 📚 更多信息

- **完整文档**: `CIFAR数据格式支持说明.md`
- **代码实现**: `data_loaders.py`
- **检测工具**: `check_data_format.py`
- **测试脚本**: `test_dataloader.py`

如有问题,请参考完整文档或运行检测工具查看详细信息!
