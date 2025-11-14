"""
诊断类别数不匹配问题

用于检测数据集实际类别数和模型类别数是否匹配
"""
import sys
import torch
from data_loaders import auto_build_cifar
from models.VGG_models import VGGSNN
from models.resnet_models import resnet19


def diagnose_class_mismatch(data_path, model_name='VGGSNN', use_cifar10=False):
    """
    诊断类别数不匹配问题
    """
    print(f"\n{'='*80}")
    print(f"诊断类别数不匹配问题")
    print(f"{'='*80}\n")
    
    # 1. 加载数据集
    print("1. 加载数据集...")
    try:
        train_dataset, val_dataset = auto_build_cifar(
            data_path=data_path,
            use_cifar10=use_cifar10,
            download=False
        )
        print(f"✅ 数据集加载成功")
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   测试集: {len(val_dataset)} 样本")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 2. 统计实际类别数
    print(f"\n2. 统计数据集实际类别...")
    labels_set = set()
    max_label = -1
    min_label = float('inf')
    
    # 检查前1000个样本(足够统计类别)
    check_size = min(1000, len(train_dataset))
    for i in range(check_size):
        _, label = train_dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        labels_set.add(label)
        max_label = max(max_label, label)
        min_label = min(min_label, label)
    
    actual_classes = len(labels_set)
    print(f"✅ 统计完成 (检查了前{check_size}个样本)")
    print(f"   实际类别数: {actual_classes}")
    print(f"   标签范围: [{min_label}, {max_label}]")
    print(f"   标签列表: {sorted(list(labels_set))[:20]}{'...' if len(labels_set) > 20 else ''}")
    
    # 3. 检查模型类别数
    print(f"\n3. 检查模型配置...")
    if model_name == 'VGGSNN':
        model = VGGSNN()
        # VGG默认10类
        model_classes = 10
        print(f"   模型: VGGSNN")
        print(f"   默认类别数: {model_classes}")
    elif model_name == 'resnet19':
        model = resnet19(num_classes=10)
        model_classes = 10
        print(f"   模型: ResNet19")
        print(f"   当前类别数: {model_classes}")
    else:
        print(f"❌ 未知模型: {model_name}")
        return
    
    # 4. 诊断结果
    print(f"\n{'='*80}")
    print(f"诊断结果:")
    print(f"{'='*80}")
    
    if actual_classes == model_classes and max_label == model_classes - 1:
        print(f"✅ 类别数匹配!")
        print(f"   数据集类别: {actual_classes}")
        print(f"   模型类别: {model_classes}")
    else:
        print(f"❌ 类别数不匹配!")
        print(f"   数据集实际类别: {actual_classes}")
        print(f"   数据集标签范围: [{min_label}, {max_label}]")
        print(f"   模型类别数: {model_classes}")
        print(f"   模型接受范围: [0, {model_classes-1}]")
        
        print(f"\n问题分析:")
        if max_label >= model_classes:
            print(f"   ⚠️  标签最大值 {max_label} 超出模型范围 [0, {model_classes-1}]")
        if actual_classes > model_classes:
            print(f"   ⚠️  数据集有 {actual_classes} 个类别,但模型只支持 {model_classes} 类")
        if min_label < 0:
            print(f"   ⚠️  标签包含负值 {min_label}")
        
        print(f"\n修复建议:")
        print(f"{'='*80}")
        
        # 建议1: 修改模型类别数
        print(f"\n方案1: 修改模型类别数(推荐)")
        print(f"---------------------------------------")
        print(f"在 main_training_distribute_improved.py 中修改:")
        print(f"")
        print(f"if args.model == 'VGGSNN':")
        print(f"    model = VGGSNN(num_classes={actual_classes})  # 改为实际类别数")
        print(f"elif args.model == 'resnet19':")
        print(f"    model = resnet19(num_classes={actual_classes})")
        
        # 建议2: 检查数据集
        print(f"\n方案2: 检查数据集是否正确")
        print(f"---------------------------------------")
        print(f"运行检测工具:")
        print(f"  python check_data_format.py {data_path}")
        print(f"")
        print(f"确认:")
        print(f"  - 数据集是否为{'CIFAR-10' if use_cifar10 else 'CIFAR-100'}")
        print(f"  - 类别文件夹是否正确组织")
        print(f"  - 标签是否从0开始")
        
        # 建议3: 使用正确的参数
        print(f"\n方案3: 使用正确的数据集参数")
        print(f"---------------------------------------")
        if actual_classes == 100 and use_cifar10:
            print(f"  ⚠️  检测到100个类别,但使用了 use_cifar10=True")
            print(f"  建议: 使用 use_cifar10=False 或 --cifar100")
        elif actual_classes == 10 and not use_cifar10:
            print(f"  ⚠️  检测到10个类别,但使用了 use_cifar10=False")
            print(f"  建议: 使用 use_cifar10=True 或 --cifar10")
    
    print(f"\n{'='*80}\n")
    
    return actual_classes, model_classes


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='诊断类别数不匹配')
    parser.add_argument('data_path', type=str, help='数据集路径')
    parser.add_argument('--model', type=str, default='VGGSNN',
                       choices=['VGGSNN', 'resnet19'],
                       help='模型名称')
    parser.add_argument('--cifar10', action='store_true',
                       help='使用CIFAR-10 (默认CIFAR-100)')
    
    args = parser.parse_args()
    
    diagnose_class_mismatch(args.data_path, args.model, args.cifar10)
