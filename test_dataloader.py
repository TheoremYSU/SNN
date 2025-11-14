"""
快速测试CIFAR数据加载

用于验证你的数据格式是否正确加载
"""
import sys
import torch
from data_loaders import auto_build_cifar, build_cifar, build_cifar_from_images


def test_data_loading(data_path, use_cifar10=False, format_type='auto'):
    """
    测试数据加载
    
    Args:
        data_path: 数据集路径
        use_cifar10: True为CIFAR-10, False为CIFAR-100
        format_type: 'auto', 'binary', 'images'
    """
    print(f"\n{'='*80}")
    print(f"测试数据加载")
    print(f"{'='*80}")
    print(f"数据路径: {data_path}")
    print(f"数据集: {'CIFAR-10' if use_cifar10 else 'CIFAR-100'}")
    print(f"加载方式: {format_type}")
    print(f"{'='*80}\n")
    
    try:
        # 加载数据集
        if format_type == 'auto':
            print("使用自动检测格式...")
            train_dataset, val_dataset = auto_build_cifar(
                data_path=data_path,
                use_cifar10=use_cifar10,
                download=False
            )
        elif format_type == 'binary':
            print("使用标准二进制格式...")
            train_dataset, val_dataset = build_cifar(
                use_cifar10=use_cifar10,
                data_path=data_path,
                download=False
            )
        elif format_type == 'images':
            print("使用图片文件夹格式...")
            train_dataset, val_dataset = build_cifar_from_images(
                data_path=data_path,
                use_cifar10=use_cifar10
            )
        else:
            raise ValueError(f"未知格式类型: {format_type}")
        
        print(f"\n✅ 数据集加载成功!")
        print(f"{'='*80}")
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(val_dataset)}")
        
        # 测试加载一个样本
        print(f"\n测试加载单个样本...")
        sample_data, sample_label = train_dataset[0]
        print(f"  数据形状: {sample_data.shape}")
        print(f"  数据类型: {sample_data.dtype}")
        print(f"  数据范围: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
        print(f"  标签: {sample_label}")
        print(f"  标签类型: {type(sample_label)}")
        
        # 测试加载一个batch
        print(f"\n测试加载batch...")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        batch_data, batch_labels = next(iter(train_loader))
        print(f"  Batch数据形状: {batch_data.shape}")
        print(f"  Batch标签形状: {batch_labels.shape}")
        print(f"  Batch标签: {batch_labels.tolist()}")
        
        # 统计类别分布(测试集前100个样本)
        print(f"\n统计类别分布(测试集前100个样本)...")
        labels_count = {}
        for i in range(min(100, len(val_dataset))):
            _, label = val_dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels_count[label] = labels_count.get(label, 0) + 1
        
        print(f"  检测到 {len(labels_count)} 个不同类别")
        print(f"  类别分布: {dict(sorted(labels_count.items())[:10])}...")
        
        expected_classes = 10 if use_cifar10 else 100
        if len(set(labels_count.keys())) <= expected_classes:
            print(f"  ✅ 类别数符合预期 (≤{expected_classes})")
        else:
            print(f"  ⚠️  警告: 检测到 {len(labels_count)} 个类别,预期 {expected_classes}")
        
        print(f"\n{'='*80}")
        print(f"✅ 所有测试通过!")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ 数据加载失败!")
        print(f"{'='*80}")
        print(f"错误信息: {e}")
        print(f"\n建议:")
        print(f"1. 检查数据路径是否正确")
        print(f"2. 运行格式检测: python check_data_format.py {data_path}")
        print(f"3. 查看文档: CIFAR数据格式支持说明.md")
        print(f"{'='*80}\n")
        
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试CIFAR数据加载')
    parser.add_argument('data_path', type=str, help='数据集路径')
    parser.add_argument('--cifar10', action='store_true', 
                       help='使用CIFAR-10 (默认CIFAR-100)')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'binary', 'images'],
                       help='数据格式: auto(自动检测), binary(二进制), images(图片)')
    
    args = parser.parse_args()
    
    success = test_data_loading(
        data_path=args.data_path,
        use_cifar10=args.cifar10,
        format_type=args.format
    )
    
    sys.exit(0 if success else 1)
