"""
CIFAR数据格式检测工具

用于检测和验证CIFAR-10/100数据集的格式
"""
import os
import sys


def check_data_format(data_path):
    """检测数据集格式"""
    print(f"\n{'='*80}")
    print(f"检查数据路径: {data_path}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 路径不存在: {data_path}")
        return None
    
    # 列出根目录内容
    contents = os.listdir(data_path)
    print(f"根目录内容: {contents}\n")
    
    # 检测格式1: 标准二进制格式
    has_cifar10_batch = any('data_batch' in f for f in contents)
    has_cifar100_train = 'train' in contents and os.path.isfile(os.path.join(data_path, 'train'))
    has_cifar100_test = 'test' in contents and os.path.isfile(os.path.join(data_path, 'test'))
    has_meta = 'meta' in contents or 'batches.meta' in contents
    
    if has_cifar10_batch or (has_cifar100_train and has_cifar100_test):
        print("✅ 检测到格式1: 标准二进制格式")
        if has_cifar10_batch:
            print("   数据集: CIFAR-10")
            print(f"   训练文件: data_batch_1~5")
            print(f"   测试文件: test_batch")
        else:
            print("   数据集: CIFAR-100")
            print(f"   训练文件: train (二进制)")
            print(f"   测试文件: test (二进制)")
        if has_meta:
            print(f"   元数据: meta/batches.meta")
        print("\n使用方法:")
        print(f"   train_dataset, val_dataset = build_cifar(")
        print(f"       use_cifar10={'True' if has_cifar10_batch else 'False'},")
        print(f"       data_path='{data_path}'")
        print(f"   )")
        return "binary"
    
    # 检测格式2: 图片文件夹格式
    train_folder = os.path.join(data_path, 'train')
    test_folder = os.path.join(data_path, 'test')
    
    if os.path.isdir(train_folder) and os.path.isdir(test_folder):
        print("✅ 检测到格式2: 图片文件夹格式")
        
        # 检查train文件夹结构
        train_contents = os.listdir(train_folder)
        test_contents = os.listdir(test_folder)
        
        # 检查是否有类别子文件夹
        has_class_folders = False
        if train_contents:
            first_item = os.path.join(train_folder, train_contents[0])
            if os.path.isdir(first_item):
                has_class_folders = True
                classes = [d for d in train_contents if os.path.isdir(os.path.join(train_folder, d))]
                print(f"   训练集: {len(train_contents)} 个类别文件夹")
                print(f"   类别: {classes[:5]}{'...' if len(classes) > 5 else ''}")
                
                # 统计图片数量
                total_train = sum(len(os.listdir(os.path.join(train_folder, c))) 
                                 for c in classes if os.path.isdir(os.path.join(train_folder, c)))
                print(f"   训练图片数: {total_train}")
        
        if test_contents:
            first_item = os.path.join(test_folder, test_contents[0])
            if os.path.isdir(first_item):
                test_classes = [d for d in test_contents if os.path.isdir(os.path.join(test_folder, d))]
                print(f"   测试集: {len(test_contents)} 个类别文件夹")
                total_test = sum(len(os.listdir(os.path.join(test_folder, c))) 
                                for c in test_classes if os.path.isdir(os.path.join(test_folder, c)))
                print(f"   测试图片数: {total_test}")
        
        if not has_class_folders:
            print("   ⚠️  警告: train文件夹中未检测到类别子文件夹")
            print("   如果图片直接在train/test文件夹中,需要先组织成类别子文件夹结构")
            image_files = [f for f in train_contents 
                          if f.endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.PNG', '.JPG'))]
            print(f"   检测到 {len(image_files)} 个图片文件")
        
        print("\n使用方法:")
        print(f"   train_dataset, val_dataset = build_cifar_from_images(")
        print(f"       data_path='{data_path}',")
        print(f"       use_cifar10=False  # CIFAR-100使用False")
        print(f"   )")
        print("\n或使用自动检测:")
        print(f"   train_dataset, val_dataset = auto_build_cifar(")
        print(f"       data_path='{data_path}',")
        print(f"       use_cifar10=False")
        print(f"   )")
        return "images"
    
    # 未识别的格式
    print("❌ 未能识别数据格式")
    print("\n支持的格式:")
    print("1. 标准二进制格式:")
    print("   - CIFAR-10: data_batch_1~5, test_batch, batches.meta")
    print("   - CIFAR-100: train, test, meta")
    print("\n2. 图片文件夹格式:")
    print("   data_path/")
    print("   ├── train/")
    print("   │   ├── class0/")
    print("   │   │   ├── img1.png")
    print("   │   │   └── ...")
    print("   │   └── class1/")
    print("   └── test/")
    print("       ├── class0/")
    print("       └── class1/")
    return None


def convert_flat_to_folders(data_path, label_file):
    """
    将平铺的图片转换为类别文件夹结构
    
    如果你的图片直接在train/test文件夹中,没有类别子文件夹,
    可以使用此函数进行转换
    
    Args:
        data_path: 数据根目录
        label_file: 标签文件路径 (每行: 图片名 类别ID)
    """
    import shutil
    
    print(f"\n{'='*80}")
    print(f"转换图片为类别文件夹结构")
    print(f"{'='*80}\n")
    
    if not os.path.exists(label_file):
        print(f"❌ 标签文件不存在: {label_file}")
        return
    
    # 读取标签
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name, class_id = parts[0], parts[1]
                labels[img_name] = class_id
    
    print(f"读取 {len(labels)} 个标签")
    
    for split in ['train', 'test']:
        src_dir = os.path.join(data_path, split)
        if not os.path.exists(src_dir):
            continue
        
        print(f"\n处理 {split} 数据...")
        
        # 获取所有图片
        images = [f for f in os.listdir(src_dir) 
                 if f.endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.PNG', '.JPG'))]
        
        # 创建类别文件夹并移动图片
        moved = 0
        for img in images:
            if img in labels:
                class_id = labels[img]
                class_dir = os.path.join(src_dir, f'class_{class_id}')
                os.makedirs(class_dir, exist_ok=True)
                
                src_path = os.path.join(src_dir, img)
                dst_path = os.path.join(class_dir, img)
                shutil.move(src_path, dst_path)
                moved += 1
        
        print(f"  移动 {moved} 张图片到类别文件夹")
    
    print(f"\n✅ 转换完成!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CIFAR数据格式检测工具')
    parser.add_argument('data_path', type=str, help='数据集路径')
    parser.add_argument('--convert', action='store_true', 
                       help='转换平铺图片为类别文件夹结构')
    parser.add_argument('--label-file', type=str, default='',
                       help='标签文件路径 (用于转换)')
    
    args = parser.parse_args()
    
    if args.convert:
        if not args.label_file:
            print("❌ 错误: 转换模式需要指定 --label-file")
            sys.exit(1)
        convert_flat_to_folders(args.data_path, args.label_file)
    else:
        check_data_format(args.data_path)
