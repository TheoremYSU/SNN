import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
from os.path import isfile, join

warnings.filterwarnings('ignore')


def build_cifar(cutout=False, use_cifar10=True, download=False, data_path='./raw/'):
    """
    构建CIFAR数据集,支持多种格式:
    1. 标准二进制格式 (file.txt, meta, test, train文件)
    2. 图片文件夹格式 (train/, test/ 文件夹包含图片)
    
    Args:
        cutout: 是否使用cutout数据增强
        use_cifar10: True使用CIFAR-10, False使用CIFAR-100
        download: 是否下载数据集
        data_path: 数据集路径
    """
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=data_path,
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root=data_path,
                              train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=data_path,
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root=data_path,
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset


def build_cifar_from_images(data_path, use_cifar10=True, cutout=False):
    """
    从图片文件夹构建CIFAR数据集
    
    数据格式要求:
    data_path/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.png
    │   │   └── img2.png
    │   └── class2/
    │       └── ...
    └── test/
        ├── class1/
        └── class2/
    
    或者:
    data_path/
    ├── train/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── test/
        ├── img1.png
        └── ...
    
    Args:
        data_path: 数据集根目录
        use_cifar10: True使用CIFAR-10标准化, False使用CIFAR-100标准化
        cutout: 是否使用cutout数据增强
    """
    aug = [transforms.Resize(32), transforms.RandomCrop(32, padding=4), 
           transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError(f"数据路径 {data_path} 必须包含 train/ 和 test/ 文件夹")
    
    train_dataset = ImageFolder(train_path, transform=transform_train)
    val_dataset = ImageFolder(test_path, transform=transform_test)
    
    print(f"加载图片数据集:")
    print(f"  训练集: {len(train_dataset)} 张图片")
    print(f"  测试集: {len(val_dataset)} 张图片")
    print(f"  类别数: {len(train_dataset.classes)}")
    
    return train_dataset, val_dataset


def auto_build_cifar(data_path, use_cifar10=True, download=False, cutout=False):
    """
    自动检测CIFAR数据格式并加载
    
    支持的格式:
    1. 标准二进制格式: data_path包含 train, test 二进制文件
    2. 图片文件夹格式: data_path包含 train/, test/ 文件夹
    
    Args:
        data_path: 数据集路径
        use_cifar10: True使用CIFAR-10, False使用CIFAR-100
        download: 是否下载数据集(仅对标准格式有效)
        cutout: 是否使用cutout数据增强
    """
    data_path = os.path.expanduser(data_path)
    
    # 检查是否为图片文件夹格式
    train_folder = os.path.join(data_path, 'train')
    test_folder = os.path.join(data_path, 'test')
    
    if os.path.isdir(train_folder) and os.path.isdir(test_folder):
        # 检查train文件夹是否包含子文件夹(类别)或直接包含图片
        train_contents = os.listdir(train_folder)
        if train_contents and os.path.isdir(os.path.join(train_folder, train_contents[0])):
            print(f"检测到图片文件夹格式 (带类别子文件夹): {data_path}")
            return build_cifar_from_images(data_path, use_cifar10, cutout)
        elif any(f.endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.PNG', '.JPG')) 
                 for f in train_contents if os.path.isfile(os.path.join(train_folder, f))):
            print(f"检测到图片文件夹格式 (无类别子文件夹): {data_path}")
            print("警告: 图片格式需要类别子文件夹结构,当前格式可能无法正确加载")
            return build_cifar_from_images(data_path, use_cifar10, cutout)
    
    # 检查是否为标准二进制格式
    train_file = os.path.join(data_path, 'cifar-100-python', 'train') if not use_cifar10 else None
    test_file = os.path.join(data_path, 'cifar-100-python', 'test') if not use_cifar10 else None
    
    if not use_cifar10:
        if os.path.isfile(train_file) or download:
            print(f"使用标准CIFAR-100二进制格式: {data_path}")
            return build_cifar(cutout, use_cifar10, download, data_path)
    else:
        print(f"使用标准CIFAR-10二进制格式: {data_path}")
        return build_cifar(cutout, use_cifar10, download, data_path)
    
    # 如果都不匹配,尝试使用标准格式
    print(f"未能自动检测数据格式,尝试使用标准二进制格式: {data_path}")
    return build_cifar(cutout, use_cifar10, download, data_path)

def build_mnist(download=False):
    train_dataset = MNIST(root='./raw/',
                             train=True, download=download, transform=transforms.ToTensor())
    val_dataset = MNIST(root='./raw/',
                           train=False, download=download, transform=transforms.ToTensor())
    return train_dataset, val_dataset


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        # print(data.shape)
        # if self.train:
        new_data = []
        for t in range(data.size(-1)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[...,t]))))
        data = torch.stack(new_data, dim=0)
        if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def build_dvscifar(path):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVSCifar10(root=train_path, transform=True)
    val_dataset = DVSCifar10(root=val_path)

    return train_dataset, val_dataset

def build_imagenet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    root = '/data_smr/dataset/ImageNet'
    train_root = os.path.join(root,'train')
    val_root = os.path.join(root,'val')
    train_dataset = ImageFolder(
        train_root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return train_dataset, val_dataset

if __name__ == '__main__':
    train_set, test_set = build_mnist(download=True)
