"""
TET分布式训练改进版
添加功能:
1. TensorBoard日志记录
2. 完善的checkpoint保存机制
3. 超参数配置保存
4. 训练历史记录
"""
import argparse
import shutil
import os
import time
import warnings
import json
from datetime import datetime
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from models.resnet_models import resnet19
from models.VGG_models import VGGSNN
import data_loaders
from functions import TET_loss, seed_all


# 默认GPU设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training (Improved)')

# ==================== 数据和训练参数 ====================
parser.add_argument('--data-path',
                    default='/data_smr/dataset/cifar10-dvs',
                    type=str,
                    help='path to the dataset')
parser.add_argument('-j', '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), total batch size of all GPUs')
parser.add_argument('--lr', '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('-p', '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')

# ==================== SNN参数 ====================
parser.add_argument('--T',
                    default=10,
                    type=int,
                    metavar='N',
                    help='snn simulation time steps (default: 10)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='target mean for MSE regularization (default: 1.0)')
parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lamb',
                    default=0.0001,
                    type=float,
                    metavar='N',
                    help='MSE regularization weight (default: 0.0001)')

# ==================== 模型和优化器 ====================
parser.add_argument('--model',
                    default='VGGSNN',
                    type=str,
                    choices=['VGGSNN', 'resnet19'],
                    help='model architecture')
parser.add_argument('--num-classes',
                    default=10,
                    type=int,
                    help='number of classes (default: 10 for DVS-CIFAR10)')
parser.add_argument('--dataset',
                    default='dvscifar10',
                    type=str,
                    choices=['dvscifar10', 'cifar10', 'cifar100'],
                    help='dataset type: dvscifar10, cifar10, cifar100')
parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training')

# ==================== 保存和日志参数 (新增) ====================
parser.add_argument('--output-dir',
                    default='./runs',
                    type=str,
                    help='directory to save checkpoints and logs')
parser.add_argument('--exp-name',
                    default='',
                    type=str,
                    help='experiment name (default: auto-generated from params)')
parser.add_argument('--save-freq',
                    default=10,
                    type=int,
                    help='save checkpoint every N epochs (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-tensorboard',
                    action='store_true',
                    help='disable tensorboard logging')

# ==================== 评估参数 ====================
parser.add_argument('-e', '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

args = parser.parse_args()


def reduce_mean(tensor, nprocs):
    """所有GPU的tensor取平均"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def setup_experiment_dir(args):
    """设置实验目录结构"""
    if args.exp_name == '':
        # 自动生成实验名称
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'{args.model}_T{args.T}_lr{args.lr}_lamb{args.lamb}_{timestamp}'
    
    # 转换为绝对路径,确保路径有效
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    
    # 创建目录结构
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    args.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    args.log_dir = os.path.join(exp_dir, 'logs')
    
    if args.local_rank == 0:  # 只在主进程创建目录
        try:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            os.makedirs(args.log_dir, exist_ok=True)
        except PermissionError as e:
            print(f"错误: 无法创建目录 {exp_dir}")
            print(f"权限错误: {e}")
            print(f"请确保output_dir指向有写入权限的目录")
            raise
        
        # 保存超参数配置
        config_path = os.path.join(exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        print(f"\n{'='*80}")
        print(f"实验目录: {exp_dir}")
        print(f"Checkpoints: {args.checkpoint_dir}")
        print(f"Logs: {args.log_dir}")
        print(f"Config saved to: {config_path}")
        print(f"{'='*80}\n")
    
    return exp_dir


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """保存完整checkpoint"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    # 同时保存为latest
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    shutil.copyfile(filepath, latest_path)
    
    # 如果是最佳模型,额外保存
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        shutil.copyfile(filepath, best_path)


def load_checkpoint(args, model, optimizer, scheduler):
    """加载checkpoint恢复训练"""
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        
        # 加载到CPU,避免GPU内存问题
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        
        # 加载模型权重
        model.load_state_dict(checkpoint['state_dict'])
        
        # 加载优化器和调度器
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}, best_acc1 {best_acc1:.3f})")
        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")
        return 0.0


def main():
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    """每个GPU上的训练worker"""
    args.local_rank = local_rank

    # 设置随机种子
    if args.seed is not None:
        seed_all(args.seed + local_rank)  # 每个进程使用不同种子
        cudnn.deterministic = True
        if local_rank == 0:
            warnings.warn('You have chosen to seed training. '
                         'This will turn on the CUDNN deterministic setting, '
                         'which can slow down your training considerably!')

    best_acc1 = 0.0

    # 初始化分布式进程组
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        world_size=args.nprocs,
        rank=local_rank
    )

    # 设置实验目录 (等待主进程创建完成)
    if local_rank == 0:
        exp_dir = setup_experiment_dir(args)
    dist.barrier()  # 同步所有进程
    
    # 其他进程也需要知道目录路径
    if local_rank != 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.exp_name == '':
            args.exp_name = f'{args.model}_T{args.T}_lr{args.lr}_lamb{args.lamb}_{timestamp}'
        exp_dir = os.path.join(args.output_dir, args.exp_name)
        args.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        args.log_dir = os.path.join(exp_dir, 'logs')

    # 创建模型 (使用正确的类别数)
    if local_rank == 0:
        print(f"创建模型: {args.model}, 类别数: {args.num_classes}")
    
    if args.model == 'VGGSNN':
        model = VGGSNN(num_classes=args.num_classes)
    elif args.model == 'resnet19':
        model = resnet19(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model.T = args.T

    # 分配到对应GPU
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    
    # 调整batch size
    args.batch_size = int(args.batch_size / args.nprocs)
    
    # 包装为DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=0, T_max=args.epochs
    )

    # 加载checkpoint (如果指定)
    if args.resume:
        best_acc1 = load_checkpoint(args, model.module, optimizer, scheduler)

    cudnn.benchmark = True

    # 加载数据 (支持多种数据集)
    if local_rank == 0:
        print(f"加载数据集: {args.dataset}, 路径: {args.data_path}")
    
    if args.dataset == 'dvscifar10':
        train_dataset, val_dataset = data_loaders.build_dvscifar(args.data_path)
    elif args.dataset == 'cifar10':
        train_dataset, val_dataset = data_loaders.auto_build_cifar(
            data_path=args.data_path,
            use_cifar10=True
        )
    elif args.dataset == 'cifar100':
        train_dataset, val_dataset = data_loaders.auto_build_cifar(
            data_path=args.data_path,
            use_cifar10=False
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if local_rank == 0:
        print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(val_dataset)}")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler
    )

    # 创建TensorBoard writer (只在主进程)
    writer = None
    if local_rank == 0 and not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"TensorBoard logging to: {args.log_dir}")
        print(f"Run: tensorboard --logdir={args.log_dir}")

    # 仅评估模式
    if args.evaluate:
        acc1, acc5 = validate(val_loader, model, criterion, local_rank, args)
        if local_rank == 0:
            print(f'Validation Results: Acc@1 {acc1:.3f}, Acc@5 {acc5:.3f}')
        return

    # 训练循环
    if local_rank == 0:
        print(f"\n{'='*80}")
        print(f"开始训练: {args.epochs} epochs")
        print(f"模型: {args.model}, T={args.T}, Batch size={args.batch_size * args.nprocs}")
        print(f"TET: {args.TET}, lamb={args.lamb}, means={args.means}")
        print(f"{'='*80}\n")

    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # 训练一个epoch
        train_loss, train_acc1, train_acc5 = train(
            train_loader, model, criterion, optimizer, 
            epoch, local_rank, args
        )

        # 验证
        val_loss, val_acc1, val_acc5 = validate(
            val_loader, model, criterion, local_rank, args
        )

        # 学习率调度
        scheduler.step()

        # 记录到TensorBoard (只在主进程)
        if local_rank == 0 and writer is not None:
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Acc1', train_acc1, epoch)
            writer.add_scalar('Train/Acc5', train_acc5, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Acc1', val_acc1, epoch)
            writer.add_scalar('Val/Acc5', val_acc5, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # 判断是否最佳
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        # 保存checkpoint (只在主进程)
        if local_rank == 0:
            t2 = time.time()
            epoch_time = t2 - t1
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train - Loss: {train_loss:.4f}, Acc@1: {train_acc1:.2f}%, Acc@5: {train_acc5:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")
            print(f"  Best Acc@1: {best_acc1:.2f}% {'(*)' if is_best else ''}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存checkpoint
            save_flag = (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1
            if save_flag or is_best:
                state = {
                    'epoch': epoch + 1,
                    'model': args.model,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': vars(args)
                }
                
                if save_flag:
                    filename = f'checkpoint_epoch{epoch}.pth'
                    save_checkpoint(state, is_best, args.checkpoint_dir, filename)
                    print(f"  Saved: {filename}")
                
                if is_best:
                    save_checkpoint(state, is_best, args.checkpoint_dir)
                    print(f"  Saved best model!")
            
            print(f"{'='*80}\n")

    # 训练结束
    if local_rank == 0:
        if writer is not None:
            writer.close()
        
        print(f"\n{'='*80}")
        print(f"训练完成!")
        print(f"最佳验证准确率: {best_acc1:.2f}%")
        print(f"模型和日志保存在: {exp_dir}")
        print(f"{'='*80}\n")


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    """训练一个epoch"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()
    end = time.time()
    
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # 前向传播
        output = model(images)
        mean_out = torch.mean(output, dim=1)
        
        # 计算损失
        if not args.TET:
            loss = criterion(mean_out, target)
        else:
            loss = TET_loss(output, target, criterion, args.means, args.lamb)

        # 计算准确率
        acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

        # 同步所有GPU的指标
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and local_rank == 0:
            progress.display(i)

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, local_rank, args):
    """验证模型"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Val: '
    )

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            output = model(images)
            mean_out = torch.mean(output, dim=1)
            loss = criterion(mean_out, target)

            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and local_rank == 0:
                progress.display(i)

    if local_rank == 0:
        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """进度显示"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
