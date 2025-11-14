#!/usr/bin/env python3
"""
测试GPU在multiprocessing中的可用性
用于诊断 "ProcessGroupNCCL is only supported with GPUs, no GPUs found!" 错误
"""
import os
import sys
import torch
import torch.multiprocessing as mp


def print_gpu_info(rank, title):
    """打印GPU信息"""
    print(f"\n{'='*80}")
    print(f"{title} (Rank {rank})")
    print(f"{'='*80}")
    print(f"Process ID: {os.getpid()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA不可用!")
    
    print(f"{'='*80}\n")


def worker(rank, world_size):
    """子进程worker"""
    print_gpu_info(rank, f"子进程 {rank}")
    
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] ❌ 错误: CUDA在子进程中不可用!")
        return False
    
    if torch.cuda.device_count() == 0:
        print(f"[Rank {rank}] ❌ 错误: 可用GPU数量为0!")
        return False
    
    print(f"[Rank {rank}] ✓ CUDA可用,GPU数量: {torch.cuda.device_count()}")
    
    # 尝试初始化NCCL
    try:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:23457',  # 使用不同端口避免冲突
            world_size=world_size,
            rank=rank
        )
        print(f"[Rank {rank}] ✓ NCCL初始化成功!")
        dist.destroy_process_group()
        return True
    except Exception as e:
        print(f"[Rank {rank}] ❌ NCCL初始化失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "="*80)
    print("GPU Multiprocessing 诊断工具")
    print("="*80)
    
    # 检查主进程
    print_gpu_info(0, "主进程")
    
    if not torch.cuda.is_available():
        print("❌ 主进程中CUDA不可用!")
        print("\n可能的原因:")
        print("  1. PyTorch未安装CUDA版本")
        print("  2. CUDA驱动未正确安装")
        print("  3. CUDA_VISIBLE_DEVICES环境变量设置错误")
        print("\n解决方案:")
        print("  1. 检查PyTorch安装: python -c 'import torch; print(torch.version.cuda)'")
        print("  2. 检查CUDA驱动: nvidia-smi")
        print("  3. 设置环境变量: export CUDA_VISIBLE_DEVICES=0,1,2,3")
        return
    
    nprocs = torch.cuda.device_count()
    print(f"\n检测到 {nprocs} 个GPU,将启动 {nprocs} 个子进程进行测试...\n")
    
    # 测试multiprocessing spawn
    try:
        mp.spawn(worker, args=(nprocs,), nprocs=nprocs, join=True)
        print("\n" + "="*80)
        print("✓ 所有测试通过! GPU在multiprocessing中正常工作")
        print("="*80 + "\n")
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ 测试失败: {e}")
        print("="*80)
        print("\n这说明你的环境存在以下问题:")
        print("  1. PyTorch的CUDA支持在multiprocessing中不工作")
        print("  2. NCCL后端无法初始化")
        print("\n可能的解决方案:")
        print("  1. 重新安装PyTorch CUDA版本:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("  2. 检查CUDA版本兼容性:")
        print("     nvidia-smi 查看CUDA版本")
        print("     python -c 'import torch; print(torch.version.cuda)' 查看PyTorch CUDA版本")
        print("  3. 检查NCCL库:")
        print("     python -c 'import torch.distributed'")
        print("\n")
        raise


if __name__ == '__main__':
    main()
