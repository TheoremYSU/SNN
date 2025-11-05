"""
测试单步训练模式是否正常工作
"""
import torch
import sys
sys.path.append('.')

from spiking_resnet import spiking_resnet18
from spikingjelly.clock_driven import functional

def test_single_step_mode():
    print("=" * 60)
    print("测试单步训练模式")
    print("=" * 60)
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    T = 4
    batch_size = 2
    print(f"时间步 T = {T}")
    print(f"批次大小 = {batch_size}")
    
    # 创建模型
    model = spiking_resnet18(T=T, num_classes=1000)
    model.to(device)
    model.eval()
    
    # 创建随机输入
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 重置网络
    functional.reset_net(model)
    
    # 测试训练模式
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 模拟一次训练迭代
    target = torch.randint(0, 1000, (batch_size,)).to(device)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"\n训练测试:")
    print(f"损失值: {loss.item():.4f}")
    print(f"梯度是否正常: {all(p.grad is not None for p in model.parameters() if p.requires_grad)}")
    
    # 重置网络
    functional.reset_net(model)
    
    print("\n" + "=" * 60)
    print("✓ 单步训练模式测试通过!")
    print("=" * 60)
    
    # 显存使用情况
    if torch.cuda.is_available():
        print(f"\nGPU显存使用:")
        print(f"已分配: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"已缓存: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

if __name__ == '__main__':
    test_single_step_mode()
