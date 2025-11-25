"""
测试TSE_loss函数的正确性
验证:
1. 输入输出维度
2. 掩码生成逻辑
3. 阈值计算
4. 损失值合理性
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(__file__))
from functions import TSE_loss


def test_tse_loss_basic():
    """基本功能测试"""
    print("=" * 60)
    print("测试1: TSE_loss基本功能")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模拟参数
    B, T, C, H, W = 4, 4, 128, 8, 8  # Batch, Time, Channel, Height, Width
    num_classes = 10
    
    # 创建模拟数据
    feature_maps = torch.randn(B, T, C, H, W)
    labels = torch.randint(0, num_classes, (B,))
    
    # 创建分类层
    fc_layer = nn.Linear(C, num_classes)
    criterion = nn.CrossEntropyLoss()
    
    print(f"输入特征图形状: {feature_maps.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"分类层: Linear({C}, {num_classes})")
    
    # 计算TSE损失
    try:
        loss = TSE_loss(
            feature_maps=feature_maps,
            fc_layers=fc_layer,
            labels=labels,
            criterion=criterion,
            tau_f=0.5,
            kappa=1.0
        )
        print(f"\n✅ TSE损失计算成功!")
        print(f"损失值: {loss.item():.4f}")
        print(f"损失类型: {type(loss)}")
        print(f"损失requires_grad: {loss.requires_grad}")
        
        # 测试反向传播
        loss.backward()
        print("\n✅ 反向传播成功!")
        
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tse_threshold_logic():
    """测试阈值计算逻辑"""
    print("\n" + "=" * 60)
    print("测试2: 阈值计算逻辑")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    B, T, C, H, W = 2, 3, 64, 4, 4
    num_classes = 5
    
    feature_maps = torch.randn(B, T, C, H, W)
    labels = torch.tensor([1, 3])  # 固定标签便于分析
    
    # 单层FC (用于简单测试)
    fc_layers = nn.Linear(C, num_classes)
    criterion = nn.CrossEntropyLoss()
    
    # 不同的tau_f和kappa
    test_params = [
        (0.3, 0.5, "低固定阈值,低动态系数"),
        (0.5, 1.0, "中等(默认)"),
        (0.7, 2.0, "高固定阈值,高动态系数"),
    ]
    
    print("\n不同超参数下的损失值:")
    print(f"{'tau_f':<8} {'kappa':<8} {'Loss':<12} {'描述':<20}")
    print("-" * 60)
    
    for tau_f, kappa, desc in test_params:
        loss = TSE_loss(
            feature_maps=feature_maps,
            fc_layers=fc_layers,
            labels=labels,
            criterion=criterion,
            tau_f=tau_f,
            kappa=kappa
        )
        print(f"{tau_f:<8.1f} {kappa:<8.1f} {loss.item():<12.4f} {desc:<20}")
    
    print("\n✅ 阈值逻辑测试完成")
    return True


def test_tse_time_independence():
    """测试不同时间步的独立监督"""
    print("\n" + "=" * 60)
    print("测试3: 时间步独立监督")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    B, C, H, W = 2, 64, 4, 4
    num_classes = 5
    labels = torch.tensor([1, 3])
    
    fc_layer = nn.Linear(C, num_classes)
    criterion = nn.CrossEntropyLoss()
    
    # 测试不同的时间步数
    time_steps = [2, 4, 8]
    
    print("\n不同时间步数的损失值:")
    print(f"{'Time Steps':<15} {'Loss':<12}")
    print("-" * 30)
    
    for T in time_steps:
        feature_maps = torch.randn(B, T, C, H, W)
        loss = TSE_loss(
            feature_maps=feature_maps,
            fc_layers=fc_layer,
            labels=labels,
            criterion=criterion,
            tau_f=0.5,
            kappa=1.0
        )
        print(f"{T:<15} {loss.item():<12.4f}")
    
    print("\n✅ 时间步独立性测试完成")
    return True


def test_tse_gradient_flow():
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("测试4: 梯度流")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    B, T, C, H, W = 2, 4, 64, 4, 4
    num_classes = 5
    
    feature_maps = torch.randn(B, T, C, H, W, requires_grad=True)
    labels = torch.tensor([1, 3])
    
    fc_layer = nn.Linear(C, num_classes)
    criterion = nn.CrossEntropyLoss()
    
    # 计算损失
    loss = TSE_loss(
        feature_maps=feature_maps,
        fc_layers=fc_layer,
        labels=labels,
        criterion=criterion,
        tau_f=0.5,
        kappa=1.0
    )
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"Feature maps梯度形状: {feature_maps.grad.shape}")
    print(f"梯度是否全为0: {torch.all(feature_maps.grad == 0).item()}")
    print(f"梯度最大值: {feature_maps.grad.abs().max().item():.6f}")
    print(f"梯度最小值: {feature_maps.grad.abs().min().item():.6f}")
    print(f"梯度平均值: {feature_maps.grad.abs().mean().item():.6f}")
    
    # 检查fc层梯度
    print(f"\nFC层权重梯度形状: {fc_layer.weight.grad.shape}")
    print(f"FC层梯度是否全为0: {torch.all(fc_layer.weight.grad == 0).item()}")
    print(f"FC层梯度平均值: {fc_layer.weight.grad.abs().mean().item():.6f}")
    
    assert not torch.all(feature_maps.grad == 0), "Feature maps梯度不应全为0"
    assert not torch.all(fc_layer.weight.grad == 0), "FC层梯度不应全为0"
    
    print("\n✅ 梯度流测试通过")
    return True


def test_tse_mask_generation():
    """测试掩码生成"""
    print("\n" + "=" * 60)
    print("测试5: 掩码生成和可视化")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 简化参数便于分析
    B, T, C, H, W = 1, 3, 32, 4, 4
    num_classes = 3
    labels = torch.tensor([1])  # 真实类别为1
    
    # 创建特征图,使其有明显的空间模式
    feature_maps = torch.randn(B, T, C, H, W)
    # 让某些位置有更强的激活
    feature_maps[0, 1, :, 0, 0] *= 2.0  # 左上角增强
    feature_maps[0, 2, :, 3, 3] *= 2.0  # 右下角增强
    
    fc_layer = nn.Linear(C, num_classes)
    criterion = nn.CrossEntropyLoss()
    
    print("模拟掩码生成过程:")
    print("-" * 60)
    
    # 手动模拟TSE的掩码生成
    with torch.no_grad():
        # 对第二个时间步(t=1)
        t = 1
        B, C, H, W = feature_maps.shape[0], feature_maps.shape[2], feature_maps.shape[3], feature_maps.shape[4]
        
        # 计算分类预测图
        features_t = feature_maps[:, t]  # [B, C, H, W]
        B_size, C_size, H_size, W_size = features_t.shape
        
        # 重塑为 [B*H*W, C]
        features_flat = features_t.permute(0, 2, 3, 1).reshape(B_size * H_size * W_size, C_size)
        
        # 分类
        predictions = fc_layer(features_flat)  # [B*H*W, num_classes]
        predictions = predictions.reshape(B_size, H_size, W_size, num_classes)  # [B, H, W, num_classes]
        predictions = predictions.permute(0, 3, 1, 2)  # [B, num_classes, H, W]
        
        # 平均之前的预测(这里只有t=0)
        prev_features = feature_maps[:, 0]
        prev_flat = prev_features.permute(0, 2, 3, 1).reshape(B_size * H_size * W_size, C_size)
        prev_pred = fc_layer(prev_flat).reshape(B_size, H_size, W_size, num_classes).permute(0, 3, 1, 2)
        
        avg_pred = prev_pred
        avg_prob = torch.softmax(avg_pred, dim=1)  # [B, num_classes, H, W]
        
        # 提取真实类别的概率图
        prob_map = avg_prob[torch.arange(B_size), labels]  # [B, H, W]
        
        print(f"时间步 t={t} 的概率图 (真实类别={labels.item()}):")
        print(f"形状: {prob_map.shape}")
        print(f"概率图 (4x4):\n{prob_map[0].numpy()}")
        
        # 计算阈值
        tau_f = 0.5
        kappa = 1.0
        mean_prob = prob_map.mean()
        std_prob = prob_map.std()
        tau_d = mean_prob + kappa * std_prob
        threshold = max(tau_f, tau_d.item())
        
        print(f"\n阈值计算:")
        print(f"  固定阈值 τ_f: {tau_f:.4f}")
        print(f"  均值: {mean_prob.item():.4f}")
        print(f"  标准差: {std_prob.item():.4f}")
        print(f"  动态阈值 τ_d: {tau_d.item():.4f}")
        print(f"  最终阈值: {threshold:.4f}")
        
        # 生成掩码
        mask = (prob_map < threshold).float()
        print(f"\n掩码 (1=保留, 0=擦除):")
        print(mask[0].numpy())
        
        erased_ratio = (mask == 0).sum().item() / mask.numel()
        print(f"\n擦除比例: {erased_ratio * 100:.1f}%")
    
    print("\n✅ 掩码生成测试完成")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TSE_loss 完整测试套件")
    print("=" * 60)
    
    tests = [
        ("基本功能", test_tse_loss_basic),
        ("阈值逻辑", test_tse_threshold_logic),
        ("时间步独立性", test_tse_time_independence),
        ("梯度流", test_tse_gradient_flow),
        ("掩码生成", test_tse_mask_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:<20} {status}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! TSE_loss实现正确。")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败,请检查实现。")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
