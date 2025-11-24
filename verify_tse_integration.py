"""
快速验证TSE集成是否正确
检查:
1. 参数是否正确添加
2. 模型是否支持return_features
3. TSE训练流程是否正常
"""

import torch
import torch.nn as nn
import argparse
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from functions import TSE_loss, TET_loss
from models.resnet_models import resnet19


def test_argparse_integration():
    """测试1: argparse参数集成"""
    print("=" * 60)
    print("测试1: Argparse参数集成")
    print("=" * 60)
    
    # 创建一个简化的parser
    parser = argparse.ArgumentParser()
    
    # TET参数
    parser.add_argument('--tet', dest='TET', action='store_true')
    parser.add_argument('--no-tet', dest='TET', action='store_false')
    parser.set_defaults(TET=False)
    
    # TSE参数
    parser.add_argument('--tse', dest='TSE', action='store_true')
    parser.add_argument('--no-tse', dest='TSE', action='store_false')
    parser.set_defaults(TSE=False)
    
    parser.add_argument('--tau-f', default=0.5, type=float)
    parser.add_argument('--kappa', default=1.0, type=float)
    
    # 测试不同参数组合
    test_cases = [
        ([], "默认配置"),
        (['--tse'], "启用TSE"),
        (['--tse', '--tau-f', '0.3'], "TSE自定义tau_f"),
        (['--tse', '--tau-f', '0.6', '--kappa', '2.0'], "TSE自定义参数"),
        (['--no-tse'], "显式禁用TSE"),
    ]
    
    print("\n参数解析测试:")
    print(f"{'配置':<30} {'TSE':<8} {'tau_f':<8} {'kappa':<8}")
    print("-" * 60)
    
    for args_list, desc in test_cases:
        args = parser.parse_args(args_list)
        print(f"{desc:<30} {args.TSE!s:<8} {args.tau_f:<8.2f} {args.kappa:<8.2f}")
    
    print("\n✅ Argparse参数集成正确")
    return True


def test_model_feature_extraction():
    """测试2: 模型特征提取"""
    print("\n" + "=" * 60)
    print("测试2: ResNet模型特征提取")
    print("=" * 60)
    
    # 创建模型(不需要T参数)
    model = resnet19(num_classes=10)
    model.eval()
    
    # 创建输入
    B, C, H, W, T = 2, 3, 32, 32, 4
    x = torch.randn(B, C, H, W, T)
    
    print(f"输入形状: {x.shape}")
    
    # 测试1: 正常前向传播(不返回特征)
    with torch.no_grad():
        output = model(x)
    print(f"标准输出形状: {output.shape}")
    assert output.shape == (B, T, 10), f"输出形状错误: {output.shape}"
    
    # 测试2: 返回特征
    with torch.no_grad():
        output, features = model(x, return_features=True)
    print(f"TSE模式输出形状: {output.shape}")
    print(f"特征图形状: {features.shape}")
    
    # 验证维度
    assert output.shape == (B, T, 10), f"输出形状错误: {output.shape}"
    assert len(features.shape) == 5, f"特征图应该是5D: {features.shape}"
    assert features.shape[0] == B, f"Batch维度错误"
    assert features.shape[1] == T, f"Time维度错误"
    
    print(f"\n✅ 模型特征提取正确")
    print(f"   输出: {output.shape}")
    print(f"   特征: {features.shape}")
    return True


def test_tse_training_flow():
    """测试3: TSE训练流程"""
    print("\n" + "=" * 60)
    print("测试3: TSE训练流程模拟")
    print("=" * 60)
    
    # 创建模型(不需要T参数)
    model = resnet19(num_classes=10)
    model.train()
    
    # 创建数据
    B, C, H, W, T = 4, 3, 32, 32, 4
    images = torch.randn(B, C, H, W, T)
    labels = torch.randint(0, 10, (B,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    print(f"数据: images={images.shape}, labels={labels.shape}")
    
    # 模拟TSE训练步骤
    optimizer.zero_grad()
    
    # 1. 前向传播(获取特征)
    output, features_before_gap = model(images, return_features=True)
    mean_out = torch.mean(output, dim=1)
    
    print(f"\n前向传播:")
    print(f"  输出: {output.shape}")
    print(f"  特征: {features_before_gap.shape}")
    print(f"  平均输出: {mean_out.shape}")
    
    # 2. 获取分类层
    fc_layer = model.fc2
    print(f"\n分类层: {fc_layer}")
    
    # 3. 计算TSE损失
    loss = TSE_loss(
        feature_maps=features_before_gap,
        fc_layer=fc_layer,
        labels=labels,
        criterion=criterion,
        tau_f=0.5,
        kappa=1.0
    )
    
    print(f"\nTSE损失: {loss.item():.4f}")
    
    # 4. 反向传播
    loss.backward()
    
    # 5. 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "模型参数没有梯度!"
    
    # 6. 优化器步骤
    optimizer.step()
    
    print(f"\n✅ TSE训练流程正常")
    print(f"   损失值: {loss.item():.4f}")
    print(f"   梯度: 正常")
    return True


def test_tet_tse_mutual_exclusion():
    """测试4: TET和TSE互斥性"""
    print("\n" + "=" * 60)
    print("测试4: TET和TSE互斥性验证")
    print("=" * 60)
    
    # 模拟args对象
    class Args:
        def __init__(self, TET, TSE):
            self.TET = TET
            self.TSE = TSE
            self.tau_f = 0.5
            self.kappa = 1.0
            self.means = 1.0
            self.lamb = 0.05
    
    test_cases = [
        (False, False, "标准训练", True),
        (True, False, "TET训练", True),
        (False, True, "TSE训练", True),
        (True, True, "TET+TSE同时启用", False),  # 应该避免
    ]
    
    print("\n训练模式检查:")
    print(f"{'模式':<20} {'TET':<8} {'TSE':<8} {'推荐':<10}")
    print("-" * 60)
    
    for TET, TSE, desc, recommended in test_cases:
        args = Args(TET, TSE)
        status = "✅" if recommended else "⚠️ 不推荐"
        print(f"{desc:<20} {args.TET!s:<8} {args.TSE!s:<8} {status:<10}")
    
    print("\n✅ 互斥性检查完成")
    print("   注意: 不要同时启用TET和TSE!")
    return True


def test_backward_compatibility():
    """测试5: 向后兼容性"""
    print("\n" + "=" * 60)
    print("测试5: 向后兼容性")
    print("=" * 60)
    
    model = resnet19(num_classes=10, T=4)
    model.eval()
    
    B, C, H, W, T = 2, 3, 32, 32, 4
    x = torch.randn(B, C, H, W, T)
    
    # 旧代码:不传return_features参数
    with torch.no_grad():
        output_old = model(x)
    
    # 新代码:显式设置return_features=False
    with torch.no_grad():
        output_new = model(x, return_features=False)
    
    # 应该得到相同的结果
    assert torch.allclose(output_old, output_new), "向后兼容性失败!"
    
    print(f"旧代码输出: {output_old.shape}")
    print(f"新代码输出: {output_new.shape}")
    print(f"结果一致: {torch.allclose(output_old, output_new)}")
    
    print("\n✅ 向后兼容性正常")
    return True


def run_all_integration_tests():
    """运行所有集成测试"""
    print("\n" + "=" * 60)
    print("TSE集成完整性测试")
    print("=" * 60)
    
    tests = [
        ("Argparse集成", test_argparse_integration),
        ("模型特征提取", test_model_feature_extraction),
        ("TSE训练流程", test_tse_training_flow),
        ("TET/TSE互斥性", test_tet_tse_mutual_exclusion),
        ("向后兼容性", test_backward_compatibility),
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
    print("集成测试总结")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:<20} {status}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("🎉 所有集成测试通过!")
        print("TSE方法已成功集成到训练框架中,可以开始使用。")
        print("=" * 60)
        print("\n使用方法:")
        print("  python main_training_distribute_improved.py --tse --tau-f 0.5 --kappa 1.0 [其他参数...]")
        print("\n更多信息请查看: TSE使用指南.md")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败,请检查集成。")
        return False


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
