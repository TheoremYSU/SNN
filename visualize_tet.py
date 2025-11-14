"""
TET (Temporal Efficient Training) 可视化演示
展示TET_loss、ZIF替代梯度、LIF动力学等核心概念
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_tet_loss_comparison():
    """对比传统loss和TET loss"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    T = 4
    num_classes = 10
    x = np.arange(T)
    
    # 模拟logits输出
    np.random.seed(42)
    logits = np.random.randn(T, num_classes) * 0.5 + np.array([0, 1, 2, 3])[:, None]
    
    # 1. 传统方法: 只用最后时间步
    ax = axes[0]
    for c in range(3):
        ax.plot(x, logits[:, c], 'o-', label=f'Class {c}', alpha=0.7)
    ax.axvline(x=T-1, color='red', linestyle='--', linewidth=2, label='Only use T-1')
    ax.set_xlabel('时间步 t', fontsize=12)
    ax.set_ylabel('Logits', fontsize=12)
    ax.set_title('传统方法\n只利用最后时间步', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.95, '信息利用率: 1/T = 25%', 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 2. TET方法: 使用所有时间步
    ax = axes[1]
    for c in range(3):
        ax.plot(x, logits[:, c], 'o-', label=f'Class {c}', alpha=0.7)
    for t in range(T):
        ax.axvline(x=t, color='green', linestyle='--', alpha=0.3)
    ax.set_xlabel('时间步 t', fontsize=12)
    ax.set_ylabel('Logits', fontsize=12)
    ax.set_title('TET方法\n利用所有时间步', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.95, '信息利用率: 100%', 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 3. Loss公式对比
    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # 传统loss
    y_pos = 8
    ax.text(5, y_pos, '传统Loss', ha='center', fontsize=14, fontweight='bold', color='red')
    y_pos -= 1
    ax.text(5, y_pos, r'$L = CE(output_{T-1}, label)$', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#FFE5E5'))
    
    y_pos -= 2
    ax.text(5, y_pos, 'TET Loss', ha='center', fontsize=14, fontweight='bold', color='green')
    y_pos -= 1
    ax.text(5, y_pos, r'$L_{TET} = \frac{1}{T} \sum_{t=0}^{T-1} CE(output_t, label)$', 
           ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#E5FFE5'))
    y_pos -= 1.5
    ax.text(5, y_pos, r'$L_{MSE} = MSE(outputs, means)$', 
           ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#E5F5FF'))
    y_pos -= 1.5
    ax.text(5, y_pos, r'$L_{Total} = (1-\lambda) \cdot L_{TET} + \lambda \cdot L_{MSE}$', 
           ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFFACD'))
    
    plt.tight_layout()
    return fig

def plot_zif_surrogate():
    """可视化ZIF替代梯度"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.linspace(-3, 3, 1000)
    
    # 1. 前向: 阶跃函数
    ax = axes[0]
    y_heaviside = (x >= 0).astype(float)
    ax.plot(x, y_heaviside, 'b-', linewidth=2, label='Heaviside函数')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('膜电位 V', fontsize=12)
    ax.set_ylabel('脉冲输出', fontsize=12)
    ax.set_title('前向传播\n阶跃函数', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.2, 1.2)
    
    # 2. 反向: ZIF替代梯度
    ax = axes[1]
    gama = 1.0
    zif_grad = (1/gama)**2 * np.maximum(0, gama - np.abs(x))
    
    ax.plot(x, zif_grad, 'r-', linewidth=2, label=f'ZIF (γ={gama})')
    ax.fill_between(x, 0, zif_grad, alpha=0.3, color='red')
    ax.axvline(x=-gama, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=gama, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('膜电位 V', fontsize=12)
    ax.set_ylabel('梯度', fontsize=12)
    ax.set_title('反向传播\nZIF替代梯度 (三角形)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.text(0, 0.8, r'$\frac{\partial L}{\partial V} = \frac{1}{\gamma^2} \max(0, \gamma - |V|)$',
           ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 3. 对比不同替代梯度
    ax = axes[2]
    # Sigmoid
    sigmoid_grad = 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
    # ATan
    alpha = 2.0
    atan_grad = alpha / (1 + (alpha * x)**2)
    # ZIF
    zif_grad = (1/gama)**2 * np.maximum(0, gama - np.abs(x))
    
    ax.plot(x, sigmoid_grad, 'g-', linewidth=2, label='Sigmoid', alpha=0.7)
    ax.plot(x, atan_grad, 'b-', linewidth=2, label='ATan', alpha=0.7)
    ax.plot(x, zif_grad, 'r-', linewidth=2, label='ZIF (TET)', alpha=0.7)
    ax.set_xlabel('输入', fontsize=12)
    ax.set_ylabel('梯度', fontsize=12)
    ax.set_title('替代梯度对比', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_lif_dynamics():
    """可视化LIF神经元动力学"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 参数
    T = 50
    tau = 0.5
    thresh = 1.0
    dt = 1.0
    
    # 模拟LIF动力学
    t = np.arange(T)
    
    # 1. 恒定输入
    ax = axes[0, 0]
    input_current = np.ones(T) * 0.3
    mem = np.zeros(T)
    spikes = np.zeros(T)
    
    for i in range(1, T):
        mem[i] = mem[i-1] * tau + input_current[i]
        if mem[i] >= thresh:
            spikes[i] = 1
            mem[i] = 0  # hard reset
    
    ax.plot(t, input_current, 'b-', label='输入电流', linewidth=2)
    ax.plot(t, mem, 'g-', label='膜电位', linewidth=2)
    ax.stem(t, spikes, linefmt='r-', markerfmt='ro', basefmt=' ', label='脉冲')
    ax.axhline(y=thresh, color='orange', linestyle='--', label='阈值')
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('值', fontsize=11)
    ax.set_title('LIF动力学: 恒定输入', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. 脉冲输入
    ax = axes[0, 1]
    input_current = np.zeros(T)
    input_current[[5, 15, 25, 35, 45]] = 0.8
    mem = np.zeros(T)
    spikes = np.zeros(T)
    
    for i in range(1, T):
        mem[i] = mem[i-1] * tau + input_current[i]
        if mem[i] >= thresh:
            spikes[i] = 1
            mem[i] = 0
    
    ax.stem(t, input_current, linefmt='b-', markerfmt='bo', basefmt=' ', label='输入脉冲')
    ax.plot(t, mem, 'g-', label='膜电位', linewidth=2)
    ax.stem(t, spikes, linefmt='r-', markerfmt='ro', basefmt=' ', label='输出脉冲')
    ax.axhline(y=thresh, color='orange', linestyle='--', label='阈值')
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('值', fontsize=11)
    ax.set_title('LIF动力学: 脉冲输入', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. 不同tau的影响
    ax = axes[1, 0]
    input_current = np.ones(T) * 0.3
    taus = [0.3, 0.5, 0.7, 0.9]
    colors = ['blue', 'green', 'orange', 'red']
    
    for tau, color in zip(taus, colors):
        mem = np.zeros(T)
        for i in range(1, T):
            mem[i] = mem[i-1] * tau + input_current[i]
            if mem[i] >= thresh:
                mem[i] = 0
        ax.plot(t, mem, color=color, label=f'τ={tau}', linewidth=2, alpha=0.7)
    
    ax.axhline(y=thresh, color='black', linestyle='--', label='阈值')
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('膜电位', fontsize=11)
    ax.set_title('不同衰减系数τ的影响', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. LIF公式说明
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    y_pos = 9
    ax.text(5, y_pos, 'LIF神经元模型', ha='center', fontsize=16, fontweight='bold')
    
    y_pos -= 1.5
    ax.text(5, y_pos, '1. 膜电位更新', ha='center', fontsize=13, fontweight='bold', color='green')
    y_pos -= 1
    ax.text(5, y_pos, r'$V[t] = \tau \cdot V[t-1] + I[t]$', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#E5FFE5'))
    
    y_pos -= 1.5
    ax.text(5, y_pos, '2. 脉冲发放', ha='center', fontsize=13, fontweight='bold', color='red')
    y_pos -= 1
    ax.text(5, y_pos, r'$spike = \mathbb{1}(V[t] \geq V_{thresh})$', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#FFE5E5'))
    
    y_pos -= 1.5
    ax.text(5, y_pos, '3. 膜电位重置', ha='center', fontsize=13, fontweight='bold', color='blue')
    y_pos -= 1
    ax.text(5, y_pos, r'$V[t] = (1 - spike) \cdot V[t]$', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#E5F5FF'))
    y_pos -= 0.7
    ax.text(5, y_pos, '(soft reset)', ha='center', fontsize=10, style='italic')
    
    y_pos -= 1.2
    ax.text(5, y_pos, '参数说明', ha='center', fontsize=11, fontweight='bold')
    y_pos -= 0.5
    ax.text(5, y_pos, r'$\tau$: 膜电位衰减系数 (0~1)', ha='center', fontsize=10)
    y_pos -= 0.5
    ax.text(5, y_pos, r'$V_{thresh}$: 发放阈值 (默认1.0)', ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_training_flow():
    """可视化TET训练流程"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',
        'forward': '#E8F5E9',
        'loss': '#FFF3E0',
        'backward': '#FCE4EC',
        'update': '#F3E5F5'
    }
    
    y_pos = 19
    
    # 标题
    ax.text(5, y_pos, 'TET训练流程', ha='center', fontsize=18, fontweight='bold')
    y_pos -= 1.5
    
    # 1. 输入
    ax.add_patch(mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.7, 
                 boxstyle="round,pad=0.1", facecolor=colors['input'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.45, '1. 输入图像 [N, C, H, W]', ha='center', va='center', fontsize=11, fontweight='bold')
    y_pos -= 1.5
    ax.annotate('', xy=(5, y_pos+0.3), xytext=(5, y_pos+0.7), 
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 2. 时间维度展开
    ax.add_patch(mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.7,
                 boxstyle="round,pad=0.1", facecolor=colors['forward'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.45, '2. 时间维度展开 [N, T, C, H, W]', ha='center', va='center', fontsize=11, fontweight='bold')
    y_pos -= 1.5
    ax.annotate('', xy=(5, y_pos+0.3), xytext=(5, y_pos+0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 3. 前向传播
    ax.add_patch(mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.7,
                 boxstyle="round,pad=0.1", facecolor=colors['forward'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.45, '3. 前向传播: Conv→LIF→ResBlock→...', ha='center', va='center', fontsize=11, fontweight='bold')
    y_pos -= 1.5
    ax.annotate('', xy=(5, y_pos+0.3), xytext=(5, y_pos+0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 4. 输出
    ax.add_patch(mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.7,
                 boxstyle="round,pad=0.1", facecolor=colors['forward'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.45, '4. 输出logits [N, T, num_classes]', ha='center', va='center', fontsize=11, fontweight='bold')
    y_pos -= 1.5
    ax.annotate('', xy=(5, y_pos+0.3), xytext=(5, y_pos+0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 5. TET Loss计算
    ax.add_patch(mpatches.FancyBboxPatch((1, y_pos-1.2), 8, 1.1,
                 boxstyle="round,pad=0.1", facecolor=colors['loss'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.3, '5. TET Loss计算', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos-0.7, r'$L_{TET} = \frac{1}{T}\sum_t CE(output_t, label)$', ha='center', fontsize=10)
    ax.text(5, y_pos-1.0, r'$L_{Total} = (1-\lambda)L_{TET} + \lambda L_{MSE}$', ha='center', fontsize=10)
    y_pos -= 2.0
    ax.annotate('', xy=(5, y_pos+0.3), xytext=(5, y_pos+0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 6. 反向传播
    ax.add_patch(mpatches.FancyBboxPatch((1, y_pos-1.2), 8, 1.1,
                 boxstyle="round,pad=0.1", facecolor=colors['backward'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.3, '6. 反向传播 (ZIF替代梯度)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos-0.7, r'梯度通过ZIF: $\frac{\partial L}{\partial V} = \frac{1}{\gamma^2}\max(0, \gamma-|V|)$', 
           ha='center', fontsize=9)
    ax.text(5, y_pos-1.0, '时间维度完整展开,每个时间步都有梯度', ha='center', fontsize=9)
    y_pos -= 2.0
    ax.annotate('', xy=(5, y_pos+0.3), xytext=(5, y_pos+0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 7. 参数更新
    ax.add_patch(mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.7,
                 boxstyle="round,pad=0.1", facecolor=colors['update'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.45, '7. 参数更新: optimizer.step()', ha='center', va='center', fontsize=11, fontweight='bold')
    y_pos -= 1.5
    ax.annotate('', xy=(5, y_pos+0.3), xytext=(5, y_pos+0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 8. 膜电位重置
    ax.add_patch(mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.7,
                 boxstyle="round,pad=0.1", facecolor='#FFE5E5', edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.45, '8. 重置神经元状态 (准备下一batch)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 添加关键特性标注
    ax.text(9.5, 15, '关键特性:', fontsize=12, fontweight='bold', ha='right')
    ax.text(9.5, 14.3, '✓ 所有时间步参与训练', fontsize=10, ha='right', color='green')
    ax.text(9.5, 13.7, '✓ MSE正则化稳定训练', fontsize=10, ha='right', color='green')
    ax.text(9.5, 13.1, '✓ ZIF替代梯度', fontsize=10, ha='right', color='green')
    ax.text(9.5, 12.5, '✓ Soft reset膜电位', fontsize=10, ha='right', color='green')
    
    return fig

def plot_accuracy_calculation():
    """对比TET和SEW的准确率计算方式"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 模拟数据
    T = 4
    N = 2
    C = 3
    np.random.seed(42)
    logits_tet = np.random.randn(N, T, C) * 2 + np.array([[[1, 0, -1], [2, 1, 0], [3, 2, 1], [4, 3, 2]]])
    logits_sew = logits_tet.transpose(1, 0, 2)  # [T, N, C]
    
    # 1. TET方法
    ax = axes[0]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    y_pos = 9.5
    ax.text(5, y_pos, 'TET准确率计算', ha='center', fontsize=16, fontweight='bold', color='blue')
    
    y_pos -= 1
    ax.text(5, y_pos, '1. 输出格式: [N, T, C]', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#E5F5FF'))
    
    y_pos -= 1.5
    ax.text(5, y_pos, '2. 先平均logits (时间维度)', ha='center', fontsize=12, fontweight='bold')
    y_pos -= 0.7
    ax.text(5, y_pos, r'$mean\_out = \frac{1}{T} \sum_{t=0}^{T-1} output[:, t, :]$', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='#FFFACD'))
    y_pos -= 0.7
    ax.text(5, y_pos, 'mean_out.shape = [N, C]', ha='center', fontsize=10, style='italic')
    
    y_pos -= 1.5
    ax.text(5, y_pos, '3. 计算准确率', ha='center', fontsize=12, fontweight='bold')
    y_pos -= 0.7
    ax.text(5, y_pos, r'acc = accuracy(mean_out, labels)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='#E5FFE5'))
    
    y_pos -= 1.5
    ax.text(5, y_pos, '特点:', ha='center', fontsize=12, fontweight='bold')
    y_pos -= 0.6
    ax.text(5, y_pos, '✓ 先融合时间信息', ha='center', fontsize=10, color='green')
    y_pos -= 0.5
    ax.text(5, y_pos, '✓ 计算速度快', ha='center', fontsize=10, color='green')
    y_pos -= 0.5
    ax.text(5, y_pos, '✓ 与TET_loss一致', ha='center', fontsize=10, color='green')
    
    # 可视化示例
    y_pos -= 1
    example_logits = logits_tet[0]  # [T, C]
    mean_logits = example_logits.mean(axis=0)
    
    ax.text(5, y_pos, '示例计算:', ha='center', fontsize=10, fontweight='bold')
    y_pos -= 0.5
    t_pos = np.linspace(1, 9, T)
    for i, t in enumerate(range(T)):
        ax.text(t_pos[i], y_pos-0.3, f't={t}\n{example_logits[t,0]:.1f}', ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    y_pos -= 0.8
    ax.text(5, y_pos, '↓ 平均 ↓', ha='center', fontsize=10, fontweight='bold')
    y_pos -= 0.5
    ax.text(5, y_pos, f'mean = {mean_logits[0]:.2f}', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow'))
    
    # 2. SEW方法
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    y_pos = 9.5
    ax.text(5, y_pos, 'SEW准确率计算', ha='center', fontsize=16, fontweight='bold', color='red')
    
    y_pos -= 1
    ax.text(5, y_pos, '1. 输出格式: [T, N, C]', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#FFE5E5'))
    
    y_pos -= 1.5
    ax.text(5, y_pos, '2. 先softmax每个时间步', ha='center', fontsize=12, fontweight='bold')
    y_pos -= 0.7
    ax.text(5, y_pos, r'$prob_t = softmax(output[t, :, :])$', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='#FFFACD'))
    y_pos -= 0.7
    ax.text(5, y_pos, 'prob.shape = [T, N, C]', ha='center', fontsize=10, style='italic')
    
    y_pos -= 1.0
    ax.text(5, y_pos, '3. 再平均概率 (时间维度)', ha='center', fontsize=12, fontweight='bold')
    y_pos -= 0.7
    ax.text(5, y_pos, r'$mean\_prob = \frac{1}{T} \sum_{t=0}^{T-1} prob_t$', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='#FFFACD'))
    
    y_pos -= 1.0
    ax.text(5, y_pos, '4. 计算准确率', ha='center', fontsize=12, fontweight='bold')
    y_pos -= 0.7
    ax.text(5, y_pos, r'acc = accuracy(mean_prob, labels)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='#E5FFE5'))
    
    y_pos -= 1.5
    ax.text(5, y_pos, '特点:', ha='center', fontsize=12, fontweight='bold')
    y_pos -= 0.6
    ax.text(5, y_pos, '✓ 集成学习风格', ha='center', fontsize=10, color='green')
    y_pos -= 0.5
    ax.text(5, y_pos, '✓ 每个时间步独立预测', ha='center', fontsize=10, color='green')
    y_pos -= 0.5
    ax.text(5, y_pos, '✗ 计算稍慢 (需softmax)', ha='center', fontsize=10, color='orange')
    
    # 可视化示例
    y_pos -= 1
    example_logits_sew = logits_sew[:, 0, :]  # [T, C]
    # Softmax
    exp_logits = np.exp(example_logits_sew - example_logits_sew.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    mean_probs = probs.mean(axis=0)
    
    ax.text(5, y_pos, '示例计算:', ha='center', fontsize=10, fontweight='bold')
    y_pos -= 0.5
    t_pos = np.linspace(1, 9, T)
    for i, t in enumerate(range(T)):
        ax.text(t_pos[i], y_pos-0.3, f't={t}\n{probs[t,0]:.2f}', ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.5))
    y_pos -= 0.8
    ax.text(5, y_pos, '↓ 平均 ↓', ha='center', fontsize=10, fontweight='bold')
    y_pos -= 0.5
    ax.text(5, y_pos, f'mean = {mean_probs[0]:.2f}', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow'))
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("=" * 80)
    print("TET (Temporal Efficient Training) 可视化演示")
    print("=" * 80)
    
    print("\n1. 生成TET Loss对比图...")
    fig1 = plot_tet_loss_comparison()
    fig1.savefig('tet_loss_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ 保存: tet_loss_comparison.png")
    
    print("\n2. 生成ZIF替代梯度可视化...")
    fig2 = plot_zif_surrogate()
    fig2.savefig('zif_surrogate_gradient.png', dpi=300, bbox_inches='tight')
    print("   ✓ 保存: zif_surrogate_gradient.png")
    
    print("\n3. 生成LIF神经元动力学...")
    fig3 = plot_lif_dynamics()
    fig3.savefig('lif_neuron_dynamics.png', dpi=300, bbox_inches='tight')
    print("   ✓ 保存: lif_neuron_dynamics.png")
    
    print("\n4. 生成训练流程图...")
    fig4 = plot_training_flow()
    fig4.savefig('tet_training_flow.png', dpi=300, bbox_inches='tight')
    print("   ✓ 保存: tet_training_flow.png")
    
    print("\n5. 生成准确率计算对比...")
    fig5 = plot_accuracy_calculation()
    fig5.savefig('accuracy_calculation_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ 保存: accuracy_calculation_comparison.png")
    
    print("\n" + "=" * 80)
    print("所有可视化图表生成完成!")
    print("=" * 80)
    
    plt.show()
