import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total


def TSE_loss(feature_maps, fc_layers, labels, criterion, tau_f=0.5, kappa=1.0):
    """
    Temporal-Self-Erasing (TSE) supervision loss.
    
    根据论文 "Towards More Discriminative Feature Learning in SNNs with 
    Temporal-Self-Erasing Supervision" (AAAI 2025) 实现。
    
    核心思想:
    - 对每个时间步t>1,基于前t-1步的预测构建erasing mask
    - Mask抑制已激活区域,引导网络关注新区域
    - 每个时间步独立监督,增强时间维度特征判别性
    
    Args:
        feature_maps: [B, T, C, H, W] - T个时间步的特征图(GAP之前)
        fc_layers: 分类器层(可以是单层或多层Sequential)
                  - 单层: nn.Linear(C, num_classes)
                  - 多层: nn.Sequential(fc1, fc2, ...)
        labels: [B] - 真实标签
        criterion: 交叉熵损失函数
        tau_f: 固定阈值 (论文中的τ_f)
        kappa: 动态阈值的标准差倍数 (论文中的κ)
    
    Returns:
        loss: TSE总损失
    
    论文公式对应:
    - Eq.(7): 平均前t-1步预测并Softmax
    - Eq.(9): τ_d = mean(P) + κ·std(P)
    - Eq.(10): M_{i,j} = 0 if P_{i,j} >= max(τ_f, τ_d) else 1
    - Eq.(11): p_t = FC(GAP(M_t · F_t))
    - Eq.(12): L = L_CE(p_1, y) + Σ L_CE(p_t, y)
    """
    B, T, C, H, W = feature_maps.shape
    
    # 获取输出类别数
    if isinstance(fc_layers, nn.Sequential):
        # 多层FC,取最后一层的输出维度
        num_classes = list(fc_layers.children())[-1].out_features
    else:
        # 单层FC
        num_classes = fc_layers.out_features
    
    device = feature_maps.device
    
    # 初始化分类预测图列表
    # P_t shape: [B, num_classes, H, W]
    prediction_maps = []
    
    # Step 1: 对每个时间步,在每个空间位置上做分类预测
    # 论文: "we apply the fully connected layer to the features at each location (i,j)"
    for t in range(T):
        F_t = feature_maps[:, t]  # [B, C, H, W]
        # 重排为 [B, H, W, C] 以对每个位置应用FC
        F_t_reshaped = F_t.permute(0, 2, 3, 1)  # [B, H, W, C]
        # 应用FC层(支持单层或多层): [B, H, W, C] -> [B, H, W, num_classes]
        P_t = fc_layers(F_t_reshaped)  # [B, H, W, num_classes]
        # 转回 [B, num_classes, H, W]
        P_t = P_t.permute(0, 3, 1, 2)  # [B, num_classes, H, W]
        prediction_maps.append(P_t)
    
    # 初始化总损失
    total_loss = 0.0
    
    # Step 2: 对每个时间步计算损失
    for t in range(T):
        if t == 0:
            # 第一个时间步: 直接使用原始特征
            # 论文 Eq.(12): L_CE(p_1, y)
            F_t = feature_maps[:, t]  # [B, C, H, W]
            # GAP + FC
            pooled = torch.nn.functional.adaptive_avg_pool2d(F_t, (1, 1))  # [B, C, 1, 1]
            pooled = pooled.view(B, -1)  # [B, C]
            p_t = fc_layers(pooled)  # [B, num_classes]
            loss_t = criterion(p_t, labels)
        else:
            # t > 0: 使用TSE机制
            # Step 2.1: 平均**前t-1步**的预测图并Softmax (论文 Eq.7)
            # 重要: 是前t-1步(即索引0到t-1),不包括当前步t
            # 公式: P̄^{t-1} = Softmax(1/(t-1) * Σ_{k=1}^{t-1} P^k)
            # 注意: Python索引从0开始,所以t对应的是第t+1个元素
            # prediction_maps[:t]包含索引0到t-1,对应时间步1到t,共t个
            # 我们需要的是前t-1个,所以应该是prediction_maps[:t-1]? 不对!
            # 因为当前循环的t从0开始,所以prediction_maps[:t]已经是前t个了
            # 等等,让我重新理清楚:
            # - 外层循环: for t in range(T), 所以t=0,1,2,...,T-1
            # - 当t=0时,是第1个时间步
            # - 当t=1时,是第2个时间步,应该用前1步(t=0)的预测
            # - 当t=2时,是第3个时间步,应该用前2步(t=0,1)的预测
            # 所以prediction_maps[:t]正好对应前t步,这是对的!
            
            # P_avg shape: [B, num_classes, H, W]
            # 先求和再平均(虽然mean()直接做也行,但这样更清晰对应论文公式)
            if t == 1:
                # 第2个时间步,只有1个历史
                P_avg = prediction_maps[0]  # [B, num_classes, H, W]
            else:
                # 第3+个时间步,有多个历史
                P_sum = torch.stack(prediction_maps[:t], dim=1).sum(dim=1)  # [B, num_classes, H, W]
                P_avg = P_sum / t  # 平均
            
            P_avg_prob = torch.softmax(P_avg, dim=1)  # Softmax (沿class维度)
            
            # Step 2.2: 提取真实类别的概率图
            # P_{t-1}_y shape: [B, H, W]
            P_y = P_avg_prob[torch.arange(B), labels]  # [B, H, W]
            
            # Step 2.3: 计算动态阈值 (论文 Eq.9)
            # τ_d = mean(P) + κ·std(P)
            mean_val = P_y.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
            std_val = P_y.std(dim=(1, 2), keepdim=True)  # [B, 1, 1]
            tau_d = mean_val + kappa * std_val  # [B, 1, 1]
            
            # Step 2.4: 构建Erasing Mask (论文 Eq.10)
            # M_{i,j} = 0 if P_{i,j} >= max(τ_f, τ_d) else 1
            tau_threshold = torch.maximum(
                torch.tensor(tau_f, device=device), 
                tau_d
            )  # [B, 1, 1]
            M_t = (P_y < tau_threshold).float()  # [B, H, W]
            
            # 扩展mask到通道维度 [B, 1, H, W]
            M_t = M_t.unsqueeze(1)  # [B, 1, H, W]
            
            # Step 2.5: 应用mask调制特征图 (论文 Eq.11)
            F_t = feature_maps[:, t]  # [B, C, H, W]
            F_t_erased = F_t * M_t  # [B, C, H, W]
            
            # Step 2.6: GAP + FC 得到调制后的预测
            pooled = torch.nn.functional.adaptive_avg_pool2d(F_t_erased, (1, 1))
            pooled = pooled.view(B, -1)  # [B, C]
            p_t_tilde = fc_layers(pooled)  # [B, num_classes]
            
            # 计算损失
            loss_t = criterion(p_t_tilde, labels)
        
        # 累加损失 (论文 Eq.12)
        total_loss += loss_t
    
    return total_loss