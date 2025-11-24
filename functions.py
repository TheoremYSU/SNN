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


def TSE_loss(feature_maps, fc_layer, labels, criterion, tau_f=0.5, kappa=1.0):
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
        fc_layer: 全连接分类层 (input: C, output: num_classes)
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
    num_classes = fc_layer.out_features
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
        # 应用FC层: [B, H, W, C] -> [B, H, W, num_classes]
        P_t = fc_layer(F_t_reshaped)  # [B, H, W, num_classes]
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
            p_t = fc_layer(pooled)  # [B, num_classes]
            loss_t = criterion(p_t, labels)
        else:
            # t > 0: 使用TSE机制
            # Step 2.1: 平均前t步的预测图并Softmax (论文 Eq.7)
            # P_avg shape: [B, num_classes, H, W]
            P_avg = torch.stack(prediction_maps[:t], dim=1).mean(dim=1)  # [B, num_classes, H, W]
            P_avg_prob = torch.softmax(P_avg, dim=1)  # Softmax along class dimension
            
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
            p_t_tilde = fc_layer(pooled)  # [B, num_classes]
            
            # 计算损失
            loss_t = criterion(p_t_tilde, labels)
        
        # 累加损失 (论文 Eq.12)
        total_loss += loss_t
    
    return total_loss