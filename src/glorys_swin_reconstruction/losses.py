from __future__ import annotations
from torch.nn import functional as F
import torch

# 深度均衡的基础均方误差损失
# 传统 MSE 会将三维空间所有有效像素点混在一起求均值，这会导致模型过度拟合数据密集的浅层。该函数首先计算每一个独立深度层上的掩码均方误差，再对所有“有数据的深度层”求平均。
# 这意味着每个深度在最终总损失中的权重都是相等的，强迫模型同样重视深层水体的预测精度。
def depth_balanced_masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target and mask must have identical shapes.")
    mask_f = (mask > 0).to(pred.dtype)
    counts = mask_f.sum(dim=(0, 2, 3))
    # 计算每个深度层的独立均方误差 (MSE)
    per_depth = ((pred - target).square() * mask_f).sum(dim=(0, 2, 3)) / (counts + eps)
    valid_depths = counts > 0
    if not torch.any(valid_depths):
        return pred.sum() * 0.0
    return per_depth[valid_depths].mean()


# 约束模型恢复正确的相邻层物理温度梯度
def vertical_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    depths_m: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    reference_distance_m: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Depth-balanced physical temperature-gradient error."""
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    if depths_m.numel() != pred.shape[1]:
        raise ValueError("depths_m length must equal the output-depth dimension.")
    
    # 计算真实的深度间距 dz
    dz = torch.diff(depths_m).view(1, -1, 1, 1)
    if torch.any(dz <= 0):
        raise ValueError("Depth values must be strictly increasing.")
    valid = (mask[:, 1:] > 0) & (mask[:, :-1] > 0)
    if not torch.any(valid):
        return pred.sum() * 0.0
    
    # 将网络输出逆归一化，恢复真实物理温度
    pred_c = pred * norm_std + norm_mean
    target_c = target * norm_std + norm_mean
    scale = float(reference_distance_m)

    # 核心计算：预测梯度与真实梯度的均方误差
    # 为了防止“每米的温度变化率误差”过小在平方后导致梯度消失，这里乘以了 scale，将误差放缩为“每百米的温度梯度差值”。
    error2 = (((pred_c[:, 1:] - pred_c[:, :-1]) / dz -(target_c[:, 1:] - target_c[:, :-1]) / dz) * scale).square()
    mask_f = valid.to(error2.dtype)
    counts = mask_f.sum(dim=(0, 2, 3))
    per_gap = (error2 * mask_f).sum(dim=(0, 2, 3)) / (counts + eps)
    return per_gap[counts > 0].mean()


# 约束强梯度的深度位置
# 该损失只关注温度随深度下降的部分（clamp_min(0.0)），忽略逆温层
# 利用 1D Wasserstein-1 距离，衡量预测温跃层和真实温跃层在深度方向上的平移距离。
def vertical_gradient_transport_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    depths_m: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    # ---------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(
            "pred, target and mask must have identical shapes."
        )
    if pred.ndim != 4:
        raise ValueError(
            "pred, target and mask must have shape [B, D, H, W]."
        )
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    dtype = torch.float32
    device = pred.device
  
    depths = depths_m.to(
        device=device,
        dtype=dtype,
    )
    if depths.numel() != pred.shape[1]:
        raise ValueError(
            "depths_m length must equal output-depth dimension."
        )
    # 构造实际深度间隔
    dz = torch.diff(depths)
    if torch.any(dz <= 0):
        raise ValueError(
            "Depth values must be strictly increasing."
        )
    dz4 = dz.view(1, -1, 1, 1)

    # ---------------------------------------------------------
    # 确定第 i 层和第 i+1 层都存在有效梯度
    # ---------------------------------------------------------
    valid = (
        (mask[:, 1:] > 0)
        & (mask[:, :-1] > 0)
    )
    if not torch.any(valid):
        return pred.sum() * 0.0
    valid_f = valid.to(dtype)

    # ---------------------------------------------------------
    # 反标准化为物理温度
    # ---------------------------------------------------------
    mean = norm_mean.to(
        device=device,
        dtype=dtype,
    )
    std = norm_std.to(
        device=device,
        dtype=dtype,
    )
    pred_c = (
        pred.to(dtype) * std + mean
    )
    target_c = (
        target.to(dtype) * std + mean
    )

    # ---------------------------------------------------------
    # 定义“向下冷却量”，max(T_i-T_{i+1},0)。代表预测剖面中，第 i 层到第 i+1 层之间，温度向下下降了多少摄氏度。
    # 逆温层本身不会被这个 loss 直接约束，需要依靠主回归损失和 signed vertical-gradient loss。
    # ---------------------------------------------------------
    raw_drop_pred = (
        pred_c[:, :-1]
        - pred_c[:, 1:]
    ).clamp_min(0.0)

    raw_drop_true = (
        target_c[:, :-1]
        - target_c[:, 1:]
    ).clamp_min(0.0)

    drop_pred = torch.where(
        valid,
        raw_drop_pred,
        torch.zeros_like(raw_drop_pred),
    )

    drop_true = torch.where(
        valid,
        raw_drop_true,
        torch.zeros_like(raw_drop_true),
    )

    # ---------------------------------------------------------
    # 计算整条水柱一共降了多少 ℃
    # ---------------------------------------------------------
    total_pred = drop_pred.sum(
        dim=1,
        keepdim=True,
    )

    total_true = drop_true.sum(
        dim=1,
        keepdim=True,
    )

    valid_count = valid_f.sum(
        dim=1,
        keepdim=True,
    )

    # 将“温度下降量”转化为概率密度函数（PDF），进而求累计概率分布（CDF）
    '''
    drop_pred =[1, 1, 4, 2]

    ↓ 除以总冷却量 8

    p_pred=[0.125, 0.125, 0.500, 0.250]

    ↓ 累计

    cdf_pred =[0.125, 0.250, 0.750, 1.000]

    '''
    uniform = (valid_f / valid_count.clamp_min(1.0))

    p_pred_normal = (drop_pred/ total_pred.clamp_min(eps))

    p_true_normal = (drop_true/ total_true.clamp_min(eps))

    p_pred = torch.where(
        total_pred > eps,
        p_pred_normal,
        uniform,
    )

    p_true = torch.where(
        total_true > eps,
        p_true_normal,
        uniform,
    )

    # 计算CDF，从海面向下走，到当前深度为止，整条水柱的总冷却已经发生了多少比例
    cdf_pred = torch.cumsum( p_pred,dim=1,)

    cdf_true = torch.cumsum(p_true,dim=1,)


    # -------------------------------------------------------------------------------
    # 核心设计：计算一维 Wasserstein-1 距离，用于衡量预测温跃结构与真实温跃结构在深度轴上的错位距离
    # -------------------------------------------------------------------------------
    
    z_mid = 0.5 * (depths[:-1] + depths[1:])       # 每个 cooling interval 放在哪个深度位置（区间中点）

    dz_mid = torch.diff(z_mid).view(1, -1, 1, 1)   # 相邻 cooling bin 之间到底距离多远

    edge_valid = (valid[:, :-1]& valid[:, 1:])

    edge_valid_f = edge_valid.to(dtype)

    # CDF 差，表示在这个深度位置上，预测的冷却结构相对于真实结构存在多少m的误差
    displacement_m = ((cdf_pred[:, :-1] - cdf_true[:, :-1]).abs()* dz_mid* edge_valid_f).sum(dim=1,keepdim=True,)

    # ---------------------------------------------------------
    # Parameter-free fixed normalization.
    #
    # Unlike per-profile observed-span normalization,
    # the same physical displacement receives the same
    # scale for every water column.
    # ---------------------------------------------------------
    global_span_m = (
        z_mid[-1] - z_mid[0]
    ).clamp_min(eps)

    displacement = (
        displacement_m
        / global_span_m
    )

    # ---------------------------------------------------------
    # Parameter-free target structural confidence
    # ---------------------------------------------------------

    # Physical target cooling gradient, °C/m.
    cooling_grad_true = (
        drop_true
        / dz4
    )

    peak_grad_true = cooling_grad_true.amax(
        dim=1,
        keepdim=True,
    )

    # Total physically observed vertical span represented
    # by valid temperature-gradient intervals.
    valid_depth_span = (
        dz4 * valid_f
    ).sum(
        dim=1,
        keepdim=True,
    )

    # Mean cooling gradient through the observed water column.
    mean_grad_true = (
        total_true
        / valid_depth_span.clamp_min(eps)
    )

    # Structural sharpness:
    #
    # uniform gradual cooling -> ~0
    # highly concentrated cooling -> close to 1
    # 若整条水柱没有一个特别集中的温跃结构，则没有必要强制模型去做“温跃层位置调整”
    sharpness = (
        1.0
        - mean_grad_true
        / peak_grad_true.clamp_min(eps)
    ).clamp(
        min=0.0,
        max=1.0,
    )

    # Combine:
    #
    # amplitude: how much cooling exists?
    # sharpness: how concentrated is that cooling?
    #
    # Units disappear in the final normalized weighted mean.
    confidence = (
        total_true
        * sharpness
    )

    # Need at least two adjacent valid gradient bins to
    # define a transport distance.
    active = (
        edge_valid_f.sum(
            dim=1,
            keepdim=True,
        ) > 0
    ).to(dtype)

    weights = (
        confidence
        * active
    )

    return (
        (displacement * weights).sum()
        /
        weights.sum().clamp_min(eps)
    )