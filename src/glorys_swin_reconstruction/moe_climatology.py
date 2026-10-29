"""
输入数据集 (Dataset & Norm)
       │
       ▼
 1. 尺度检查与网格下采样空间划分 (scale)
       │
       ▼
 2. 逐时间步累加（反归一化物理量 + 空间下采样求和）
       │
       ▼
 3. 计算 12 个月的粗网格的温度气候态 (Climatology)
       │
       ▼
 4. 计算垂直温度梯度 (Vertical Gradient)
       │
       ▼
 5. 计算海洋混合层深度
       │
       ▼
 输出供 MoE 路由使用的温度先验统计量

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .time_utils import month_numbers


def fit_moe_statistics(dataset, norm, surface_norm_path: str, scale: int) -> dict:

    # 读取表层变量的归一化统计量
    data = json.loads(Path(surface_norm_path).read_text(encoding="utf-8"))["variables"]
    variables = ("sst", "sss", "sla", "ussw", "vssw")
    surface_mean = np.asarray([data[name]["mean"] for name in variables], np.float32)
    surface_std = np.asarray([data[name]["std"] for name in variables], np.float32)

    depths = dataset.depth_values.astype(np.float32)
    height, width = dataset.label.shape[-2:]
    if height % scale or width % scale:
        raise ValueError("MoE climatology grid must be divisible by patch size.")
    low_h, low_w = height // scale, width // scale
    shape = (12, len(depths), low_h, low_w)
    total = np.zeros(shape, np.float64)
    count = np.zeros(shape, np.float64)

    # 遍历数据集中的每一个时间步，将高分辨率的温度数据降采样到粗网格上
    months = month_numbers(dataset._surface_ds, dataset.spec.time_text_var)
    mean, std = norm.numpy_mean_std()
    for time_index in dataset.time_indices:
        index = int(time_index)
        month = int(months[index]) - 1
        target = np.asarray(dataset.label.isel(time=index).values, np.float32)
        mask = np.asarray(dataset.mask.isel(time=index).values) > 0
        valid = mask & (dataset.domain_mask_values > 0) & np.isfinite(target)
        physical = np.where(valid, target * std + mean, 0)
        total[month] += physical.reshape(len(depths), low_h, scale, low_w, scale).sum((2, 4))
        count[month] += valid.reshape(len(depths), low_h, scale, low_w, scale).sum((2, 4))

    # 计算月度平均温度（气候态）
    climatology = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    # 将气候态沿深度方向求导，得到某层的垂直温度梯度（相邻深度层间的温度变化）
    gradient = np.maximum(0, -np.gradient(climatology, depths, axis=1))
    gradient *= count > 0
    if len(depths) > 2:
        gradient[:, 1:-1] *= (count[:, :-2] > 0) & (count[:, 2:] > 0)
    gradient[:, 0] *= count[:, 1] > 0
    gradient[:, -1] *= count[:, -2] > 0

    # 计算混合层深度
    # 经典的混合层判别准则：对于每个月、每个地理位置，从海洋最浅层开始向下寻找，找到第一个与表层温差达到 0.2°C 的深度，并把该深度作为混合层深度
    crossing = (np.abs(climatology - climatology[:, :1]) >= 0.2) & (count > 0)
    first = crossing.argmax(axis=1)
    has_crossing = crossing.any(axis=1)
    iso_depth = np.where(has_crossing, depths[first], 50.0).astype(np.float32)

    return {
        "surface_mean": surface_mean,
        "surface_std": surface_std,
        "depth_m": depths,
        "temperature_mean": norm.mean,
        "temperature_std": norm.std,
        "clim_gradient": gradient.astype(np.float32),
        "isothermal_depth": iso_depth[:, None],
    }

