'''   双层路由、四类物理过程专家、残差修正式 MoE  '''

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .front import AnisotropicFront


def geographic_conv(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    x = F.pad(x, (1, 1, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, 1, 1), mode="replicate")
    return conv(x)

def geographic_pool(x: torch.Tensor) -> torch.Tensor:
    x = F.pad(x, (2, 2, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, 2, 2), mode="replicate")
    return F.avg_pool2d(x, 5, stride=1)

# 计算海表水平梯度
def gradient_100km(field: torch.Tensor, valid: torch.Tensor, latitude_sin: torch.Tensor) -> torch.Tensor:
    """Masked central gradient; longitude is periodic, latitude is not."""
    step_y = 111.2 * 0.25
    step_x = (step_y * torch.sqrt((1 - latitude_sin.square()).clamp_min(0))).clamp_min(1)
    east, west = field.roll(-1, -1), field.roll(1, -1)
    north = F.pad(field[..., :-1, :], (0, 0, 1, 0), mode="replicate")
    south = F.pad(field[..., 1:, :], (0, 0, 0, 1), mode="replicate")

    x_valid = valid * valid.roll(-1, -1) * valid.roll(1, -1)
    y_valid = valid * F.pad(valid[..., :-1, :], (0, 0, 1, 0), mode="replicate")
    y_valid = y_valid * F.pad(valid[..., 1:, :], (0, 0, 0, 1), mode="replicate")
    gx = (east - west) * (50 / step_x) * x_valid
    gy = (south - north) * (50 / step_y) * y_valid
    return torch.asinh((torch.sqrt(gx.square() + gy.square() + 1e-12) - 1e-6).clamp_min(0))


class ProcessMoE(nn.Module):
    """ 四个观测变量组路由器 and 四个按物理过程设计的残差专家"""

    def __init__(self, channels: int, depths: int, low_size: tuple[int, int],
                 scale: int, width: int = 16, prior_strength: float = 0.5,
                 local_gradient_weight: float = 0.0,
                 front: dict | None = None) -> None:
        super().__init__()
        if width <= 0 or prior_strength < 0:
            raise ValueError("Invalid MoE width or prior strength.")
        if not 0.0 <= local_gradient_weight <= 1.0:
            raise ValueError("local_gradient_weight must be between 0 and 1.")
        self.depth_count, self.low_size, self.scale = depths, low_size, scale
        self.prior_strength = prior_strength
        self.local_gradient_weight = local_gradient_weight
        self.shared = nn.Conv2d(channels, width, 1)
        self.groups = nn.ModuleList(nn.Conv2d(n, width, scale, stride=scale)
                                    for n in (3, 3, 3, 4))
        self.geo = nn.Conv2d(5, width, scale, stride=scale)
        self.variable_router = nn.ModuleList(nn.Conv2d(3 * width + 1, 1, 1)
                                             for _ in range(4))

        # 物理过程路由
        self.process_router = nn.Conv2d(2 * width + 1, 4, 1)

        ''' 4个物理过程专家:Mixed、Heave、Front、Deep  '''
        # Mixed Expert（混合层专家）
        self.mixed = nn.Conv2d(width, 5, 1)

        # Heave Expert（等温面位移专家）
        self.heave = nn.Conv2d(width, 1, 1)
        self.heave_shape = nn.Conv2d(width, 1, 1)

        # Front Expert（水平锋面专家）
        self.front_conv = nn.Conv2d(width, width, 3)
        self.front_out = nn.Conv2d(width, 1, 1)
        front_cfg = dict(front or {})
        front_type = front_cfg.pop("type", "fixed")
        self.front_checkpoint = bool(front_cfg.pop("checkpoint", True))
        if front_type not in ("fixed", "anisotropic"):
            raise ValueError("moe.front.type must be fixed or anisotropic.")
        if front_type == "fixed" and front_cfg:
            raise ValueError("Fixed FRONT does not accept anisotropic options.")
        self.front_anisotropic = (AnisotropicFront(width, scale, **front_cfg)
                                  if front_type == "anisotropic" else None)
        self.detail_conv = nn.Conv2d(3, width, 3)
        self.detail_out = nn.Conv2d(width, 1, 1)

        # Deep Expert（深层背景专家）
        self.deep = nn.Conv2d(width, 3, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))

        self.register_buffer("surface_mean", torch.zeros(5))
        self.register_buffer("surface_std", torch.ones(5))
        self.register_buffer("depth_m", torch.linspace(0, 2000, depths))
        self.register_buffer("temperature_mean", torch.zeros(depths))
        self.register_buffer("temperature_std", torch.ones(depths))
        self.register_buffer("clim_gradient", torch.zeros(12, depths, *low_size))
        self.register_buffer("isothermal_depth", torch.full((12, 1, *low_size), 50.0))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Older MoE checkpoints do not contain this buffer. Their default
        # routing weight is zero, so its value cannot affect their predictions.
        key = prefix + "temperature_mean"
        if key not in state_dict and self.local_gradient_weight == 0:
            state_dict[key] = self.temperature_mean
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @torch.no_grad()
    def set_statistics(self, **statistics) -> None:
        for name, values in statistics.items():
            buffer = getattr(self, name)
            values = torch.as_tensor(values, dtype=buffer.dtype, device=buffer.device)
            if values.shape != buffer.shape:
                raise ValueError(f"{name}: expected {tuple(buffer.shape)}, got {tuple(values.shape)}")
            buffer.copy_(values)

    def diagnostics(self, x: torch.Tensor, validity: torch.Tensor | None):
        ocean = x[:, 8:9].clamp(0, 1)
        valid = ocean.expand(-1, 5, -1, -1) if validity is None else validity * ocean
        physical = x[:, :5] * self.surface_std.view(1, 5, 1, 1)
        physical = physical + self.surface_mean.view(1, 5, 1, 1)
        gradients = [gradient_100km(x[:, j:j + 1], valid[:, j:j + 1], x[:, 5:6])
                     for j in range(3)]
        speed = torch.linalg.vector_norm(physical[:, 3:5], dim=1, keepdim=True)
        speed = speed / torch.linalg.vector_norm(self.surface_std[3:5]).clamp_min(1)
        groups = [torch.cat((x[:, j:j + 1], gradients[j], valid[:, j:j + 1]), 1) * ocean
                  for j in range(3)]
        wind_valid = valid[:, 3:4] * valid[:, 4:5]
        groups.append(torch.cat((x[:, 3:5], speed * wind_valid, wind_valid), 1) * ocean)
        geo = x[:, [5, 6, 7, 9, 10]] * ocean
        return groups, geo, gradients

    def local_cooling_gradient(self, baseline: torch.Tensor,
                               ocean: torch.Tensor) -> torch.Tensor:
        """Normalized cooling strength at each depth from the current prediction."""
        if baseline.ndim != 4 or baseline.shape[1] != self.depth_count:
            raise ValueError("MoE baseline depth shape mismatch.")
        if baseline.shape[-2:] != ocean.shape[-2:]:
            raise ValueError("MoE baseline and ocean mask spatial shapes differ.")
        if baseline.shape[-2:] != tuple(size * self.scale for size in self.low_size):
            raise ValueError("MoE baseline resolution mismatch.")
        if self.depth_count < 2:
            return baseline.new_zeros((baseline.shape[0], self.depth_count, *self.low_size))

        # Work in physical degrees C and FP32, including under training AMP.
        physical = baseline.float() * self.temperature_std.float().view(1, -1, 1, 1)
        physical = physical + self.temperature_mean.float().view(1, -1, 1, 1)
        valid = ocean.float().clamp(0, 1)
        counts = F.avg_pool2d(valid, self.scale, stride=self.scale)
        pooled = F.avg_pool2d(
            torch.where(valid > 0, physical, torch.zeros_like(physical)),
            self.scale, stride=self.scale,
        ) / counts.clamp_min(1e-6)
        dz = torch.diff(self.depth_m.float())
        if torch.any(dz <= 0):
            raise ValueError("MoE depths must be strictly increasing.")
        cooling = ((pooled[:, :-1] - pooled[:, 1:])
                   / dz.view(1, -1, 1, 1)).clamp_min(0)
        local = torch.cat((
            cooling[:, :1],
            0.5 * (cooling[:, :-1] + cooling[:, 1:]),
            cooling[:, -1:],
        ), dim=1)
        local = local / local.amax(dim=1, keepdim=True).clamp_min(1e-5)
        return local * (counts > 0).to(local.dtype)

    def forward(self, feature: torch.Tensor, surface: torch.Tensor,
                baseline: torch.Tensor, validity: torch.Tensor | None = None,
                return_routes: bool = False):
        batch, _, low_h, low_w = feature.shape
        if (low_h, low_w) != self.low_size:
            raise ValueError("MoE feature resolution mismatch.")
        if baseline.shape[0] != batch:
            raise ValueError("MoE baseline batch shape mismatch.")
        groups, geo_input, gradients = self.diagnostics(surface, validity)
        group_features = [stem(group) for stem, group in zip(self.groups, groups)]
        geo, shared = self.geo(geo_input), self.shared(feature)
        front_input = torch.cat(gradients, 1)
        front_low = F.avg_pool2d(front_input.mean(1, keepdim=True), self.scale)
        detail = self.detail_out(F.gelu(geographic_conv(front_input, self.detail_conv)))
        angle = torch.atan2(surface[:, 9, 0, 0], surface[:, 10, 0, 0])
        month = torch.remainder(torch.round(angle * (6 / math.pi)).long(), 12)
        climatology = self.clim_gradient[month]
        isothermal = self.isothermal_depth[month]
        max_gradient = climatology.amax(1, keepdim=True).clamp_min(1e-5)
        local_gradient = (
            self.local_cooling_gradient(baseline, surface[:, 8:9])
            if self.local_gradient_weight > 0 else None
        )
        depth_max = self.depth_m[-1].clamp_min(1)
        front_frame = front_coefficients = None
        if self.front_anisotropic is not None:
            front_frame = self.front_anisotropic.frame(surface, validity)
            # Depth-independent context; all depth variation in the geometry
            # comes explicitly from the normalized physical-depth basis.
            front_context = shared + geo + 0.25 * sum(group_features)
            front_coefficients = self.front_anisotropic.coefficients(front_context)
        residuals, alphas, betas = [], [], []
        for j in range(self.depth_count):
            z = self.depth_m[j]
            fraction = z / depth_max
            depth_map = fraction.expand(batch, 1, low_h, low_w)
            scores = torch.cat([
                router(torch.cat((shared, geo, group, depth_map), 1))
                for router, group in zip(self.variable_router, group_features)
            ], 1)
            bias = torch.stack((-fraction, fraction * 0,
                                -(fraction - 0.2).abs(), -2 * fraction))
            alpha = torch.softmax(scores.float() + self.prior_strength * bias.view(1, 4, 1, 1), 1)
            alpha = alpha.to(shared.dtype)
            fused = shared + geo + sum(alpha[:, k:k + 1] * group_features[k]
                                       for k in range(4))
            process_scores = self.process_router(torch.cat((fused, geo, depth_map), 1))
            gradient = climatology[:, j:j + 1]
            q_ml = torch.sigmoid((isothermal - z) / 40)
            q_th = (gradient / max_gradient).clamp_min(0)
            if local_gradient is not None:
                q_th = ((1 - self.local_gradient_weight) * q_th
                        + self.local_gradient_weight * local_gradient[:, j:j + 1])
            q_fe = torch.sigmoid(front_low - 0.5) * torch.exp(-z / 500)
            q_db = torch.sigmoid((z - 700) / 150).expand_as(q_ml)
            prior = torch.cat((q_ml, q_th, q_fe, q_db), 1).clamp_min(0.05)
            beta = torch.softmax(process_scores.float() + self.prior_strength * prior.log(), 1)
            beta = beta.to(fused.dtype)

            # Mixed Expert（混合层专家）
            coefficients = self.mixed(fused)
            thickness = 20 + 480 * torch.sigmoid(coefficients[:, 4:5])
            ratio = z / thickness
            mixed = torch.exp(-ratio) * sum(coefficients[:, k:k + 1] * ratio.pow(k)
                                            for k in range(4))

            # Heave Expert（等温面位移专家），直接使用 clim_gradient 来计算
            displacement = 200 * torch.tanh(self.heave(fused))
            heave = -displacement * gradient / self.temperature_std[j].clamp_min(1e-6)
            heave = heave + 0.1 * self.heave_shape(fused) * q_th

            # Front Expert（水平锋面专家）
            front_hidden = geographic_conv(fused, self.front_conv)
            if self.front_anisotropic is not None:
                def anisotropic_branch(values, coefficients, depth_fraction):
                    return self.front_anisotropic(values, coefficients, depth_fraction, front_frame)
                if self.training and torch.is_grad_enabled() and self.front_checkpoint:
                    correction = checkpoint(anisotropic_branch, fused, front_coefficients,
                                            fraction, use_reentrant=False)
                else:
                    correction = anisotropic_branch(fused, front_coefficients, fraction)
                front_hidden = front_hidden + correction
            front = self.front_out(F.gelu(front_hidden))

            # Deep Expert（深层背景专家）
            deep_coeff = self.deep(geographic_pool(fused))
            deep = sum(deep_coeff[:, k:k + 1] * fraction.pow(k) for k in range(3))

            # 专家输出的加权融合
            # mixed、heave、front、deep 是四个专家实际计算的温度残差
            # beta 是前面 Process Router 计算的四个专家权重
            low = (beta[:, 0:1] * mixed + beta[:, 1:2] * heave
                   + beta[:, 2:3] * front + beta[:, 3:4] * deep)
            up = F.interpolate(low, scale_factor=self.scale, mode="bilinear",
                               align_corners=False)
            front_beta = F.interpolate(beta[:, 2:3], scale_factor=self.scale,
                                       mode="bilinear", align_corners=False)
            residuals.append(up + front_beta * detail)
            if return_routes:
                alphas.append(alpha)
                betas.append(beta)
        residual = self.residual_scale * torch.cat(residuals, 1) * surface[:, 8:9]
        if return_routes:
            return residual, torch.stack(alphas, 1), torch.stack(betas, 1)
        return residual



# """双层路由、四类物理过程专家、残差修正式 MoE。 加入可变形卷积"""

# from __future__ import annotations

# import math

# import torch
# from torch import nn
# from torch.nn import functional as F


# def geographic_conv(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
#     x = F.pad(x, (1, 1, 0, 0), mode="circular")
#     x = F.pad(x, (0, 0, 1, 1), mode="replicate")
#     return conv(x)


# def geographic_pool(x: torch.Tensor) -> torch.Tensor:
#     x = F.pad(x, (2, 2, 0, 0), mode="circular")
#     x = F.pad(x, (0, 0, 2, 2), mode="replicate")
#     return F.avg_pool2d(x, 5, stride=1)


# def gradient_100km(
#     field: torch.Tensor,
#     valid: torch.Tensor,
#     latitude_sin: torch.Tensor,
# ) -> torch.Tensor:
#     """Masked central gradient; longitude is periodic, latitude is not."""
#     step_y = 111.2 * 0.25
#     step_x = (
#         step_y * torch.sqrt((1 - latitude_sin.square()).clamp_min(0))
#     ).clamp_min(1)

#     east, west = field.roll(-1, -1), field.roll(1, -1)
#     north = F.pad(field[..., :-1, :], (0, 0, 1, 0), mode="replicate")
#     south = F.pad(field[..., 1:, :], (0, 0, 0, 1), mode="replicate")

#     x_valid = valid * valid.roll(-1, -1) * valid.roll(1, -1)
#     y_valid = valid * F.pad(
#         valid[..., :-1, :], (0, 0, 1, 0), mode="replicate"
#     )
#     y_valid = y_valid * F.pad(
#         valid[..., 1:, :], (0, 0, 0, 1), mode="replicate"
#     )

#     gx = (east - west) * (50 / step_x) * x_valid
#     gy = (south - north) * (50 / step_y) * y_valid
#     return torch.asinh(
#         (torch.sqrt(gx.square() + gy.square() + 1e-12) - 1e-6).clamp_min(0)
#     )


# class ProcessMoE(nn.Module):
#     """四个观测变量组路由器和四个物理过程残差专家。"""

#     def __init__(
#         self,
#         channels: int,
#         depths: int,
#         low_size: tuple[int, int],
#         scale: int,
#         width: int = 16,
#         prior_strength: float = 0.5,
#         local_gradient_weight: float = 0.0,
#     ) -> None:
#         super().__init__()

#         if width <= 0 or prior_strength < 0:
#             raise ValueError("Invalid MoE width or prior strength.")
#         if not 0.0 <= local_gradient_weight <= 1.0:
#             raise ValueError(
#                 "local_gradient_weight must be between 0 and 1."
#             )

#         self.depth_count = depths
#         self.low_size = low_size
#         self.scale = scale
#         self.prior_strength = prior_strength
#         self.local_gradient_weight = local_gradient_weight

#         self.shared = nn.Conv2d(channels, width, 1)
#         self.groups = nn.ModuleList(
#             nn.Conv2d(n, width, scale, stride=scale)
#             for n in (3, 3, 3, 4)
#         )
#         self.geo = nn.Conv2d(5, width, scale, stride=scale)
#         self.variable_router = nn.ModuleList(
#             nn.Conv2d(3 * width + 1, 1, 1)
#             for _ in range(4)
#         )

#         # 物理过程路由
#         self.process_router = nn.Conv2d(2 * width + 1, 4, 1)

#         # Mixed Expert
#         self.mixed = nn.Conv2d(width, 5, 1)

#         # Heave Expert
#         self.heave = nn.Conv2d(width, 1, 1)
#         self.heave_shape = nn.Conv2d(width, 1, 1)

#         # Front Expert：沿用原 3×3 卷积的九组权重，
#         # 将固定位置采样改为受锋面方向约束的采样。
#         self.front_conv = nn.Conv2d(width, width, 3)
#         self.front_out = nn.Conv2d(width, 1, 1)

#         # 每个位置仅预测沿锋面、跨锋面两个采样距离。
#         self.front_span = nn.Conv2d(width, 2, 1)
#         nn.init.zeros_(self.front_span.weight)
#         nn.init.zeros_(self.front_span.bias)

#         # 初始为零，使初始采样位置等同原固定 3×3 网格。
#         self.front_deform_gain = nn.Parameter(torch.zeros(()))

#         self.detail_conv = nn.Conv2d(3, width, 3)
#         self.detail_out = nn.Conv2d(width, 1, 1)

#         # Deep Expert
#         self.deep = nn.Conv2d(width, 3, 1)
#         self.residual_scale = nn.Parameter(torch.zeros(()))

#         self.register_buffer("surface_mean", torch.zeros(5))
#         self.register_buffer("surface_std", torch.ones(5))
#         self.register_buffer("depth_m", torch.linspace(0, 2000, depths))
#         self.register_buffer("temperature_mean", torch.zeros(depths))
#         self.register_buffer("temperature_std", torch.ones(depths))
#         self.register_buffer(
#             "clim_gradient", torch.zeros(12, depths, *low_size)
#         )
#         self.register_buffer(
#             "isothermal_depth",
#             torch.full((12, 1, *low_size), 50.0),
#         )

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         # 兼容不含 temperature_mean 的旧检查点。
#         key = prefix + "temperature_mean"
#         if key not in state_dict and self.local_gradient_weight == 0:
#             state_dict[key] = self.temperature_mean

#         # 兼容尚未加入 Front 自适应采样的旧检查点。
#         front_defaults = {
#             "front_span.weight": self.front_span.weight,
#             "front_span.bias": self.front_span.bias,
#             "front_deform_gain": self.front_deform_gain,
#         }
#         for name, value in front_defaults.items():
#             key = prefix + name
#             if key not in state_dict:
#                 state_dict[key] = value.detach().clone()

#         super()._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )

#     @torch.no_grad()
#     def set_statistics(self, **statistics) -> None:
#         for name, values in statistics.items():
#             buffer = getattr(self, name)
#             values = torch.as_tensor(
#                 values,
#                 dtype=buffer.dtype,
#                 device=buffer.device,
#             )
#             if values.shape != buffer.shape:
#                 raise ValueError(
#                     f"{name}: expected {tuple(buffer.shape)}, "
#                     f"got {tuple(values.shape)}"
#                 )
#             buffer.copy_(values)

#     def diagnostics(
#         self,
#         x: torch.Tensor,
#         validity: torch.Tensor | None,
#     ):
#         ocean = x[:, 8:9].clamp(0, 1)
#         valid = (
#             ocean.expand(-1, 5, -1, -1)
#             if validity is None
#             else validity * ocean
#         )

#         physical = x[:, :5] * self.surface_std.view(1, 5, 1, 1)
#         physical = physical + self.surface_mean.view(1, 5, 1, 1)

#         gradients = [
#             gradient_100km(
#                 x[:, j:j + 1],
#                 valid[:, j:j + 1],
#                 x[:, 5:6],
#             )
#             for j in range(3)
#         ]

#         speed = torch.linalg.vector_norm(
#             physical[:, 3:5], dim=1, keepdim=True
#         )
#         speed = speed / torch.linalg.vector_norm(
#             self.surface_std[3:5]
#         ).clamp_min(1)

#         groups = [
#             torch.cat(
#                 (
#                     x[:, j:j + 1],
#                     gradients[j],
#                     valid[:, j:j + 1],
#                 ),
#                 1,
#             ) * ocean
#             for j in range(3)
#         ]

#         wind_valid = valid[:, 3:4] * valid[:, 4:5]
#         groups.append(
#             torch.cat(
#                 (
#                     x[:, 3:5],
#                     speed * wind_valid,
#                     wind_valid,
#                 ),
#                 1,
#             ) * ocean
#         )
#         geo = x[:, [5, 6, 7, 9, 10]] * ocean
#         return groups, geo, gradients

#     def local_cooling_gradient(
#         self,
#         baseline: torch.Tensor,
#         ocean: torch.Tensor,
#     ) -> torch.Tensor:
#         """由当月共享头预测剖面计算逐深度的局地降温强度。"""
#         if baseline.ndim != 4 or baseline.shape[1] != self.depth_count:
#             raise ValueError("MoE baseline depth shape mismatch.")
#         if baseline.shape[-2:] != ocean.shape[-2:]:
#             raise ValueError(
#                 "MoE baseline and ocean mask spatial shapes differ."
#             )
#         if baseline.shape[-2:] != tuple(
#             size * self.scale for size in self.low_size
#         ):
#             raise ValueError("MoE baseline resolution mismatch.")
#         if self.depth_count < 2:
#             return baseline.new_zeros(
#                 (baseline.shape[0], self.depth_count, *self.low_size)
#             )

#         # 在物理温度和 FP32 中计算；不使用真实次表层标签。
#         physical = (
#             baseline.float()
#             * self.temperature_std.float().view(1, -1, 1, 1)
#         )
#         physical = (
#             physical
#             + self.temperature_mean.float().view(1, -1, 1, 1)
#         )

#         valid = ocean.float().clamp(0, 1)
#         counts = F.avg_pool2d(
#             valid, self.scale, stride=self.scale
#         )
#         pooled = F.avg_pool2d(
#             torch.where(
#                 valid > 0,
#                 physical,
#                 torch.zeros_like(physical),
#             ),
#             self.scale,
#             stride=self.scale,
#         ) / counts.clamp_min(1e-6)

#         dz = torch.diff(self.depth_m.float())
#         if torch.any(dz <= 0):
#             raise ValueError("MoE depths must be strictly increasing.")

#         cooling = (
#             (pooled[:, :-1] - pooled[:, 1:])
#             / dz.view(1, -1, 1, 1)
#         ).clamp_min(0)

#         local = torch.cat(
#             (
#                 cooling[:, :1],
#                 0.5 * (cooling[:, :-1] + cooling[:, 1:]),
#                 cooling[:, -1:],
#             ),
#             dim=1,
#         )
#         local = local / local.amax(
#             dim=1, keepdim=True
#         ).clamp_min(1e-5)

#         return local * (counts > 0).to(local.dtype)

#     def front_frame(
#         self,
#         surface: torch.Tensor,
#         validity: torch.Tensor | None,
#     ):
#         """计算低分辨率锋面的网格坐标法向、切向和可信度。"""
#         ocean = surface[:, 8:9].float().clamp(0, 1)
#         sst_valid = (
#             ocean
#             if validity is None
#             else ocean * validity[:, 0:1].float().clamp(0, 1)
#         )

#         counts = F.avg_pool2d(
#             sst_valid, self.scale, stride=self.scale
#         )
#         sst = F.avg_pool2d(
#             surface[:, 0:1].float() * sst_valid,
#             self.scale,
#             stride=self.scale,
#         ) / counts.clamp_min(1e-6)
#         valid_low = (counts > 0.5).float()
#         ocean_low = F.avg_pool2d(
#             ocean, self.scale, stride=self.scale
#         )

#         latitude_sin = F.avg_pool2d(
#             surface[:, 5:6].float(),
#             self.scale,
#             stride=self.scale,
#         )

#         # 一个低分辨率网格单元对应的实际距离，单位 km。
#         step_y = 111.2 * 0.25 * self.scale
#         step_x = (
#             step_y
#             * torch.sqrt(
#                 (1 - latitude_sin.square()).clamp_min(0)
#             )
#         ).clamp_min(1)

#         east = sst.roll(-1, -1)
#         west = sst.roll(1, -1)
#         north = F.pad(
#             sst[..., :-1, :],
#             (0, 0, 1, 0),
#             mode="replicate",
#         )
#         south = F.pad(
#             sst[..., 1:, :],
#             (0, 0, 0, 1),
#             mode="replicate",
#         )

#         x_valid = (
#             valid_low
#             * valid_low.roll(-1, -1)
#             * valid_low.roll(1, -1)
#         )
#         y_valid = valid_low * F.pad(
#             valid_low[..., :-1, :],
#             (0, 0, 1, 0),
#             mode="replicate",
#         )
#         y_valid = y_valid * F.pad(
#             valid_low[..., 1:, :],
#             (0, 0, 0, 1),
#             mode="replicate",
#         )

#         # gx、gy 是按实际距离计算的温度梯度。
#         gx = (east - west) / (2 * step_x) * x_valid
#         gy = (south - north) / (2 * step_y) * y_valid
#         magnitude = torch.sqrt(gx.square() + gy.square())

#         # 先在物理距离坐标中确定跨锋面法向。
#         normal_x_physical = gx / magnitude.clamp_min(1e-8)
#         normal_y_physical = gy / magnitude.clamp_min(1e-8)

#         # 物理距离坐标中的沿锋面切向。
#         tangent_x_physical = -normal_y_physical
#         tangent_y_physical = normal_x_physical

#         # 再转换为“经度、纬度网格单元”坐标中的方向。
#         # 这一步避免将公里坐标中的方向直接用作 dx/dy。
#         normal_x_grid = normal_x_physical / step_x
#         normal_y_grid = normal_y_physical / step_y
#         normal_length_grid = torch.sqrt(
#             normal_x_grid.square() + normal_y_grid.square()
#         ).clamp_min(1e-8)
#         normal_x = normal_x_grid / normal_length_grid
#         normal_y = normal_y_grid / normal_length_grid

#         tangent_x_grid = tangent_x_physical / step_x
#         tangent_y_grid = tangent_y_physical / step_y
#         tangent_length_grid = torch.sqrt(
#             tangent_x_grid.square() + tangent_y_grid.square()
#         ).clamp_min(1e-8)
#         tangent_x = tangent_x_grid / tangent_length_grid
#         tangent_y = tangent_y_grid / tangent_length_grid

#         reference = (
#             magnitude.sum(dim=(-2, -1), keepdim=True)
#             / valid_low.sum(
#                 dim=(-2, -1), keepdim=True
#             ).clamp_min(1)
#         ).detach()
#         confidence = (
#             magnitude
#             / (magnitude + reference.clamp_min(1e-8))
#         ) * valid_low

#         return (
#             normal_x,
#             normal_y,
#             tangent_x,
#             tangent_y,
#             confidence,
#             ocean_low,
#         )

#     def front_sampling_geometry(
#         self,
#         surface: torch.Tensor,
#         validity: torch.Tensor | None,
#         context: torch.Tensor,
#     ):
#         """构造九点采样网格；在所有深度层复用。"""
#         (
#             normal_x,
#             normal_y,
#             tangent_x,
#             tangent_y,
#             confidence,
#             ocean_low,
#         ) = self.front_frame(surface, validity)

#         _, _, height, width = context.shape

#         # 两个有界距离分别控制沿锋面和跨锋面的成对采样。
#         # 单位为低分辨率网格单元。
#         span = self.front_span(context).float()
#         tangent_radius = 1.0 + 0.5 * torch.tanh(span[:, 0:1])
#         normal_radius = 1.0 + 0.5 * torch.tanh(span[:, 1:2])

#         # 零初始化时 strength=0，严格使用原 3×3 网格。
#         strength = torch.tanh(self.front_deform_gain) * confidence

#         # 顺序对应 Conv2d 的 3×3 卷积核，按行展开。
#         base_positions = (
#             (-1, -1), (0, -1), (1, -1),
#             (-1,  0), (0,  0), (1,  0),
#             (-1,  1), (0,  1), (1,  1),
#         )

#         # 相对点成对布置：四点沿锋面、四点跨锋面、
#         # 一个中心点固定不动。
#         target_positions = (
#             (
#                 -2 * tangent_radius * tangent_x,
#                 -2 * tangent_radius * tangent_y,
#             ),
#             (
#                 -tangent_radius * tangent_x,
#                 -tangent_radius * tangent_y,
#             ),
#             (
#                 -2 * normal_radius * normal_x,
#                 -2 * normal_radius * normal_y,
#             ),
#             (
#                 -normal_radius * normal_x,
#                 -normal_radius * normal_y,
#             ),
#             (None, None),
#             (
#                 normal_radius * normal_x,
#                 normal_radius * normal_y,
#             ),
#             (
#                 2 * normal_radius * normal_x,
#                 2 * normal_radius * normal_y,
#             ),
#             (
#                 tangent_radius * tangent_x,
#                 tangent_radius * tangent_y,
#             ),
#             (
#                 2 * tangent_radius * tangent_x,
#                 2 * tangent_radius * tangent_y,
#             ),
#         )

#         yy = torch.arange(
#             height,
#             device=context.device,
#             dtype=torch.float32,
#         ).view(1, 1, height, 1)
#         xx = torch.arange(
#             width,
#             device=context.device,
#             dtype=torch.float32,
#         ).view(1, 1, 1, width)

#         ocean_tiled = torch.cat(
#             (ocean_low, ocean_low, ocean_low),
#             dim=-1,
#         )
#         grids = []
#         coast_blends = []

#         for (base_x, base_y), (target_x, target_y) in zip(
#             base_positions,
#             target_positions,
#         ):
#             if target_x is None:
#                 dx = torch.zeros_like(confidence)
#                 dy = torch.zeros_like(confidence)
#             else:
#                 dx = base_x + strength * (target_x - base_x)
#                 dy = base_y + strength * (target_y - base_y)

#             # 三份经度场中的中间一份是采样中心；
#             # 左右两份处理全球经度周期接缝。
#             grid_x = 2 * (xx + width + dx) / (3 * width - 1) - 1
#             grid_y = 2 * (yy + dy) / (height - 1) - 1
#             grid = torch.stack(
#                 (grid_x[:, 0], grid_y[:, 0]),
#                 dim=-1,
#             )
#             grids.append(grid)

#             sampled_ocean = F.grid_sample(
#                 ocean_tiled,
#                 grid,
#                 mode="bilinear",
#                 padding_mode="border",
#                 align_corners=True,
#             )

#             # 当变形采样落到陆地时，逐渐退回中心特征。
#             # strength=0 时不改变原固定卷积的行为。
#             coast_blends.append(
#                 strength.abs()
#                 * (1 - sampled_ocean).clamp(0, 1)
#             )

#         return grids, coast_blends

#     def adaptive_front_conv(
#         self,
#         fused: torch.Tensor,
#         grids: list[torch.Tensor],
#         coast_blends: list[torch.Tensor],
#     ) -> torch.Tensor:
#         """在受约束的九个位置应用原 3×3 卷积的九组权重。"""
#         source = fused.float()
#         source_tiled = torch.cat(
#             (source, source, source),
#             dim=-1,
#         )

#         result = None
#         for index, (grid, coast_blend) in enumerate(
#             zip(grids, coast_blends)
#         ):
#             sampled = F.grid_sample(
#                 source_tiled,
#                 grid,
#                 mode="bilinear",
#                 padding_mode="border",
#                 align_corners=True,
#             )
#             sampled = (
#                 sampled
#                 + coast_blend * (source - sampled)
#             )

#             row, column = divmod(index, 3)
#             weight = self.front_conv.weight[
#                 :, :, row, column
#             ].unsqueeze(-1).unsqueeze(-1)
#             contribution = F.conv2d(sampled, weight)
#             result = (
#                 contribution
#                 if result is None
#                 else result + contribution
#             )

#         if self.front_conv.bias is not None:
#             result = result + self.front_conv.bias.view(1, -1, 1, 1)
#         return result

#     def forward(
#         self,
#         feature: torch.Tensor,
#         surface: torch.Tensor,
#         baseline: torch.Tensor,
#         validity: torch.Tensor | None = None,
#         return_routes: bool = False,
#     ):
#         batch, _, low_h, low_w = feature.shape
#         if (low_h, low_w) != self.low_size:
#             raise ValueError("MoE feature resolution mismatch.")
#         if baseline.shape[0] != batch:
#             raise ValueError("MoE baseline batch shape mismatch.")

#         groups, geo_input, gradients = self.diagnostics(
#             surface, validity
#         )
#         group_features = [
#             stem(group)
#             for stem, group in zip(self.groups, groups)
#         ]
#         geo = self.geo(geo_input)
#         shared = self.shared(feature)

#         front_input = torch.cat(gradients, 1)
#         front_low = F.avg_pool2d(
#             front_input.mean(1, keepdim=True),
#             self.scale,
#         )
#         detail = self.detail_out(
#             F.gelu(
#                 geographic_conv(
#                     front_input,
#                     self.detail_conv,
#                 )
#             )
#         )

#         # 采样方向和位置只计算一次，所有深度层共用。
#         front_context = (
#             shared + geo + 0.25 * sum(group_features)
#         )
#         front_grids, coast_blends = (
#             self.front_sampling_geometry(
#                 surface,
#                 validity,
#                 front_context,
#             )
#         )

#         angle = torch.atan2(
#             surface[:, 9, 0, 0],
#             surface[:, 10, 0, 0],
#         )
#         month = torch.remainder(
#             torch.round(angle * (6 / math.pi)).long(),
#             12,
#         )
#         climatology = self.clim_gradient[month]
#         isothermal = self.isothermal_depth[month]
#         max_gradient = climatology.amax(
#             1, keepdim=True
#         ).clamp_min(1e-5)

#         local_gradient = (
#             self.local_cooling_gradient(
#                 baseline,
#                 surface[:, 8:9],
#             )
#             if self.local_gradient_weight > 0
#             else None
#         )

#         depth_max = self.depth_m[-1].clamp_min(1)
#         residuals, alphas, betas = [], [], []

#         for j in range(self.depth_count):
#             z = self.depth_m[j]
#             fraction = z / depth_max
#             depth_map = fraction.expand(
#                 batch, 1, low_h, low_w
#             )

#             scores = torch.cat(
#                 [
#                     router(
#                         torch.cat(
#                             (
#                                 shared,
#                                 geo,
#                                 group,
#                                 depth_map,
#                             ),
#                             1,
#                         )
#                     )
#                     for router, group in zip(
#                         self.variable_router,
#                         group_features,
#                     )
#                 ],
#                 1,
#             )

#             bias = torch.stack(
#                 (
#                     -fraction,
#                     fraction * 0,
#                     -(fraction - 0.2).abs(),
#                     -2 * fraction,
#                 )
#             )
#             alpha = torch.softmax(
#                 scores.float()
#                 + self.prior_strength
#                 * bias.view(1, 4, 1, 1),
#                 1,
#             )
#             alpha = alpha.to(shared.dtype)

#             fused = (
#                 shared
#                 + geo
#                 + sum(
#                     alpha[:, k:k + 1]
#                     * group_features[k]
#                     for k in range(4)
#                 )
#             )

#             process_scores = self.process_router(
#                 torch.cat(
#                     (fused, geo, depth_map),
#                     1,
#                 )
#             )

#             gradient = climatology[:, j:j + 1]
#             q_ml = torch.sigmoid(
#                 (isothermal - z) / 40
#             )
#             q_th = (
#                 gradient / max_gradient
#             ).clamp_min(0)

#             if local_gradient is not None:
#                 q_th = (
#                     (1 - self.local_gradient_weight)
#                     * q_th
#                     + self.local_gradient_weight
#                     * local_gradient[:, j:j + 1]
#                 )

#             q_fe = (
#                 torch.sigmoid(front_low - 0.5)
#                 * torch.exp(-z / 500)
#             )
#             q_db = torch.sigmoid(
#                 (z - 700) / 150
#             ).expand_as(q_ml)

#             prior = torch.cat(
#                 (q_ml, q_th, q_fe, q_db),
#                 1,
#             ).clamp_min(0.05)
#             beta = torch.softmax(
#                 process_scores.float()
#                 + self.prior_strength * prior.log(),
#                 1,
#             )
#             beta = beta.to(fused.dtype)

#             # Mixed Expert
#             coefficients = self.mixed(fused)
#             thickness = (
#                 20
#                 + 480
#                 * torch.sigmoid(
#                     coefficients[:, 4:5]
#                 )
#             )
#             ratio = z / thickness
#             mixed = torch.exp(-ratio) * sum(
#                 coefficients[:, k:k + 1]
#                 * ratio.pow(k)
#                 for k in range(4)
#             )

#             # Heave Expert
#             displacement = (
#                 200 * torch.tanh(self.heave(fused))
#             )
#             heave = (
#                 -displacement
#                 * gradient
#                 / self.temperature_std[j].clamp_min(1e-6)
#             )
#             heave = (
#                 heave
#                 + 0.1
#                 * self.heave_shape(fused)
#                 * q_th
#             )

#             # Front Expert：跨锋面、沿锋面成对采样。
#             front = self.front_out(
#                 F.gelu(
#                     self.adaptive_front_conv(
#                         fused,
#                         front_grids,
#                         coast_blends,
#                     )
#                 )
#             )

#             # Deep Expert
#             deep_coeff = self.deep(
#                 geographic_pool(fused)
#             )
#             deep = sum(
#                 deep_coeff[:, k:k + 1]
#                 * fraction.pow(k)
#                 for k in range(3)
#             )

#             # 四个专家的温度残差由过程路由权重融合。
#             low = (
#                 beta[:, 0:1] * mixed
#                 + beta[:, 1:2] * heave
#                 + beta[:, 2:3] * front
#                 + beta[:, 3:4] * deep
#             )
#             up = F.interpolate(
#                 low,
#                 scale_factor=self.scale,
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             front_beta = F.interpolate(
#                 beta[:, 2:3],
#                 scale_factor=self.scale,
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             residuals.append(
#                 up + front_beta * detail
#             )

#             if return_routes:
#                 alphas.append(alpha)
#                 betas.append(beta)

#         residual = (
#             self.residual_scale
#             * torch.cat(residuals, 1)
#             * surface[:, 8:9]
#         )
#         if return_routes:
#             return (
#                 residual,
#                 torch.stack(alphas, 1),
#                 torch.stack(betas, 1),
#             )
#         return residual
