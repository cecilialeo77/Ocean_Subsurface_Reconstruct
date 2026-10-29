"""SST-guided, depth-conditioned anisotropic FRONT features.

Distances are in km before conversion to the model's longitude/latitude grid.
Only surface inputs and learned context are used; no subsurface labels enter
sampling geometry. Bilinear gather avoids a grid_sample dependency.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def periodic_bilinear_sample(source: torch.Tensor, dx: torch.Tensor,
                             dy: torch.Tensor) -> torch.Tensor:
    """Sample B,C,H,W at B,K,H,W offsets, periodic x and replicated y.

    Returns B,C,K,H,W. Coordinates and interpolation accumulate in FP32.
    """
    b, c, h, w = source.shape
    if dx.shape != dy.shape or dx.shape[0] != b or dx.shape[-2:] != (h, w):
        raise ValueError("Sampling offsets must have shape B,K,H,W.")
    yy = torch.arange(h, device=source.device, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=source.device, dtype=torch.float32).view(1, 1, 1, w)
    # Wrap integer indices, not floating coordinates: remainder(-tiny, w)
    # can round to exactly w in FP32 and otherwise produce an invalid index.
    x = xx + dx.float()
    y = (yy + dy.float()).clamp(0, h - 1)
    x_floor, y_floor = x.floor(), y.floor()
    wx, wy = x - x_floor, y - y_floor
    x0, y0 = x_floor.long().remainder(w), y_floor.long()
    x1, y1 = (x0 + 1).remainder(w), (y0 + 1).clamp_max(h - 1)
    flat = source.float().flatten(2)
    def gather(ix, iy):
        index = (iy * w + ix).flatten(1).unsqueeze(1).expand(-1, c, -1)
        return flat.gather(2, index).reshape(b, c, dx.shape[1], h, w)
    return ((1 - wx).unsqueeze(1) * (1 - wy).unsqueeze(1) * gather(x0, y0)
            + wx.unsqueeze(1) * (1 - wy).unsqueeze(1) * gather(x1, y0)
            + (1 - wx).unsqueeze(1) * wy.unsqueeze(1) * gather(x0, y1)
            + wx.unsqueeze(1) * wy.unsqueeze(1) * gather(x1, y1))


class AnisotropicFront(nn.Module):
    """Additive FRONT features with a zero-initialized output projection.

    Five geometry fields (two radii, angle, two center shifts) each have
    three learned polynomial coefficients in normalized physical depth.
    Setting depth_conditioning=False removes the two nonconstant terms.
    """
    def __init__(self, width: int, scale: int, *, depth_conditioning: bool = True,
                 tangent_radius_km: float = 110.0, normal_radius_km: float = 40.0,
                 max_rotation_deg: float = 30.0, max_shift_km: float = 30.0,
                 max_offset_cells: float = 4.0, grid_spacing_deg: float = 0.25):
        super().__init__()
        values = (tangent_radius_km, normal_radius_km, max_offset_cells, grid_spacing_deg)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("FRONT radii, offset cap and grid spacing must be finite and positive.")
        if not 0 <= max_rotation_deg <= 90 or not math.isfinite(max_shift_km) or max_shift_km < 0:
            raise ValueError("Invalid FRONT rotation or shift bound.")
        self.scale = scale
        self.depth_conditioning = depth_conditioning
        self.tangent_radius_km = tangent_radius_km
        self.normal_radius_km = normal_radius_km
        self.max_rotation = math.radians(max_rotation_deg)
        self.max_shift_km = max_shift_km
        self.max_offset_cells = max_offset_cells
        self.grid_spacing_deg = grid_spacing_deg
        self.geometry_head = nn.Conv2d(width, 15, 1)
        self.projection = nn.Conv2d(4 * width, width, 1)
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def frame(self, surface: torch.Tensor, validity: torch.Tensor | None):
        """Return signed SST normal, confidence, cell distances, ocean fraction.

        Gradient stencils require valid center and both neighbors in both axes.
        Weak/invalid gradients have zero confidence. SST is standardized, which
        preserves its gradient direction under a positive global scale.
        """
        with torch.autocast(device_type=surface.device.type, enabled=False):
            ocean = surface[:, 8:9].float().clamp(0, 1)
            valid = ocean if validity is None else ocean * validity[:, :1].float().clamp(0, 1)
            valid = valid * torch.isfinite(surface[:, :1]).float()
            count = F.avg_pool2d(valid, self.scale, self.scale)
            sst = F.avg_pool2d(torch.where(valid > 0, surface[:, :1].float(), 0) * valid,
                               self.scale, self.scale) / count.clamp_min(1e-6)
            good = (count > 0.5).float()
            ocean_low = F.avg_pool2d(ocean, self.scale, self.scale)
            lat = F.avg_pool2d(surface[:, 5:6].float(), self.scale, self.scale)
            step_y = torch.full_like(lat, 111.2 * self.grid_spacing_deg * self.scale)
            step_x = (step_y * (1 - lat.square()).clamp_min(0).sqrt()).clamp_min(1)
            def above(x):
                return F.pad(x[..., :-1, :], (0, 0, 1, 0), mode="replicate")
            def below(x):
                return F.pad(x[..., 1:, :], (0, 0, 0, 1), mode="replicate")
            stencil = good * good.roll(1, -1) * good.roll(-1, -1) * above(good) * below(good)
            gx = (sst.roll(-1, -1) - sst.roll(1, -1)) / (2 * step_x) * stencil
            gy = (below(sst) - above(sst)) / (2 * step_y) * stencil
            magnitude = (gx.square() + gy.square() + 1e-16).sqrt()
            strength = (magnitude - 1e-8).clamp_min(0)
            reference = (strength.sum((-2, -1), keepdim=True)
                         / stencil.sum((-2, -1), keepdim=True).clamp_min(1)).detach()
            confidence = strength / (strength + reference.clamp_min(1e-6)) * stencil
            nx = torch.where(strength > 1e-8, gx / magnitude, torch.ones_like(gx))
            ny = torch.where(strength > 1e-8, gy / magnitude, torch.zeros_like(gy))
            return nx, ny, confidence, step_x, step_y, ocean_low

    def coefficients(self, context: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=context.device.type, enabled=False):
            return self.geometry_head(context.float())

    def sampling_geometry(self, coefficients: torch.Tensor, fraction: torch.Tensor, frame):
        nx, ny, confidence, step_x, step_y, _ = frame
        b, _, h, w = coefficients.shape
        z = fraction.float() if self.depth_conditioning else fraction.float() * 0
        basis = torch.stack((torch.ones_like(z), z, z.square()))
        fields = (coefficients.reshape(b, 5, 3, h, w) * basis.view(1, 1, 3, 1, 1)).sum(2)
        rt = self.tangent_radius_km * (1 + 0.5 * fields[:, 0:1].tanh())
        rn = self.normal_radius_km * (1 + 0.5 * fields[:, 1:2].tanh())
        angle = self.max_rotation * fields[:, 2:3].tanh()
        cosine, sine = angle.cos(), angle.sin()
        normal_x, normal_y = cosine * nx - sine * ny, sine * nx + cosine * ny
        tangent_x, tangent_y = -normal_y, normal_x
        shift_x = self.max_shift_km * fields[:, 3:4].tanh()
        shift_y = self.max_shift_km * fields[:, 4:5].tanh()
        # Ordering: center, +/-t, +/-2t, +/-n, +/-2n.
        offsets_x, offsets_y = [], []
        for direction, radius, distance in ((normal_x, rn, 0), (tangent_x, rt, 1),
                (tangent_x, rt, -1), (tangent_x, rt, 2), (tangent_x, rt, -2),
                (normal_x, rn, 1), (normal_x, rn, -1), (normal_x, rn, 2), (normal_x, rn, -2)):
            offsets_x.append((shift_x + distance * radius * direction) / step_x)
        for direction, radius, distance in ((normal_y, rn, 0), (tangent_y, rt, 1),
                (tangent_y, rt, -1), (tangent_y, rt, 2), (tangent_y, rt, -2),
                (normal_y, rn, 1), (normal_y, rn, -1), (normal_y, rn, 2), (normal_y, rn, -2)):
            offsets_y.append((shift_y + distance * radius * direction) / step_y)
        dx, dy = torch.cat(offsets_x, 1), torch.cat(offsets_y, 1)
        # Scale both components together: cap the Chebyshev radius without
        # rotating the physical direction, including at high latitudes.
        divisor = torch.maximum(dx.abs(), dy.abs()).div(self.max_offset_cells).clamp_min(1)
        return dx / divisor, dy / divisor, confidence

    def forward(self, feature, coefficients, fraction, frame):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            dx, dy, confidence = self.sampling_geometry(coefficients, fraction, frame)
            ocean = frame[-1]
            # Mask interpolation corners before sampling; land values cannot
            # contaminate ocean samples. Empty samples fall back to center.
            coverage = periodic_bilinear_sample(ocean, dx, dy)
            samples = periodic_bilinear_sample(feature.float() * ocean, dx, dy)
            center = feature.float().unsqueeze(2)
            samples = torch.where(coverage > 1e-6,
                                  samples / coverage.clamp_min(1e-6), center)
            samples = center + coverage.clamp(0, 1) * (samples - center)
            along = samples[:, :, 1:5].mean(2)
            cross_even = samples[:, :, 5:9].mean(2)
            cross_odd = 0.5 * ((samples[:, :, 5] - samples[:, :, 6])
                               + 0.5 * (samples[:, :, 7] - samples[:, :, 8]))
            features = torch.cat((samples[:, :, 0], along, cross_even, cross_odd), 1)
            correction = self.projection(features) * confidence * ocean
        return correction.to(feature.dtype)
