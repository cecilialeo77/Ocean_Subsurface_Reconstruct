from __future__ import annotations

import numpy as np
import torch

from .norm import DepthNorm


class MaskedMetricAccumulator:
    """Streaming physical-space metrics for sparse depth-resolved labels."""

    def __init__(self, norm: DepthNorm, depth_values: np.ndarray) -> None:
        self.norm = norm
        self.depth_values = np.asarray(depth_values, dtype=np.float64)
        count = self.depth_values.size
        norm.validate_depth_count(count)
        self.count = np.zeros(count, dtype=np.int64)
        self.sse = np.zeros(count, dtype=np.float64)
        self.sae = np.zeros(count, dtype=np.float64)
        self.sum_y = np.zeros(count, dtype=np.float64)
        self.sum_y2 = np.zeros(count, dtype=np.float64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError("pred, target and mask must have identical shapes.")
        mean = torch.as_tensor(self.norm.mean, device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
        std = torch.as_tensor(self.norm.std, device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
        pred_c = pred * std + mean
        target_c = target * std + mean
        self.update_physical(pred_c, target_c, mask)

    @torch.no_grad()
    def update_physical(self, pred_c: torch.Tensor, target_c: torch.Tensor, mask: torch.Tensor) -> None:
        """Accumulate temperatures already expressed in physical units."""
        if pred_c.shape != target_c.shape or pred_c.shape != mask.shape:
            raise ValueError("pred, target and mask must have identical shapes.")
        if pred_c.ndim != 4 or pred_c.shape[1] != self.count.size:
            raise ValueError("Expected (batch, depth, lat, lon) with configured depths.")
        valid_all = mask > 0
        if not torch.isfinite(pred_c[valid_all]).all() or not torch.isfinite(target_c[valid_all]).all():
            raise ValueError("Non-finite prediction or target at a valid evaluation point.")
        for depth in range(pred_c.shape[1]):
            valid = mask[:, depth] > 0
            if not torch.any(valid):
                continue
            p = pred_c[:, depth][valid].double()
            y = target_c[:, depth][valid].double()
            error = p - y
            self.count[depth] += int(valid.sum().item())
            self.sse[depth] += float(error.square().sum().item())
            self.sae[depth] += float(error.abs().sum().item())
            self.sum_y[depth] += float(y.sum().item())
            self.sum_y2[depth] += float(y.square().sum().item())

    def merge(self, other: "MaskedMetricAccumulator") -> None:
        """Merge sufficient statistics, never averages of monthly metrics."""
        if (not np.array_equal(self.depth_values, other.depth_values)
                or not np.array_equal(self.norm.mean, other.norm.mean)
                or not np.array_equal(self.norm.std, other.norm.std)
                or self.norm.units != other.norm.units):
            raise ValueError("Cannot merge different depth/normalization contracts.")
        for name in ("count", "sse", "sae", "sum_y", "sum_y2"):
            getattr(self, name)[:] += getattr(other, name)

    def statistics(self) -> list[dict[str, object]]:
        return [dict(depth_index=i, depth_m=float(depth), count=int(self.count[i]),
                     sse=float(self.sse[i]), sae=float(self.sae[i]),
                     sum_y=float(self.sum_y[i]), sum_y2=float(self.sum_y2[i]))
                for i, depth in enumerate(self.depth_values)]

    @staticmethod
    def _r2(sse: float, count: int, sum_y: float, sum_y2: float) -> float:
        if count <= 0:
            return float("nan")
        sst = sum_y2 - sum_y * sum_y / count
        return float("nan") if sst <= 0 else 1.0 - sse / sst

    def compute(self) -> dict[str, object]:
        valid = self.count > 0
        rmse = np.full(self.count.size, np.nan)
        mae = np.full(self.count.size, np.nan)
        r2 = np.full(self.count.size, np.nan)
        rmse[valid] = np.sqrt(self.sse[valid] / self.count[valid])
        mae[valid] = self.sae[valid] / self.count[valid]
        for index in np.flatnonzero(valid):
            r2[index] = self._r2(
                float(self.sse[index]), int(self.count[index]),
                float(self.sum_y[index]), float(self.sum_y2[index]),
            )

        pooled_count = int(self.count.sum())
        pooled_sse = float(self.sse.sum())
        pooled_sae = float(self.sae.sum())
        pooled_sum_y = float(self.sum_y.sum())
        pooled_sum_y2 = float(self.sum_y2.sum())
        per_depth = [
            {
                "count": int(self.count[i]),
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
                "r2": float(r2[i]),
                "depth_index": int(i),
                "depth_m": float(self.depth_values[i]),
                "nrmse_train_std": float(rmse[i] / self.norm.std[i]),
            }
            for i in range(self.count.size)
        ]
        finite_r2 = np.isfinite(r2)
        macro_nrmse = np.mean(rmse[valid] / self.norm.std[valid]) if np.any(valid) else np.nan
        return {
            "count": pooled_count,
            "pooled_rmse": float(np.sqrt(pooled_sse / pooled_count)) if pooled_count else float("nan"),
            "pooled_mae": pooled_sae / pooled_count if pooled_count else float("nan"),
            "pooled_r2": self._r2(pooled_sse, pooled_count, pooled_sum_y, pooled_sum_y2),
            "mean_depth_rmse": float(np.mean(rmse[valid])) if np.any(valid) else float("nan"),
            "mean_depth_mae": float(np.mean(mae[valid])) if np.any(valid) else float("nan"),
            "mean_depth_r2": float(np.mean(r2[finite_r2])) if np.any(finite_r2) else float("nan"),
            "macro_nrmse_depth": float(macro_nrmse),
            "units": self.norm.units,
            "per_depth": per_depth,
            "definitions": {
                "pooled": "All valid months, depths and grid cells are pooled before one calculation.",
                "per_depth": "All valid months and grid cells are pooled independently by depth.",
                "macro_nrmse_depth": "Mean per-depth RMSE divided by training-set depth standard deviation.",
                "weighting": "Unweighted valid grid points; no area or layer-thickness weighting.",
            },
        }
