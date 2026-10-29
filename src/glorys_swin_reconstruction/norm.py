from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class DepthNorm:
    mean: np.ndarray
    std: np.ndarray
    units: str

    @classmethod
    def from_config(cls, data_cfg: dict) -> "DepthNorm":
        path = Path(data_cfg["label_norm_path"])
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        # GLORYS shared statistics store mean/std at the top level.
        item = payload if "mean" in payload and "std" in payload else payload[data_cfg.get("label_var", "temperature")]
        result = cls(
            mean=np.asarray(item["mean"], dtype=np.float32),
            std=np.asarray(item["std"], dtype=np.float32),
            units=str(item.get("raw_units", item.get("units", "degree_C"))),
        )
        result.validate_depth_count(len(result.mean))
        return result

    def validate_depth_count(self, count: int) -> None:
        if self.mean.shape != (count,) or self.std.shape != (count,):
            raise ValueError("Depth-normalization length does not match target depths.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Depth normalization contains non-finite values.")
        if np.any(self.std <= 0):
            raise ValueError("Every depth standard deviation must be positive.")

    def torch_mean_std(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, device=device).view(1, -1, 1, 1)
        std = torch.as_tensor(self.std, device=device).view(1, -1, 1, 1)
        return mean, std

    def numpy_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean[:, None, None], self.std[:, None, None]
