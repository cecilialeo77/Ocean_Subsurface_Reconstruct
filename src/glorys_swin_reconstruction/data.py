from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .time_utils import month_cyclic_maps, month_keys, month_numbers


SURFACE_CHANNELS = (
    "sst", "sss", "sla", "ussw", "vssw", "lat_sin", "lon_sin", "lon_cos"
)
# 保留该常量用于checkpoint签名及重建元数据；1表示仅使用目标月。
HISTORY_MONTHS = 1
CURRENT_INPUT_CHANNELS = SURFACE_CHANNELS + (
    "surface_ocean_mask", "month_sin", "month_cos"
)
MODEL_INPUT_CHANNELS = CURRENT_INPUT_CHANNELS


def history_window_indices(keys: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    """返回每个目标月自身的索引[t]；不要求存在任何前序月份。"""
    keys = np.asarray(keys)
    months = keys.astype("datetime64[M]").astype(np.int64)
    if np.any(np.diff(months) <= 0):
        raise ValueError("Surface months must be unique and strictly chronological.")
    targets = np.asarray(target_indices, dtype=np.int64)
    if targets.ndim != 1:
        raise ValueError("Target indices must be one-dimensional.")
    if np.any(targets < 0) or np.any(targets >= len(keys)):
        raise IndexError("A requested target index lies outside the surface dataset.")
    return targets[:, None]


@dataclass(frozen=True)
class FullFieldSpec:
    surface_path: str
    label_path: str
    domain_mask_path: str
    surface_var: str = "input"
    label_var: str = "temperature"
    mask_var: str = "valid_label_mask"
    domain_mask_var: str = "valid_ocean_mask"
    surface_mask_depth_m: float = 0.0
    time_text_var: str | None = None


def spec_from_config(data_cfg: dict) -> FullFieldSpec:
    return FullFieldSpec(
        surface_path=data_cfg["surface_path"],
        label_path=data_cfg["label_path"],
        domain_mask_path=data_cfg["domain_mask_path"],
        surface_var=data_cfg.get("surface_var", "input"),
        label_var=data_cfg.get("label_var", "temperature"),
        mask_var=data_cfg.get("mask_var", "valid_label_mask"),
        domain_mask_var=data_cfg.get("domain_mask_var", "valid_ocean_mask"),
        surface_mask_depth_m=float(data_cfg.get("surface_mask_depth_m", 0.0)),
        time_text_var=data_cfg.get("time_text_var"),
    )


def validate_model_data_contract(model_cfg: dict) -> None:
    expected_channels = len(MODEL_INPUT_CHANNELS)
    if int(model_cfg.get("in_channels", -1)) != expected_channels:
        raise ValueError(
            f"model.in_channels must be {expected_channels} for {MODEL_INPUT_CHANNELS}."
        )
    if int(model_cfg.get("out_depths", -1)) <= 0:
        raise ValueError("model.out_depths must be positive.")


def assemble_model_input(
    surface: np.ndarray,
    surface_ocean_mask: np.ndarray,
    month: int,
) -> np.ndarray:
    """仅组装当月11通道：8个海表/地理通道、海洋掩膜及2个月份编码。

    NaN/Inf填为标准化空间的0；当月5个动态变量的有效性另行提供给MoE。
    """
    if surface.ndim != 3 or surface.shape[0] != len(SURFACE_CHANNELS):
        raise ValueError(f"Expected surface shape (8, H, W), got {surface.shape}.")
    if surface_ocean_mask.shape != (1, *surface.shape[-2:]):
        raise ValueError(
            f"Expected surface mask shape {(1, *surface.shape[-2:])}, "
            f"got {surface_ocean_mask.shape}."
        )
    season = month_cyclic_maps(month, surface.shape[-2], surface.shape[-1])
    result = np.concatenate(
        (surface, surface_ocean_mask, season),
        axis=0,
    )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def read_model_input(
    surface: xr.DataArray, window: np.ndarray, surface_ocean_mask: np.ndarray, month: int,
) -> tuple[np.ndarray, np.ndarray]:
    """训练和推理共用的当月读取器；拒绝传入包含其他月份的窗口。"""
    window = np.asarray(window)
    if window.shape != (1,):
        raise ValueError('Current-month input requires a single target index [t].')
    raw = np.asarray(surface.isel(time=int(window[0])).values)
    validity = np.isfinite(raw[:5]).astype(np.float32)
    return assemble_model_input(raw, surface_ocean_mask, month), validity


class FullFieldDataset(Dataset):
    """只用目标月的海表信息预测同月次表层场，不读取历史海表。"""

    def __init__(
        self,
        spec: FullFieldSpec,
        time_indices: np.ndarray,
        max_samples: int | None = None,
    ) -> None:
        self.spec = spec
        self.time_indices = np.asarray(time_indices, dtype=np.int32)
        if max_samples is not None:
            self.time_indices = self.time_indices[: int(max_samples)]
        if self.time_indices.size == 0:
            raise ValueError("The requested split contains no samples.")
        self._surface_ds: xr.Dataset | None = None
        self._label_ds: xr.Dataset | None = None
        self._domain_ds: xr.Dataset | None = None
        self._open()

    def _open(self) -> None:
        self._surface_ds = xr.open_dataset(
            self.spec.surface_path, decode_times=False, mask_and_scale=False
        )
        self._label_ds = xr.open_dataset(
            self.spec.label_path, decode_times=False, mask_and_scale=False
        )
        self._domain_ds = xr.open_dataset(
            self.spec.domain_mask_path, decode_times=False, mask_and_scale=False
        )
        self.surface = self._surface_ds[self.spec.surface_var]
        self.label = self._label_ds[self.spec.label_var]
        self.mask = self._label_ds[self.spec.mask_var]
        self.domain_mask = self._domain_ds[self.spec.domain_mask_var]
        self.months = month_numbers(self._surface_ds, self.spec.time_text_var)
        self.depth_values = np.asarray(self._label_ds["depth"].values, dtype=np.float64)
        self._validate_contract()
        self.history_indices = history_window_indices(
            month_keys(self._surface_ds, self.spec.time_text_var), self.time_indices
        )

        depth_coord = np.asarray(self.domain_mask["depth"].values, dtype=np.float64)
        surface_depth = np.flatnonzero(
            np.isclose(depth_coord, self.spec.surface_mask_depth_m, atol=1e-6, rtol=0.0)
        )
        if surface_depth.size != 1:
            raise ValueError("Domain mask must contain exactly one configured surface depth.")
        self.surface_ocean_mask = np.asarray(
            self.domain_mask.isel(depth=int(surface_depth[0])).values,
            dtype=np.float32,
        )[None]

    def _validate_contract(self) -> None:
        if tuple(self.surface.dims) != ("time", "channel", "lat", "lon"):
            raise ValueError("Surface dimensions must be (time, channel, lat, lon).")
        expected_target_dims = ("time", "depth", "lat", "lon")
        if tuple(self.label.dims) != expected_target_dims or tuple(self.mask.dims) != expected_target_dims:
            raise ValueError("Label and label mask must use (time, depth, lat, lon).")
        if tuple(self.domain_mask.dims) != ("depth", "lat", "lon"):
            raise ValueError("Domain mask must use (depth, lat, lon).")
        channel_names = tuple(map(str, self.surface["channel"].values))
        if channel_names != SURFACE_CHANNELS:
            raise ValueError(f"Surface channels must be exactly {SURFACE_CHANNELS}.")
        for coord in ("time", "lat", "lon"):
            if not np.array_equal(self._surface_ds[coord].values, self._label_ds[coord].values):
                raise ValueError(f"Surface and label coordinates differ for {coord}.")
        for coord in ("depth", "lat", "lon"):
            if not np.array_equal(self._domain_ds[coord].values, self._label_ds[coord].values):
                raise ValueError(f"Domain mask and label coordinates differ for {coord}.")

        self.domain_mask_values = np.asarray(self.domain_mask.values, dtype=np.float32)
        if not np.isfinite(self.domain_mask_values).all():
            raise ValueError("Domain mask must be finite.")
        if not np.isin(self.domain_mask_values, (0, 1)).all():
            raise ValueError("Domain mask must contain only 0 and 1.")
        if np.any(self.time_indices < 0) or np.any(self.time_indices >= self.surface.sizes["time"]):
            raise IndexError("A requested time index lies outside the dataset.")
        if self.months.shape != (self.surface.sizes["time"],):
            raise ValueError("Parsed month count does not match the time dimension.")
        if self.depth_values.ndim != 1 or np.any(np.diff(self.depth_values) <= 0):
            raise ValueError("Depth coordinate must be one-dimensional and strictly increasing.")

    def __getstate__(self):
        state = dict(self.__dict__)
        for key in (
            "_surface_ds", "_label_ds", "_domain_ds", "surface", "label", "mask",
            "domain_mask", "domain_mask_values",
        ):
            state[key] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self) -> int:
        return int(self.time_indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        time_index = int(self.time_indices[item])
        model_input, surface_valid = read_model_input(
            self.surface, self.history_indices[item], self.surface_ocean_mask,
            int(self.months[time_index]),
        )
        target_raw = self.label.isel(time=time_index).values
        label_mask = self.mask.isel(time=time_index).values
        valid = (
            (label_mask > 0)
            & (self.domain_mask_values > 0)
            & np.isfinite(target_raw)
        )
        target = np.nan_to_num(
            target_raw, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        return {
            "surface": torch.from_numpy(model_input),
            "surface_valid": torch.from_numpy(surface_valid),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(valid.astype(np.float32)),
            "time_index": torch.tensor(time_index, dtype=torch.int32),
        }

    def close(self) -> None:
        for dataset in (self._surface_ds, self._label_ds, self._domain_ds):
            if dataset is not None:
                dataset.close()
