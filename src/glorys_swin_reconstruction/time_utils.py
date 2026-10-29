from __future__ import annotations

import re

import numpy as np
import xarray as xr


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def month_keys(ds: xr.Dataset, time_text_var: str | None = None) -> np.ndarray:
    if time_text_var and time_text_var in ds:
        values = ds[time_text_var].values
    else:
        values = ds["time"].values
        if np.asarray(values).dtype.kind in "iuf":
            try:
                values = xr.decode_cf(ds[["time"]])["time"].values
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
    keys: list[str] = []
    for value in values:
        match = re.search(r"((?:19|20)\d{2})[-/]?(0[1-9]|1[0-2])", _text(value))
        if match is None:
            raise ValueError(f"Cannot infer YYYY-MM from time value {value!r}.")
        keys.append(f"{match.group(1)}-{match.group(2)}")
    return np.asarray(keys)


def month_numbers(ds: xr.Dataset, time_text_var: str | None = None) -> np.ndarray:
    return np.asarray([int(key[-2:]) for key in month_keys(ds, time_text_var)], dtype=np.int8)


def month_cyclic_maps(month: int, height: int, width: int) -> np.ndarray:
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {month}")
    theta = 2.0 * np.pi * (month - 1) / 12.0
    values = np.asarray([np.sin(theta), np.cos(theta)], dtype=np.float32)
    return np.broadcast_to(values[:, None, None], (2, height, width))


def split_indices(ds: xr.Dataset, data_cfg: dict, split: str) -> np.ndarray:
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported split: {split}")
    span = data_cfg.get(f"{split}_range")
    if span is None:
        raise ValueError(f"The {split} split is disabled for this stage.")
    start, end = map(str, span)
    keys = month_keys(ds, data_cfg.get("time_text_var"))
    return np.flatnonzero((keys >= start) & (keys <= end)).astype(np.int32)
