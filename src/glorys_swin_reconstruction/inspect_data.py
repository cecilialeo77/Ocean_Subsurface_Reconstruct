from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from .config import load_config, require_data_paths
from .data import MODEL_INPUT_CHANNELS, FullFieldDataset, history_window_indices, spec_from_config, validate_model_data_contract
from .norm import DepthNorm
from .time_utils import month_keys, split_indices
from .transfer import validate_coverage, validate_stage, stage_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all reconstruction data contracts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_stage(cfg)
    require_data_paths(cfg)
    validate_model_data_contract(cfg["model"])
    spec = spec_from_config(cfg["data"])
    with xr.open_dataset(spec.surface_path, decode_times=False, mask_and_scale=False) as ds:
        keys = month_keys(ds, spec.time_text_var)
        print(f"surface_sizes={dict(ds.sizes)}")
        print(f"time_range={keys[0]}..{keys[-1]}")
        split_map = {
            split: split_indices(ds, cfg["data"], split)
            for split in ("train", "val", "test")
            if cfg['data'].get(f'{split}_range') is not None
        }
    print(f"stage={stage_name(cfg)}")
    if cfg['data'].get('test_range') is None:
        print('test: disabled (IPRC占位月份不参与测试)')
    print(f"input_channels={MODEL_INPUT_CHANNELS}")
    for split, indices in split_map.items():
        if indices.size == 0:
            raise ValueError(f"The {split} split contains no samples.")
        windows = history_window_indices(keys, indices)
        print(f"{split}: count={indices.size}, index_range={indices[0]}..{indices[-1]}")
        print(f"  first_input_months={keys[windows[0]].tolist()} target={keys[indices[0]]}")
        checked = FullFieldDataset(spec, indices)
        try:
            validate_coverage(checked, cfg, split)
        finally:
            checked.close()
    dataset = FullFieldDataset(spec, split_map["train"][:1])
    try:
        sample = dataset[0]
        norm = DepthNorm.from_config(cfg["data"])
        norm.validate_depth_count(dataset.depth_values.size)
        print(f"surface_shape={tuple(sample['surface'].shape)} dtype={sample['surface'].dtype}")
        print(f"target_shape={tuple(sample['target'].shape)} dtype={sample['target'].dtype}")
        print(f"valid_label_count={int(sample['mask'].sum().item())}")
        print(f"depths_m={np.array2string(dataset.depth_values, separator=', ')}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
