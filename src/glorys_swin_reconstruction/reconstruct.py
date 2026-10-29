from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from netCDF4 import Dataset

from .config import ensure_dir, load_config, require_data_paths
from .data import (
    MODEL_INPUT_CHANNELS,
    SURFACE_CHANNELS,
    HISTORY_MONTHS,
    history_window_indices,
    read_model_input,
    validate_model_data_contract,
)
from .metrics import MaskedMetricAccumulator
from .metric_io import coverage_metadata, evaluation_metadata, monthly_record, write_json, write_monthly
from .model import GeoSwinUNet
from .norm import DepthNorm
from .time_utils import month_keys, month_numbers, split_indices
from .train import pick_device
from .transfer import validate_labels, validate_stage


def _copy_attrs(source: xr.DataArray, destination) -> None:
    for name, value in source.attrs.items():
        if name != "_FillValue":
            destination.setncattr(name, value)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct physical temperature fields.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--max-months", type=int)
    parser.add_argument("--output-dir", help="Override reconstruction output directory.")
    args = parser.parse_args()

    if args.max_months is not None and args.max_months <= 0:
        parser.error("--max-months must be positive")

    cfg = load_config(args.config)
    validate_stage(cfg)
    if cfg['data'].get(f'{args.split}_range') is None:
        parser.error(f'The {args.split} split is disabled for this stage.')
    require_data_paths(cfg)
    validate_model_data_contract(cfg["model"])
    checkpoint = Path(args.checkpoint or Path(cfg["training"]["output_dir"]) / "best.pt")
    output_dir = ensure_dir(args.output_dir or cfg["training"]["output_dir"])
    if args.max_months is not None and args.output_dir is None:
        output_dir = ensure_dir(output_dir / "preview")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    device = pick_device(str(cfg["training"].get("device", "auto")))
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["config"]["data"] != cfg["data"] or state["config"]["model"] != cfg["model"]:
        raise ValueError("Reconstruction requires the checkpoint's data and model configuration.")
    model_cfg = state.get("config", cfg)["model"]
    validate_model_data_contract(model_cfg)
    model = GeoSwinUNet(**model_cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    data_cfg = cfg["data"]
    norm = DepthNorm.from_config(data_cfg)
    with (
        xr.open_dataset(data_cfg["surface_path"], decode_times=False, mask_and_scale=False) as surface_ds,
        xr.open_dataset(data_cfg["label_path"], decode_times=False, mask_and_scale=False) as label_ds,
        xr.open_dataset(data_cfg["domain_mask_path"], decode_times=False, mask_and_scale=False) as domain_ds,
    ):
        indices = split_indices(surface_ds, data_cfg, args.split)
        validate_labels(cfg, label_ds, indices)
        all_keys = month_keys(surface_ds, data_cfg.get("time_text_var"))
        coverage_metadata(all_keys[indices], data_cfg, args.split)
        if args.max_months is not None:
            indices = indices[: int(args.max_months)]
        if indices.size == 0:
            raise ValueError(f"Split {args.split!r} contains no requested months.")
        coverage = coverage_metadata(all_keys[indices], data_cfg, args.split,
                                     partial=args.max_months is not None)
        windows = history_window_indices(all_keys, indices)
        records = []
        surface_input = surface_ds[data_cfg.get("surface_var", "input")]
        channels = tuple(map(str, surface_input["channel"].values))
        if channels != SURFACE_CHANNELS:
            raise ValueError(f"Surface channels must be exactly {SURFACE_CHANNELS}.")
        months = month_numbers(surface_ds, data_cfg.get("time_text_var"))

        depth_values = np.asarray(label_ds["depth"].values, dtype=np.float64)
        for coord in ("time", "lat", "lon"):
            if not np.array_equal(surface_ds[coord].values, label_ds[coord].values):
                raise ValueError(f"Surface and label coordinates differ for {coord}.")
        norm.validate_depth_count(depth_values.size)
        if int(model_cfg["out_depths"]) != depth_values.size:
            raise ValueError("Model output depth count does not match label data.")
        norm_mean, norm_std = norm.numpy_mean_std()
        domain_mask = domain_ds[data_cfg.get("domain_mask_var", "valid_ocean_mask")]
        if tuple(domain_mask.dims) != ("depth", "lat", "lon"):
            raise ValueError("Domain mask must use (depth, lat, lon).")
        for coord in ("depth", "lat", "lon"):
            if not np.array_equal(domain_ds[coord].values, label_ds[coord].values):
                raise ValueError(f"Domain-mask and label coordinates differ for {coord}.")
        domain_values = np.asarray(domain_mask.values, dtype=np.float32)
        if not np.isfinite(domain_values).all() or not np.isin(domain_values, (0, 1)).all():
            raise ValueError("Domain mask must be finite and binary.")
        surface_depth = np.flatnonzero(
            np.isclose(
                np.asarray(domain_ds["depth"].values, dtype=np.float64),
                float(data_cfg.get("surface_mask_depth_m", 0.0)),
                atol=1e-6,
                rtol=0.0,
            )
        )
        if surface_depth.size != 1:
            raise ValueError("Domain mask must contain one configured surface depth.")
        surface_ocean_mask = domain_values[int(surface_depth[0])][None]

        output = output_dir / f"reconstruction_{args.split}.nc"
        metrics_path = output_dir / f"metrics_{args.split}.json"
        metrics = MaskedMetricAccumulator(norm, depth_values)
        with Dataset(output, "w") as root:
            root.createDimension("time", len(indices))
            root.createDimension("depth", label_ds.sizes["depth"])
            root.createDimension("lat", surface_ds.sizes["lat"])
            root.createDimension("lon", surface_ds.sizes["lon"])
            coordinate_sources = {
                "time": surface_ds,
                "depth": label_ds,
                "lat": surface_ds,
                "lon": surface_ds,
            }
            for name, source_ds in coordinate_sources.items():
                source = source_ds[name]
                variable = root.createVariable(name, source.dtype, (name,))
                variable[:] = source.isel(time=indices).values if name == "time" else source.values
                _copy_attrs(source, variable)

            prediction_var = root.createVariable(
                f"pred_{data_cfg.get('label_var', 'temperature')}",
                "f4",
                ("time", "depth", "lat", "lon"),
                zlib=True,
                complevel=4,
                chunksizes=(1, label_ds.sizes["depth"], min(128, label_ds.sizes["lat"]), min(256, label_ds.sizes["lon"])),
                fill_value=np.float32(np.nan),
            )
            prediction_var.long_name = "Swin-reconstructed ocean subsurface temperature"
            prediction_var.units = norm.units
            prediction_var.source_space = "inverse-transformed physical temperature"
            prediction_var.valid_domain = "static valid_ocean_mask; invalid cells are NaN"

            for position, index in enumerate(indices):
                model_input, surface_valid = read_model_input(
                    surface_input, windows[position], surface_ocean_mask,
                    int(months[int(index)]),
                )
                prediction_norm = (
                    model(
                        torch.from_numpy(model_input).unsqueeze(0).to(device),
                        torch.from_numpy(surface_valid).unsqueeze(0).to(device),
                    ).squeeze(0).float().cpu().numpy()
                )

                target_raw = label_ds[data_cfg.get("label_var", "temperature")].isel(time=int(index)).values
                label_mask = label_ds[data_cfg.get("mask_var", "valid_label_mask")].isel(time=int(index)).values
                valid = (label_mask > 0) & (domain_values > 0) & np.isfinite(target_raw)
                target_norm = np.nan_to_num(
                    target_raw, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
                monthly = MaskedMetricAccumulator(norm, depth_values)
                monthly.update(
                    torch.from_numpy(prediction_norm).unsqueeze(0),
                    torch.from_numpy(target_norm).unsqueeze(0),
                    torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),
                )
                metrics.merge(monthly)
                records.append(monthly_record(str(all_keys[int(index)]), args.split, monthly))
                prediction_c = prediction_norm * norm_std + norm_mean
                prediction_var[position] = np.where(
                    domain_values > 0, prediction_c, np.nan
                ).astype(np.float32)
                print(f"reconstructed {position + 1}/{len(indices)}")

            root.model = "glorys_swin_reconstruction.GeoSwinUNet"
            root.checkpoint = str(checkpoint.resolve())
            root.split = args.split
            root.checkpoint_epoch = int(state.get("epoch", -1))
            root.selection_metric = str(state.get("selection_metric", "macro_nrmse_depth"))
            root.input_channels = ",".join(MODEL_INPUT_CHANNELS)
            root.history_months = HISTORY_MONTHS
            root.history_order = "current month only: 11 channels; no lag inputs"
            root.history_missing_policy = "only target month required; non-finite values filled with standardized zero"
            root.longitude_boundary = "periodic"
            root.latitude_boundary = "non-periodic"
            root.domain_mask_path = str(Path(data_cfg["domain_mask_path"]).resolve())

    result = metrics.compute()
    result.update(
        {
            "split": args.split,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_epoch": int(state.get("epoch", -1)),
            "selection_metric": str(state.get("selection_metric", "macro_nrmse_depth")),
        }
    )
    metadata = evaluation_metadata(data_cfg, norm, split=args.split, **coverage,
                                   checkpoint=str(checkpoint.resolve()),
                                   checkpoint_epoch=int(state.get("epoch", -1)),
                                   selection_metric=str(state.get("selection_metric", "macro_nrmse_depth")),
                                   evaluation_source="online_reconstruction",
                                   reconstruction_path=str(output.resolve()))
    result.update(metadata)
    write_json(metrics_path, result)
    for path in write_monthly(output_dir, args.split, records, metadata, result):
        print(f"saved: {path}")
    print(f"saved: {output}\nsaved: {metrics_path}")


if __name__ == "__main__":
    main()
