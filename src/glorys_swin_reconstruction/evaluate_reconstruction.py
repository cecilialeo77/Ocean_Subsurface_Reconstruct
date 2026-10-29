"""Evaluate saved physical-temperature fields without loading a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .config import load_config
from .metric_io import (coverage_metadata, evaluation_metadata, monthly_record,
                        write_json, write_monthly, OVERALL_FIELDS, DEPTH_FIELDS)
from .metrics import MaskedMetricAccumulator
from .norm import DepthNorm
from .time_utils import month_keys, split_indices
from .transfer import validate_labels, validate_stage


def compare_summaries(reference: dict, actual: dict) -> dict:
    """Check count exactly and numeric metrics within physical float32 tolerance."""
    checks = []
    def compare(prefix, left, right, fields):
        for key in fields:
            a, b = left[key], right[key]
            a = float("nan") if a is None else a
            b = float("nan") if b is None else b
            exact = key in ("count", "depth_index", "depth_m")
            ok = a == b if exact else bool(np.isclose(a, b, rtol=1e-6, atol=1e-7, equal_nan=True))
            checks.append(dict(field=prefix + key, reference=a, actual=b,
                               absolute_difference=abs(a-b), passed=bool(ok)))
    compare("", reference, actual, OVERALL_FIELDS)
    if len(reference["per_depth"]) != len(actual["per_depth"]):
        return dict(passed=False, reason="Different depth counts")
    for i, (left, right) in enumerate(zip(reference["per_depth"], actual["per_depth"])):
        compare(f"per_depth[{i}].", left, right, DEPTH_FIELDS)
    return dict(passed=all(item["passed"] for item in checks), rtol=1e-6, atol=1e-7, checks=checks)


def evaluate(config_path, reconstruction_path, split="test", output_dir=None, max_months=None):
    if max_months is not None and max_months <= 0:
        raise ValueError("max_months must be positive")
    cfg = load_config(config_path)
    validate_stage(cfg)
    if cfg['data'].get(f'{split}_range') is None:
        raise ValueError(f'The {split} split is disabled for this stage.')
    data_cfg = cfg["data"]
    source = Path(reconstruction_path).resolve()
    destination = Path(output_dir) if output_dir else source.parent / ("preview" if max_months else "")
    norm = DepthNorm.from_config(data_cfg)
    records = []
    with (xr.open_dataset(source) as predicted,
          xr.open_dataset(data_cfg["label_path"], decode_times=False, mask_and_scale=False) as labels,
          xr.open_dataset(data_cfg["domain_mask_path"], decode_times=False, mask_and_scale=False) as domain):
        if predicted.attrs.get("split") not in (None, split):
            raise ValueError("Saved reconstruction split differs from requested split.")
        prediction = predicted[f"pred_{data_cfg.get('label_var', 'temperature')}"]
        target = labels[data_cfg.get("label_var", "temperature")]
        label_mask = labels[data_cfg.get("mask_var", "valid_label_mask")]
        domain_mask = domain[data_cfg.get("domain_mask_var", "valid_ocean_mask")]
        if prediction.dims != ("time", "depth", "lat", "lon") or target.dims != prediction.dims or label_mask.dims != target.dims:
            raise ValueError("Prediction, labels and label masks must use (time, depth, lat, lon).")
        if domain_mask.dims != ("depth", "lat", "lon"):
            raise ValueError("Domain mask must use (depth, lat, lon).")
        if prediction.attrs.get("units") != norm.units:
            raise ValueError("Prediction units differ from physical normalization units.")
        for coord in ("depth", "lat", "lon"):
            if not np.array_equal(predicted[coord].values, labels[coord].values) or not np.array_equal(domain[coord].values, labels[coord].values):
                raise ValueError(f"Reconstruction/label/domain coordinates differ for {coord}.")
        keys = month_keys(predicted)
        coverage_metadata(keys, data_cfg, split, partial=max_months is not None)
        label_indices = split_indices(labels, data_cfg, split)
        validate_labels(cfg, labels, label_indices)
        label_keys = month_keys(labels, data_cfg.get("time_text_var"))[label_indices]
        coverage_metadata(label_keys, data_cfg, split)
        if max_months is not None:
            keys = keys[:max_months]
        coverage = coverage_metadata(keys, data_cfg, split, partial=max_months is not None)
        lookup = dict(zip(label_keys, label_indices))
        decoded_time = xr.decode_cf(labels[["time"]])["time"].values
        selected = np.asarray([lookup[key] for key in keys], dtype=int)
        if not np.array_equal(predicted.time.values[:len(keys)], decoded_time[selected]):
            raise ValueError("Prediction and label timestamps differ, despite month alignment.")
        domain_values = np.asarray(domain_mask.values, dtype=np.float32)
        if not np.isfinite(domain_values).all() or not np.isin(domain_values, (0, 1)).all():
            raise ValueError("Domain mask must be finite and binary.")
        depths = labels.depth.values
        summary_accumulator = MaskedMetricAccumulator(norm, depths)
        mean, std = norm.numpy_mean_std()
        for position, (key, index) in enumerate(zip(keys, selected)):
            raw = target.isel(time=int(index)).values
            mask = (label_mask.isel(time=int(index)).values > 0) & (domain_values > 0) & np.isfinite(raw)
            target_c = np.nan_to_num(raw, nan=0., posinf=0., neginf=0.).astype(np.float32) * std + mean
            pred_c = np.asarray(prediction.isel(time=position).values, dtype=np.float32)
            monthly = MaskedMetricAccumulator(norm, depths)
            monthly.update_physical(torch.from_numpy(pred_c).unsqueeze(0),
                                    torch.from_numpy(target_c).unsqueeze(0),
                                    torch.from_numpy(mask).unsqueeze(0))
            summary_accumulator.merge(monthly)
            records.append(monthly_record(str(key), split, monthly))
            print(f"evaluated {key} ({position + 1}/{len(keys)})", flush=True)
        checkpoint = predicted.attrs.get("checkpoint")
        epoch = predicted.attrs.get("checkpoint_epoch")
        selection = predicted.attrs.get("selection_metric", "macro_nrmse_depth")
        historical_path = source.parent / f"metrics_{split}.json"
        reference = None
        if historical_path.exists():
            reference = json.loads(historical_path.read_text(encoding="utf-8"))
            if epoch is None and checkpoint == reference.get("checkpoint"):
                epoch = reference.get("checkpoint_epoch")
        metadata = evaluation_metadata(data_cfg, norm, split=split, **coverage,
                                       checkpoint=checkpoint, checkpoint_epoch=epoch,
                                       selection_metric=selection, evaluation_source="saved_physical_netcdf",
                                       reconstruction_path=str(source))
    result = summary_accumulator.compute()
    result.update(metadata)
    if reference is not None and coverage["is_complete"] and not coverage["is_preview"]:
        result["historical_comparison"] = dict(reference_path=str(historical_path),
                                               **compare_summaries(reference, result))
    summary_path = destination / f"metrics_{split}_reevaluated.json"
    write_json(summary_path, result)
    paths = write_monthly(destination, split, records, metadata, result)
    for path in [summary_path, *paths]:
        print(f"saved: {path}", flush=True)
    if "historical_comparison" in result and not result["historical_comparison"]["passed"]:
        raise ValueError(f"Historical metrics comparison failed; inspect {summary_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reconstruction", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-months", type=int)
    args = parser.parse_args()
    evaluate(args.config, args.reconstruction, args.split, args.output_dir, args.max_months)


if __name__ == "__main__":
    main()
