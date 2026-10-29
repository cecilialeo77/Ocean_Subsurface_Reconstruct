"""Shared monthly metric schema, coverage validation and atomic file writers."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .metrics import MaskedMetricAccumulator

OVERALL_FIELDS = ["count", "pooled_rmse", "pooled_mae", "pooled_r2",
                  "mean_depth_rmse", "mean_depth_mae", "mean_depth_r2", "macro_nrmse_depth"]
DEPTH_FIELDS = ["depth_index", "depth_m", "count", "rmse", "mae", "r2", "nrmse_train_std"]
TIME_FIELDS = ["time", "year", "month", "split"]


def coverage_metadata(keys, data_cfg: dict, split: str, *, partial: bool = False) -> dict:
    keys = list(map(str, keys))
    start, end = map(str, data_cfg[f"{split}_range"])
    expected = np.arange(np.datetime64(start, "M"), np.datetime64(end, "M") + 1).astype(str).tolist()
    if not expected or not keys:
        raise ValueError("Requested range and selected months must be non-empty.")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate months in evaluation data.")
    if keys != sorted(keys):
        raise ValueError("Evaluation months must be in chronological order.")
    if any(key not in expected for key in keys):
        raise ValueError("Evaluation months fall outside the configured split.")
    missing = [key for key in expected if key not in keys]
    if missing and not partial:
        raise ValueError(f"Missing months in full evaluation: {missing}")
    return dict(configured_range=[start, end], months=keys, month_count=len(keys),
                expected_month_count=len(expected), missing_months=missing,
                is_complete=not missing, is_preview=partial)


def monthly_record(key: str, split: str, metrics: MaskedMetricAccumulator) -> dict:
    result = metrics.compute()
    result.pop("definitions")
    return dict(time=key, year=int(key[:4]), month=int(key[5:7]), split=split,
                **result, sufficient_statistics=metrics.statistics())


def evaluation_metadata(data_cfg: dict, norm, **extra) -> dict:
    return dict(schema_version=1, units=norm.units,
                sources={key: str(Path(data_cfg[key]).resolve()) for key in
                         ("label_path", "domain_mask_path", "label_norm_path")},
                variables={key: data_cfg.get(key, default) for key, default in
                           (("label_var", "temperature"), ("mask_var", "valid_label_mask"),
                            ("domain_mask_var", "valid_ocean_mask"))},
                normalization=dict(mean=norm.mean.tolist(), std=norm.std.tolist(),
                                   reference="training-set depth standard deviation"),
                valid_mask="label_mask > 0 AND domain_mask > 0 AND finite standardized target",
                prediction_policy="Non-finite predictions on valid evaluation points are errors.",
                **extra)


def _clean(value):
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n")


def write_monthly(output_dir: Path, split: str, records: list[dict], metadata: dict,
                  summary: dict) -> list[Path]:
    keys = [row["time"] for row in records]
    if keys != metadata["months"] or len(set(keys)) != len(keys):
        raise ValueError("Monthly records disagree with coverage metadata.")
    paths = [Path(output_dir) / f"metrics_{split}_monthly{suffix}"
             for suffix in (".json", "_overall.csv", "_depth.csv")]
    # CSVs and JSON always derive from the same records. Publish JSON last.
    for path, fields, depth in ((paths[1], TIME_FIELDS + OVERALL_FIELDS, False),
                                (paths[2], TIME_FIELDS + DEPTH_FIELDS, True)):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for record in records:
            base = {key: record[key] for key in TIME_FIELDS}
            rows = sorted(record["per_depth"], key=lambda row: row["depth_m"]) if depth else [record]
            for row in rows:
                writer.writerow(_clean({key: base[key] if key in base else row[key] for key in fields}))
        atomic_text(path, buffer.getvalue())
    definitions = dict(summary["definitions"])
    definitions.update(monthly_overall="Within one YYYY-MM, pool valid depths and grid cells.",
                       monthly_per_depth="Within one YYYY-MM and one depth, pool valid grid cells.",
                       sufficient_statistics="Per month and depth: count, SSE, SAE, sum(y), sum(y^2); merge before recomputing metrics.",
                       undefined="JSON null and empty CSV cells; empty depths retain count=0.")
    write_json(paths[0], dict(**metadata, definitions=definitions, records=records))
    return paths
