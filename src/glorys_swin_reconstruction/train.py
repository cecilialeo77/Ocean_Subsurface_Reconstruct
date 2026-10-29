from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader

from .config import ensure_dir, load_config, require_data_paths
from .data import FullFieldDataset, spec_from_config, validate_model_data_contract
from .losses import depth_balanced_masked_mse_loss, vertical_gradient_loss, vertical_gradient_transport_loss
from .metrics import MaskedMetricAccumulator
from .model import GeoSwinUNet
from .moe_climatology import fit_moe_statistics
from .norm import DepthNorm
from .optimization import (build_optimizer, build_scheduler, group_summary,
                           learning_rates, validate_optimizer_resume)
from .time_utils import split_indices
from .transfer import (data_signature, initialize_weights, stage_name,
                       validate_coverage, validate_stage)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_dataset(
    cfg: dict,
    split: str,
    max_samples: int | None = None,
) -> FullFieldDataset:
    spec = spec_from_config(cfg["data"])
    with xr.open_dataset(spec.surface_path, decode_times=False, mask_and_scale=False) as ds:
        indices = split_indices(ds, cfg["data"], split)
    dataset = FullFieldDataset(spec, indices, max_samples=max_samples)
    try:
        validate_coverage(dataset, cfg, split, partial=max_samples is not None)
    except Exception:
        dataset.close()
        raise
    return dataset


def build_model(model_cfg: dict) -> GeoSwinUNet:
    validate_model_data_contract(model_cfg)
    return GeoSwinUNet(**model_cfg)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    vertical_weight: float,
    depths_m: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    vertical_reference_distance_m: float,
    grad_clip_norm: float,
    metric_norm: DepthNorm | None = None,
    metric_depths: np.ndarray | None = None,
    # 位置梯度损失
    vertical_transport_weight: float = 0.0,
    vertical_transport_reference_distance_m: float = 100.0,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "value_loss": 0.0, "vertical_loss": 0.0, "vertical_transport_loss": 0.0}
    sample_count = 0
    metrics = (
        MaskedMetricAccumulator(metric_norm, metric_depths)
        if metric_norm is not None and metric_depths is not None
        else None
    )
    for batch in loader:
        surface, target, mask = (
            batch[key].to(device, non_blocking=True)
            for key in ("surface", "target", "mask")
        )
        surface_valid = batch["surface_valid"].to(device, non_blocking=True)
        if not torch.isfinite(mask).all() or torch.any(mask.sum(dim=(1,2,3)) == 0):
            raise ValueError("A sampled month has no valid labels; refuse zero-loss placeholder training.")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model(surface, surface_valid)
            value_loss = depth_balanced_masked_mse_loss(prediction, target, mask)
            # 数值梯度损失
            if vertical_weight > 0:
                gradient_loss = vertical_gradient_loss(
                    prediction,
                    target,
                    mask,
                    depths_m,
                    norm_mean,
                    norm_std,
                    reference_distance_m=vertical_reference_distance_m,
                )
            else:
                gradient_loss = prediction.sum() * 0.0

            # 位置梯度损失
            if vertical_transport_weight > 0:
                transport_loss = vertical_gradient_transport_loss(
                    prediction,
                    target,
                    mask,
                    depths_m,
                    norm_mean,
                    norm_std,
                )
            else:
                transport_loss = prediction.sum() * 0.0

            # 总损失
            loss = value_loss + vertical_weight * gradient_loss + vertical_transport_weight * transport_loss

        if not torch.isfinite(loss):
            raise FloatingPointError("Training/validation loss is non-finite.")

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_size = int(surface.shape[0])
        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["value_loss"] += float(value_loss.detach().item()) * batch_size
        totals["vertical_loss"] += float(gradient_loss.detach().item()) * batch_size
        # 新增
        totals["vertical_transport_loss"] += float(transport_loss.detach().item()) * batch_size
        sample_count += batch_size
        if metrics is not None:
            metrics.update(prediction.detach(), target, mask)

    result: dict[str, object] = {
        key: value / max(sample_count, 1) for key, value in totals.items()
    }
    if metrics is not None:
        result.update(metrics.compute())
        # 从已有逐层统计汇总深度带误差，不增加模型前向计算。
        result["depth_bands"] = {}
        for name, low, high in (("0_150m", 0, 150), ("200_2000m", 200, 2000)):
            layers = [row for row in result["per_depth"]
                      if low <= row["depth_m"] <= high and row["count"] > 0]
            count = sum(row["count"] for row in layers)
            if count > 0:
                result["depth_bands"][name] = {
                    "count": count,
                    "rmse": math.sqrt(sum(row["count"] * row["rmse"] ** 2 for row in layers) / count),
                    "mae": sum(row["count"] * row["mae"] for row in layers) / count,
                }
    return result


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    cfg: dict,
    epoch: int,
    best_score: float,
    history: list[dict],
    val_metrics: dict[str, object] | None,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
        "epoch": int(epoch),
        "best_selection_score": float(best_score),
        "selection_metric": "macro_nrmse_depth",
        "history": history,
        "val": val_metrics,
        "stage": stage_name(cfg),
        "data_signature": cfg.get("run_metadata", {}).get("data_signature"),
        "initialization": cfg.get("run_metadata", {}).get("initialization"),
        "sample_limit": cfg.get("run_metadata", {}).get("sample_limit"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train geographic Swin reconstruction model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", help="同阶段断点续训，恢复优化器及训练历史")
    group.add_argument("--init-checkpoint", help="IPRC最佳权重初始化GLORYS微调；不恢复优化状态")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_stage(cfg)
    require_data_paths(cfg)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if stage_name(cfg) == 'glorys_finetune' and not (args.resume or args.init_checkpoint):
        parser.error("GLORYS fine-tuning requires --init-checkpoint or same-stage --resume")
    if args.init_checkpoint and stage_name(cfg) != 'glorys_finetune':
        parser.error("--init-checkpoint is only allowed for glorys_finetune")
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    validate_model_data_contract(cfg["model"])
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)

    seed = int(train_cfg.get("seed", 2026))
    set_seed(seed, bool(train_cfg.get("deterministic", True)))
    device = pick_device(str(train_cfg.get("device", "auto")))
    configured_output = Path(args.output_dir or train_cfg["output_dir"])
    if args.max_samples is not None and args.output_dir is None:
        configured_output = configured_output / "smoke"
    if not args.resume and any((configured_output / name).exists() for name in ('last.pt', 'best.pt')):
        raise FileExistsError("Output already contains checkpoints; use --resume or a new --output-dir.")
    output_dir = ensure_dir(configured_output)
    train_cfg["output_dir"] = str(output_dir)

    train_dataset = make_dataset(cfg, "train", args.max_samples)
    val_dataset = make_dataset(cfg, "val", args.max_samples)
    if not np.array_equal(train_dataset.depth_values, val_dataset.depth_values):
        raise ValueError("Training and validation depth coordinates differ.")
    if int(cfg["model"]["out_depths"]) != train_dataset.depth_values.size:
        raise ValueError("model.out_depths does not match the label depth dimension.")
    norm = DepthNorm.from_config(cfg["data"])
    norm.validate_depth_count(train_dataset.depth_values.size)
    signature = data_signature(cfg, train_dataset.depth_values)
    initialization = dict(mode='random', optimizer_restored=False)

    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(cfg["model"]).to(device)
    if args.init_checkpoint:
        initialization = initialize_weights(model, args.init_checkpoint, cfg, signature,
                                            allow_partial=args.max_samples is not None)
    if model.moe is not None and not args.resume:
        surface_norm_path = cfg["data"].get("surface_norm_path")
        if not surface_norm_path:
            raise ValueError("MoE requires data.surface_norm_path.")
        statistics = fit_moe_statistics(
            train_dataset, norm, surface_norm_path, model.patch_size
        )
        model.moe.set_statistics(**statistics)
        initialization['moe_statistics_source'] = dict(
            label_path=cfg['data']['label_path'], train_range=cfg['data']['train_range'],
            sampled_month_count=len(train_dataset), partial=args.max_samples is not None)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    amp_enabled = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    depths_m = torch.as_tensor(train_dataset.depth_values, dtype=torch.float32, device=device)
    norm_mean, norm_std = norm.torch_mean_std(device)
    vertical_weight = (
        float(loss_cfg.get("vertical_weight", 0.0))
        if bool(loss_cfg.get("use_vertical_gradient", False))
        else 0.0
    )
    reference_distance = float(loss_cfg.get("vertical_reference_distance_m", 100.0))

    # 新增
    vertical_transport_weight = (
        float(loss_cfg.get("vertical_transport_weight", 0.0))
        if bool(loss_cfg.get("use_vertical_gradient_transport", False))
        else 0.0
    )
    vertical_transport_reference_distance = float(
        loss_cfg.get(
            "vertical_transport_reference_distance_m",
            100.0,
        )
    )

    if vertical_transport_weight < 0:
        raise ValueError(
            "vertical_transport_weight must be non-negative."
        )
    if vertical_transport_reference_distance <= 0:
        raise ValueError(
            "vertical_transport_reference_distance_m must be positive."
        )
    
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    if str(train_cfg.get("selection_metric", "macro_nrmse_depth")) != "macro_nrmse_depth":
        raise ValueError("Only macro_nrmse_depth is supported for checkpoint selection.")

    start_epoch = 1
    best_score = float("inf")
    history: list[dict] = []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state["config"]["data"] != cfg["data"] or state["config"]["model"] != cfg["model"]:
            raise ValueError("Resume requires the same data and model configuration.")
        if stage_name(state['config']) != stage_name(cfg) or state.get('data_signature') != signature:
            raise ValueError("Resume stage or preprocessing signature differs from checkpoint.")
        if state['config']['loss'] != cfg['loss']:
            raise ValueError("Resume requires the same loss configuration.")
        if state.get('sample_limit') != args.max_samples:
            raise ValueError('Resume requires the same --max-samples setting as the original run.')
        validate_optimizer_resume(state, optimizer, scheduler, train_cfg)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state.get("best_selection_score", float("inf")))
        history = list(state.get("history", []))
        initialization = state.get('initialization', dict(mode='resume'))

    cfg['run_metadata'] = dict(data_signature=signature, initialization=initialization,
                               sample_limit=args.max_samples,
                               optimizer_groups=group_summary(optimizer, scheduler))
    (output_dir / 'initialization.json').write_text(
        json.dumps(dict(stage=stage_name(cfg), **cfg['run_metadata']), indent=2, ensure_ascii=False), encoding='utf-8')

    epochs = int(train_cfg["epochs"])
    if epochs < start_epoch:
        raise ValueError(f"Total epochs {epochs} must be at least next epoch {start_epoch}.")
    val_every = int(train_cfg.get("val_every", 4))
    save_every = int(train_cfg.get("save_every", 5))
    last_val: dict[str, object] | None = state.get("val") if args.resume else None
    print(
        f"device={device} parameters={model.parameter_count():,} "
        f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}"
    )
    for group in group_summary(optimizer, scheduler):
        print(f"optimizer_group={group['name']} lr={group['learning_rate']:.3g} "
              f"min_lr={group['min_lr']:.3g} parameters={group['parameter_count']:,}")
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_learning_rates = learning_rates(optimizer)
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, scaler, amp_enabled,
                vertical_weight, depths_m, norm_mean, norm_std,
                reference_distance, grad_clip,
                # 新增
                vertical_transport_weight=vertical_transport_weight,
                vertical_transport_reference_distance_m=(
                    vertical_transport_reference_distance
                ),
            )
            val_metrics: dict[str, object] | None = None
            improved = False
            if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                with torch.no_grad():
                    val_metrics = run_epoch(
                        model, val_loader, device, None, scaler, amp_enabled,
                        vertical_weight, depths_m, norm_mean, norm_std,
                        reference_distance, grad_clip,
                        metric_norm=norm,
                        metric_depths=val_dataset.depth_values,
                        # 新增
                        vertical_transport_weight=vertical_transport_weight,
                        vertical_transport_reference_distance_m=(
                            vertical_transport_reference_distance
                        ),
                    )
                score = float(val_metrics["macro_nrmse_depth"])
                if not math.isfinite(score):
                    raise FloatingPointError("Validation macro_nrmse_depth is non-finite.")
                scheduler.step(score)
                improved = score < best_score
                if improved:
                    best_score = score
                last_val = val_metrics

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                # lr保持原字段语义；额外记录本轮实际使用值和调度后的各组值。
                "lr_groups_used": epoch_learning_rates,
                "lr_groups": learning_rates(optimizer),
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            # Persist recovery state each epoch, including non-validation epochs.
            torch.save(
                checkpoint_payload(
                    model, optimizer, scheduler, scaler, cfg, epoch,
                    best_score, history, last_val,
                ),
                output_dir / "last.pt",
            )
            if improved:
                torch.save(
                    checkpoint_payload(
                        model, optimizer, scheduler, scaler, cfg, epoch,
                        best_score, history, val_metrics,
                    ),
                    output_dir / "best.pt",
                )
            if epoch % save_every == 0 or epoch == epochs:
                torch.save(
                    checkpoint_payload(
                        model, optimizer, scheduler, scaler, cfg, epoch,
                        best_score, history, last_val,
                    ),
                    output_dir / f"epoch_{epoch:03d}.pt",
                )
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if val_metrics is None:
                print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
                      f"lr_groups={learning_rates(optimizer)}")
            else:
                print(
                    f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
                    f"val_macro_nrmse={val_metrics['macro_nrmse_depth']:.6f} "
                    f"val_pooled_rmse={val_metrics['pooled_rmse']:.6f} "
                    f"val_pooled_r2={val_metrics['pooled_r2']:.6f} "
                    f"lr_groups={learning_rates(optimizer)} "
                    f"val_depth_bands={val_metrics['depth_bands']}"
                )

        if history:
            torch.save(
                checkpoint_payload(
                    model, optimizer, scheduler, scaler, cfg,
                    int(history[-1]["epoch"]), best_score, history, last_val,
                ),
                output_dir / "last.pt",
            )
    finally:
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    main()
