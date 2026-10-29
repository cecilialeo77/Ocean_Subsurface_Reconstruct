"""优化器参数分组、各组学习率调度和续训一致性检查。"""
from __future__ import annotations

import math

import torch
from torch import nn


GROUP_MODULES = {
    "encoder": {"patch_embed", "encoder", "bottleneck_norm"},
    "decoder_head_moe": {
        "decoder", "continuous_head", "moe", "final_expand", "output_projection",
    },
}


def _number(value, name: str, *, positive: bool) -> float:
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return number


def _group_settings(training: dict) -> dict | None:
    settings = training.get("parameter_groups")
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) != set(GROUP_MODULES):
        raise ValueError("parameter_groups must define encoder and decoder_head_moe")
    result = {}
    for name in GROUP_MODULES:
        options = settings[name]
        if not isinstance(options, dict) or set(options) != {"learning_rate", "min_lr"}:
            raise ValueError(f"parameter_groups.{name} requires learning_rate and min_lr")
        lr = _number(options["learning_rate"], f"{name}.learning_rate", positive=True)
        floor = _number(options["min_lr"], f"{name}.min_lr", positive=False)
        if floor > lr:
            raise ValueError(f"{name}.min_lr cannot exceed its learning_rate")
        result[name] = {"learning_rate": lr, "min_lr": floor}
    return result


def build_optimizer(model: nn.Module, training: dict) -> torch.optim.AdamW:
    """未配置分组时沿用原有单组AdamW；配置分组时严格检查参数覆盖。"""
    settings = _group_settings(training)
    decay = float(training.get("weight_decay", 0.0))
    if settings is None:
        return torch.optim.AdamW(
            model.parameters(), lr=float(training["learning_rate"]), weight_decay=decay,
        )

    groups = {name: {"params": [], "param_names": [], "name": name,
                     "lr": options["learning_rate"]}
              for name, options in settings.items()}
    assigned = {}
    # 保留别名用于检查跨组共享参数，避免同一参数被两个学习率同时更新。
    for parameter_name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        module_name = parameter_name.split(".", 1)[0]
        matches = [name for name, modules in GROUP_MODULES.items() if module_name in modules]
        if len(matches) != 1:
            raise ValueError(f"Unassigned optimizer parameter: {parameter_name}")
        name = matches[0]
        identity = id(parameter)
        if identity in assigned:
            if assigned[identity] != name:
                raise ValueError(f"Shared parameter belongs to multiple groups: {parameter_name}")
            continue
        assigned[identity] = name
        groups[name]["params"].append(parameter)
        groups[name]["param_names"].append(parameter_name)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(assigned) != expected or any(not group["params"] for group in groups.values()):
        raise ValueError("Optimizer groups must cover every trainable parameter exactly once and be non-empty")
    return torch.optim.AdamW(
        list(groups.values()), lr=settings["decoder_head_moe"]["learning_rate"], weight_decay=decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, training: dict):
    settings = _group_settings(training)
    if settings is None:
        floors = float(training.get("min_lr", 0.0))
    else:
        names = [group.get("name") for group in optimizer.param_groups]
        if names != list(settings):
            raise ValueError("Optimizer group order does not match parameter_groups")
        floors = [settings[name]["min_lr"] for name in names]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(training.get("scheduler_factor", 0.5)),
        patience=int(training.get("scheduler_patience", 3)), min_lr=floors,
        # 保留原有eps，未改变基于验证指标的调度规则。
    )


def learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {group.get("name", "all" if len(optimizer.param_groups) == 1 else f"group_{i}"):
            float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}


def group_summary(optimizer: torch.optim.Optimizer, scheduler) -> list[dict]:
    names = list(learning_rates(optimizer))
    return [dict(name=names[i], learning_rate=float(group["lr"]), min_lr=float(scheduler.min_lrs[i]),
                 parameter_count=sum(parameter.numel() for parameter in group["params"]),
                 tensor_count=len(group["params"]))
            for i, group in enumerate(optimizer.param_groups)]


def validate_optimizer_resume(state: dict, optimizer: torch.optim.Optimizer,
                              scheduler, training: dict) -> None:
    """分组续训必须恢复相同参数组和策略；旧单组检查点仍可按原方式续训。"""
    saved_groups = state["optimizer"]["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("Resume optimizer parameter groups differ; start a new experiment with --init-checkpoint")
    saved_training = state["config"]["training"]
    saved_settings, settings = _group_settings(saved_training), _group_settings(training)
    if saved_settings != settings:
        raise ValueError("Resume requires identical parameter_groups; use --init-checkpoint for a new LR experiment")
    if settings is None:
        return
    for saved, current in zip(saved_groups, optimizer.param_groups):
        if (saved.get("name") != current["name"]
                or saved.get("param_names") != current["param_names"]
                or len(saved["params"]) != len(current["params"])):
            raise ValueError("Resume requires identical optimizer group names and parameter order")
    for key, default in [("weight_decay", 0.0), ("scheduler_factor", 0.5), ("scheduler_patience", 3)]:
        if saved_training.get(key, default) != training.get(key, default):
            raise ValueError(f"Resume requires identical {key} for grouped optimization")
    if state["scheduler"]["min_lrs"] != scheduler.min_lrs:
        raise ValueError("Resume scheduler minimum learning rates differ")
