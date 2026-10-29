"""两阶段训练约束、标准化溯源和仅模型权重迁移；不恢复优化器状态。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import HISTORY_MONTHS, MODEL_INPUT_CHANNELS
from .norm import DepthNorm
from .time_utils import month_keys


def stage_name(cfg):
    return cfg['training'].get('stage', 'glorys_baseline')


def validate_stage(cfg):
    stage = stage_name(cfg)
    if stage not in ('iprc_pretrain', 'glorys_finetune', 'glorys_baseline'):
        raise ValueError(f'Unknown training stage: {stage}')
    ranges = []
    for split in ('train', 'val', 'test'):
        span = cfg['data'].get(f'{split}_range')
        if span is None:
            if split != 'test' and stage != 'glorys_baseline':
                raise ValueError(f'{split}_range must be configured')
            continue
        if len(span) != 2 or str(span[0]) > str(span[1]):
            raise ValueError(f'Invalid {split}_range: {span}')
        months = np.arange(np.datetime64(str(span[0]), 'M'),
                           np.datetime64(str(span[1]), 'M') + 1).astype(str)
        ranges.append((split, set(months)))
    for i, (left, a) in enumerate(ranges):
        for right, b in ranges[i+1:]:
            if a & b:
                raise ValueError(f'Time splits overlap: {left}/{right}')
    if stage == 'iprc_pretrain':
        if cfg['data'].get('test_range') is not None:
            raise ValueError('IPRC pretraining has no test split; set test_range: null')
        train, val = cfg['data']['train_range'], cfg['data']['val_range']
        if not ('2005-01' <= train[0] <= train[1] <= '2018-12'):
            raise ValueError('IPRC training must stay within 2005-01..2018-12')
        if not ('2019-01' <= val[0] <= val[1] <= '2020-04'):
            raise ValueError('IPRC validation must stay within 2019-01..2020-04')


def data_signature(cfg, depths):
    """记录实际标准化数值和海表契约，防止路径相同但文件被替换后误迁移。"""
    norm = DepthNorm.from_config(cfg['data'])
    surface_norm = cfg['data'].get('surface_norm_path')
    surface_stats = None
    if surface_norm and cfg['model'].get('moe', {}).get('enabled', False):
        surface_stats = json.loads(Path(surface_norm).read_text(encoding='utf-8'))
    return dict(mean=norm.mean.tolist(), std=norm.std.tolist(), units=norm.units,
                depths=np.asarray(depths).tolist(), history_months=HISTORY_MONTHS,
                input_channels=list(MODEL_INPUT_CHANNELS), surface_statistics=surface_stats,
                surface_path=str(Path(cfg['data']['surface_path']).resolve()),
                domain_mask_path=str(Path(cfg['data']['domain_mask_path']).resolve()))


def validate_labels(cfg, labels, indices):
    """检查被选月份确实有标签，并验证文件声明的共享标准化指纹。"""
    expected = labels.attrs.get('standardization_sha256')
    if expected:
        actual = hashlib.sha256(Path(cfg['data']['label_norm_path']).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Label standardization SHA256 differs from label_norm_path')
    if 'label_available' in labels:
        availability = np.asarray(labels['label_available'].values)
        if labels['label_available'].dims != ('time',) or not np.isin(availability, (0,1)).all():
            raise ValueError('label_available must be binary with dimensions (time,)')
        if np.any(availability[indices] != 1):
            raise ValueError('Requested months contain unavailable placeholder labels')
    elif stage_name(cfg) == 'iprc_pretrain':
        raise ValueError('IPRC pretraining requires label_available to reject placeholders')
    if stage_name(cfg) == 'iprc_pretrain':
        variable = cfg['data'].get('label_var', 'temperature')
        if labels[variable].attrs.get('source_variable') != 'PTEMP':
            raise ValueError('IPRC labels must declare source_variable=PTEMP')


def validate_coverage(dataset, cfg, split, partial=False):
    keys = month_keys(dataset._surface_ds, dataset.spec.time_text_var)
    start, end = cfg['data'][f'{split}_range']
    expected = np.arange(np.datetime64(start, 'M'), np.datetime64(end, 'M') + 1).astype(str)
    selected = keys[dataset.time_indices]
    if not partial and not np.array_equal(selected, expected):
        raise ValueError(f'Missing or duplicated target months in {split}')
    validate_labels(cfg, dataset._label_ds, dataset.time_indices)


def initialize_weights(model, checkpoint, cfg, signature, *, allow_partial=False):
    """加载全部模型参数；统计缓冲量随后由当前阶段训练标签覆盖。"""
    path = Path(checkpoint).resolve()
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('sample_limit') is not None and not allow_partial:
        raise ValueError('A smoke checkpoint cannot initialize formal full-data fine-tuning')
    if state['config']['model'] != cfg['model']:
        raise ValueError('Initialization requires identical model architecture')
    if stage_name(state['config']) != 'iprc_pretrain' or stage_name(cfg) != 'glorys_finetune':
        raise ValueError('Weight transfer must be iprc_pretrain -> glorys_finetune')
    if state.get('data_signature') != signature:
        raise ValueError('Transfer requires identical normalization, depths and surface input contract')
    model.load_state_dict(state['model'], strict=True)
    return dict(mode='pretrained_weights', checkpoint=str(path),
                source_stage=stage_name(state['config']), source_epoch=int(state['epoch']),
                source_best_score=float(state['best_selection_score']),
                optimizer_restored=False, scheduler_restored=False, scaler_restored=False,
                moe_statistics='recomputed from current-stage training labels after weight loading')
