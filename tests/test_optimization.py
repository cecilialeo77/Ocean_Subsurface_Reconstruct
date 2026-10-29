"""验证真实Geo-Swin参数分组、学习率下限及检查点续训兼容性。"""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from glorys_swin_reconstruction.optimization import (
    build_optimizer, build_scheduler, group_summary, learning_rates,
    validate_optimizer_resume,
)
from glorys_swin_reconstruction.train import build_model


@pytest.fixture
def setup():
    project = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((project / 'configs/glorys_finetune_grouped_lr.yaml').read_text(encoding='utf-8'))
    cfg['model'].update(image_size=[16, 16], out_depths=3, embed_dim=8,
                        encoder_depths=[1, 1], decoder_depths=[1], num_heads=[1, 2],
                        window_size=[2, 2], shift_size=[1, 1], drop_path_rate=0., use_checkpoint=False)
    cfg['model']['moe']['width'] = 4
    return build_model(cfg['model']), cfg['training']


def test_groups_cover_real_model_once_without_freezing(setup):
    model, training = setup
    optimizer = build_optimizer(model, training)
    scheduler = build_scheduler(optimizer, training)
    assert learning_rates(optimizer) == {'encoder': 1e-5, 'decoder_head_moe': 5e-5}
    assert scheduler.min_lrs == [2e-7, 1e-6]
    assigned = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in model.parameters()}
    assert all(p.requires_grad for p in model.parameters())
    for g in optimizer.param_groups:
        roots = {name.split('.')[0] for name in g['param_names']}
        if g['name'] == 'encoder':
            assert roots == {'patch_embed', 'encoder', 'bottleneck_norm'}
        else:
            assert roots == {'decoder', 'continuous_head', 'moe'}
    assert sum(g['parameter_count'] for g in group_summary(optimizer, scheduler)) == model.parameter_count()


def test_plateau_scheduler_preserves_ratio_and_both_floors(setup):
    model, training = setup
    training['scheduler_patience'] = 0
    optimizer = build_optimizer(model, training)
    scheduler = build_scheduler(optimizer, training)
    scheduler.step(1.)
    for _ in range(12):
        scheduler.step(2.)
        encoder, decoder = learning_rates(optimizer).values()
        assert encoder / decoder == pytest.approx(.2)
        assert encoder >= 2e-7 and decoder >= 1e-6
    assert list(learning_rates(optimizer).values()) == pytest.approx([2e-7, 1e-6])


def test_unknown_and_cross_group_shared_parameters_are_rejected(setup):
    model, training = setup
    model.register_parameter('unexpected', torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match='Unassigned optimizer parameter'):
        build_optimizer(model, training)
    del model.unexpected
    model.moe.shared.weight = model.patch_embed.projection.weight
    with pytest.raises(ValueError, match='multiple groups'):
        build_optimizer(model, training)


@pytest.mark.parametrize('bad', ['missing_group', 'unknown_option', 'nan_lr', 'negative_floor', 'floor_above_lr'])
def test_invalid_group_configuration_is_rejected(setup, bad):
    model, training = setup
    if bad == 'missing_group':
        del training['parameter_groups']['encoder']
    elif bad == 'unknown_option':
        training['parameter_groups']['encoder']['lr'] = 1e-5
    elif bad == 'nan_lr':
        training['parameter_groups']['encoder']['learning_rate'] = float('nan')
    elif bad == 'negative_floor':
        training['parameter_groups']['encoder']['min_lr'] = -1e-6
    else:
        training['parameter_groups']['encoder']['min_lr'] = 1e-3
    with pytest.raises(ValueError):
        build_optimizer(model, training)


def test_grouped_resume_restores_moments_scheduler_and_rejects_changed_plan(setup, tmp_path):
    model, training = setup
    training['scheduler_patience'] = 0
    optimizer = build_optimizer(model, training)
    scheduler = build_scheduler(optimizer, training)
    sum(p.square().mean() for p in model.parameters()).backward()
    optimizer.step()
    scheduler.step(1.)
    scheduler.step(2.)
    path = tmp_path / 'state.pt'
    torch.save({'config': {'training': copy.deepcopy(training)},
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    restored = build_optimizer(model, training)
    restored_scheduler = build_scheduler(restored, training)
    validate_optimizer_resume(state, restored, restored_scheduler, training)
    restored.load_state_dict(state['optimizer'])
    restored_scheduler.load_state_dict(state['scheduler'])
    assert learning_rates(restored) == learning_rates(optimizer)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for a, b in zip(optimizer.state.values(), restored.state.values()):
        for key in ['step', 'exp_avg', 'exp_avg_sq']:
            torch.testing.assert_close(a[key], b[key])
    changed = copy.deepcopy(training)
    changed['parameter_groups']['encoder']['learning_rate'] = 2e-5
    changed_optimizer = build_optimizer(model, changed)
    changed_scheduler = build_scheduler(changed_optimizer, changed)
    with pytest.raises(ValueError, match='identical parameter_groups'):
        validate_optimizer_resume(state, changed_optimizer, changed_scheduler, changed)
    reordered = copy.deepcopy(state)
    reordered['optimizer']['param_groups'].reverse()
    with pytest.raises(ValueError, match='group names and parameter order'):
        validate_optimizer_resume(reordered, restored, restored_scheduler, training)


def test_legacy_single_group_resume_and_group_switch_guard(setup):
    model, grouped = setup
    training = {k: v for k, v in grouped.items() if k != 'parameter_groups'}
    optimizer = build_optimizer(model, training)
    scheduler = build_scheduler(optimizer, training)
    assert len(optimizer.param_groups) == 1
    assert learning_rates(optimizer) == {'all': training['learning_rate']}
    state = {'config': {'training': training}, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    validate_optimizer_resume(state, optimizer, scheduler, training)
    new_optimizer = build_optimizer(model, grouped)
    with pytest.raises(ValueError, match='parameter groups differ'):
        validate_optimizer_resume(state, new_optimizer, build_scheduler(new_optimizer, grouped), grouped)


def test_grouped_config_preserves_data_model_loss_and_other_training_settings():
    project = Path(__file__).resolve().parents[1]
    old = yaml.safe_load((project / 'configs/glorys_finetune.yaml').read_text(encoding='utf-8'))
    new = yaml.safe_load((project / 'configs/glorys_finetune_grouped_lr.yaml').read_text(encoding='utf-8'))
    for section in ['data', 'model', 'loss']:
        assert new[section] == old[section]
    for key in old['training'].keys() - {'learning_rate', 'output_dir'}:
        assert new['training'][key] == old['training'][key]
