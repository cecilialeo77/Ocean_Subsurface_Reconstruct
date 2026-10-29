import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
import yaml

from glorys_swin_reconstruction.data import SURFACE_CHANNELS


@pytest.mark.parametrize("with_moe", [False, True])
def test_train_resume_and_reconstruct_glorys_contract(tmp_path, with_moe):
    """Exercise the real CLI with a small dense field and flat GLORYS stats."""
    rng = np.random.default_rng(2026)
    # 仅保留目标月份，不提供前五个月；训练、续训和推理必须仍然成功。
    dates = np.array(['2005-01','2019-01','2021-01','2021-02'], dtype='datetime64[D]')
    days = (dates - np.datetime64('1993-01-01')).astype('int32')
    coords = {'time': days, 'lat': np.arange(16), 'lon': np.arange(16)}
    surface = xr.Dataset({'input': (('time', 'channel', 'lat', 'lon'),
                                   rng.normal(size=(len(dates), 8, 16, 16)).astype('float32'))},
                         coords={**coords, 'channel': list(SURFACE_CHANNELS)})
    surface.time.attrs.update(units='days since 1993-01-01', calendar='proleptic_gregorian')
    mask = np.ones((2, 16, 16), dtype='uint8')
    mask[:, 0, :] = 0
    labels = xr.Dataset({
        'temperature': (('time', 'depth', 'lat', 'lon'), rng.normal(size=(len(dates), 2, 16, 16)).astype('float32')),
        'valid_label_mask': (('time', 'depth', 'lat', 'lon'), np.broadcast_to(mask, (len(dates), 2, 16, 16)))},
        coords={**coords, 'depth': [0., 100.]})
    labels.time.attrs.update(surface.time.attrs)
    domain = xr.Dataset({'valid_ocean_mask': (('depth', 'lat', 'lon'), mask)},
                        coords={'depth': [0., 100.], 'lat': coords['lat'], 'lon': coords['lon']})
    for name, ds in [('surface', surface), ('labels', labels), ('domain', domain)]:
        ds.to_netcdf(tmp_path / f'{name}.nc', engine='scipy')
    norm = tmp_path / 'norm.json'
    norm.write_text(json.dumps({'mean': [18., 10.], 'std': [9., 5.], 'raw_units': 'degree_C'}))
    project = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((project / 'configs/glorys_baseline.yaml').read_text(encoding='utf-8'))
    cfg['data'].update(surface_path=str(tmp_path/'surface.nc'), label_path=str(tmp_path/'labels.nc'),
                       domain_mask_path=str(tmp_path/'domain.nc'), label_norm_path=str(norm), time_text_var=None,
                       train_range=['2005-01', '2005-01'], val_range=['2019-01', '2019-01'],
                       test_range=['2021-01', '2021-02'])
    cfg['model'].update(image_size=[16,16], out_depths=2, embed_dim=8, encoder_depths=[1,1],
                        decoder_depths=[1], num_heads=[1,2], window_size=[2,2], shift_size=[1,1],
                        drop_path_rate=0., use_checkpoint=False)
    cfg['model']['moe'] = {'enabled': False}
    if with_moe:
        surface_norm = tmp_path / 'surface_norm.json'
        surface_norm.write_text(json.dumps({'variables': {
            name: {'mean': 0., 'std': 1.} for name in SURFACE_CHANNELS[:5]
        }}))
        cfg['data']['surface_norm_path'] = str(surface_norm)
        cfg['model']['moe'] = {'enabled': True, 'width': 4, 'prior_strength': 0.5}
    output = tmp_path / 'output'
    cfg['training'].update(output_dir=str(output), device='cpu', epochs=1, val_every=1)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    env = {**os.environ, 'PYTHONPATH': str(project/'src'), 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
    def run(module, *args):
        result = subprocess.run([sys.executable, '-m', f'glorys_swin_reconstruction.{module}',
                                 '--config', str(config), *args], env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    run('inspect_data')
    run('train')
    assert (output/'best.pt').exists() and (output/'last.pt').exists()
    run('train', '--resume', str(output/'last.pt'), '--epochs', '2')
    history = json.loads((output/'history.json').read_text())
    assert [row['epoch'] for row in history] == [1,2]
    run('reconstruct', '--checkpoint', str(output/'best.pt'), '--split', 'test')
    metrics = json.loads((output/'metrics_test.json').read_text())
    assert metrics['units'] == 'degree_C'
    assert metrics['count'] == 960
    assert np.isfinite(metrics['pooled_rmse'])
    with xr.open_dataset(output/'reconstruction_test.nc') as result:
        assert result.pred_temperature.shape == (2,2,16,16)
        assert result.pred_temperature.units == 'degree_C'
        assert result.history_months == 1
        assert len(result.input_channels.split(',')) == 11
        assert 'current month only' in result.history_order
        assert np.isnan(result.pred_temperature.values[:,:,0,:]).all()
        assert np.isfinite(result.pred_temperature.values[:,:,1:,:]).all()

    monthly = json.loads((output/'metrics_test_monthly.json').read_text())
    assert monthly['months'] == ['2021-01', '2021-02']
    assert monthly['is_complete'] and len(monthly['records']) == 2
    assert sum(row['count'] for row in monthly['records']) == metrics['count']
    before = (output/'metrics_test.json').read_bytes()
    run('evaluate_reconstruction', '--reconstruction', str(output/'reconstruction_test.nc'))
    offline = json.loads((output/'metrics_test_reevaluated.json').read_text())
    assert offline['historical_comparison']['passed']
    assert (output/'metrics_test.json').read_bytes() == before
    offline_months = json.loads((output/'metrics_test_monthly.json').read_text())
    for online_row, offline_row in zip(monthly['records'], offline_months['records']):
        assert online_row['count'] == offline_row['count']
        assert np.isclose(online_row['pooled_rmse'], offline_row['pooled_rmse'], atol=1e-7)
    run('reconstruct', '--max-months', '1')
    preview = json.loads((output/'preview/metrics_test_monthly.json').read_text())
    assert preview['month_count'] == 1 and preview['is_preview'] and not preview['is_complete']
    assert (output/'metrics_test.json').read_bytes() == before
