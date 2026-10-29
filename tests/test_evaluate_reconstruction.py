import json

import numpy as np
import pytest
import xarray as xr
import yaml

from glorys_swin_reconstruction.evaluate_reconstruction import evaluate


@pytest.fixture
def saved_case(tmp_path):
    times = np.array(['2021-01-01','2021-02-01'], dtype='datetime64[D]')
    coords = dict(time=times, depth=[0.,100.], lat=[0.], lon=[0.,1.])
    values = np.arange(8, dtype='float32').reshape(2,2,1,2)
    labels = xr.Dataset({'temperature': (('time','depth','lat','lon'), values),
                         'valid_label_mask': (('time','depth','lat','lon'), np.ones_like(values))}, coords=coords)
    labels.to_netcdf(tmp_path/'labels.nc')
    domain = xr.Dataset({'valid_ocean_mask': (('depth','lat','lon'), np.ones((2,1,2)))},
                        coords={key: val for key,val in coords.items() if key != 'time'})
    domain.to_netcdf(tmp_path/'domain.nc')
    mean = np.array([10.,20.],dtype='float32').reshape(1,2,1,1)
    std = np.array([2.,4.],dtype='float32').reshape(1,2,1,1)
    pred = xr.Dataset({'pred_temperature': (('time','depth','lat','lon'), values*std+mean+1)},coords=coords)
    pred.pred_temperature.attrs['units'] = 'degree_C'
    pred.attrs['split'] = 'test'
    pred.to_netcdf(tmp_path/'reconstruction_test.nc')
    (tmp_path/'norm.json').write_text(json.dumps(dict(mean=[10.,20.],std=[2.,4.],raw_units='degree_C')))
    cfg = dict(data=dict(label_path=str(tmp_path/'labels.nc'), domain_mask_path=str(tmp_path/'domain.nc'),
                         label_norm_path=str(tmp_path/'norm.json'), test_range=['2021-01','2021-02']),
               model={},loss={},training={})
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    return tmp_path


def test_offline_physical_units_and_preview(saved_case):
    root = saved_case
    result = evaluate(root/'config.yaml', root/'reconstruction_test.nc')
    assert result['pooled_rmse'] == 1. and result['pooled_mae'] == 1.
    assert result['count'] == 8 and result['is_complete']
    assert result['per_depth'][1]['nrmse_train_std'] == .25
    assert not (root/'metrics_test.json').exists()
    full_bytes = (root/'metrics_test_monthly.json').read_bytes()
    preview = evaluate(root/'config.yaml', root/'reconstruction_test.nc',max_months=1)
    assert preview['month_count'] == 1 and not preview['is_complete']
    assert (root/'preview/metrics_test_monthly.json').exists()
    assert (root/'metrics_test_monthly.json').read_bytes() == full_bytes
    with pytest.raises(ValueError,match='positive'):
        evaluate(root/'config.yaml',root/'reconstruction_test.nc',max_months=0)


@pytest.mark.parametrize('change,message', [('depth','coordinates'),('date','timestamps'),
                                           ('duplicate','Duplicate'),('missing','Missing'),
                                           ('reverse','chronological'),('nan','Non-finite'),
                                           ('units','units')])
def test_bad_saved_fields_fail_without_publishing(saved_case,change,message):
    root = saved_case
    path = root/'reconstruction_test.nc'
    with xr.open_dataset(path) as source:
        ds = source.load()
    if change == 'depth':
        ds = ds.assign_coords(depth=[0.,200.])
    elif change == 'date':
        ds = ds.assign_coords(time=np.array(['2021-01-02','2021-02-01'],dtype='datetime64[D]'))
    elif change == 'duplicate':
        ds = ds.assign_coords(time=np.array(['2021-01-01','2021-01-01'],dtype='datetime64[D]'))
    elif change == 'missing':
        ds = ds.isel(time=[0])
    elif change == 'reverse':
        ds = ds.isel(time=[1,0])
    elif change == 'nan':
        ds.pred_temperature.values[0,0,0,0] = np.nan
    elif change == 'units':
        ds.pred_temperature.attrs['units'] = 'K'
    ds.to_netcdf(path,mode='w')
    with pytest.raises(ValueError,match=message):
        evaluate(root/'config.yaml',path)
    assert not (root/'metrics_test_monthly.json').exists()
