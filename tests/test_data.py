import numpy as np
import pytest
import xarray as xr

from glorys_swin_reconstruction.data import (
    HISTORY_MONTHS, MODEL_INPUT_CHANNELS, SURFACE_CHANNELS, FullFieldDataset,
    FullFieldSpec, assemble_model_input, history_window_indices, read_model_input,
    validate_model_data_contract,
)


def test_current_month_channel_order_and_month_encoding():
    surface=np.stack([np.full((2,3),i,dtype=np.float32) for i in range(8)])
    ocean=np.ones((1,2,3),dtype=np.float32)
    result=assemble_model_input(surface,ocean,month=4)
    assert HISTORY_MONTHS==1
    assert result.shape==(11,2,3)==(len(MODEL_INPUT_CHANNELS),2,3)
    assert not any('lag' in name for name in MODEL_INPUT_CHANNELS)
    np.testing.assert_array_equal(result[:8],surface)
    np.testing.assert_array_equal(result[8],ocean[0])
    np.testing.assert_allclose(result[9],1.,atol=1e-6)
    np.testing.assert_allclose(result[10],0.,atol=1e-6)
    surface[0,0,0]=np.nan; surface[1,0,1]=np.inf
    result=assemble_model_input(surface,ocean,4)
    assert np.isfinite(result).all() and result[0,0,0]==result[1,0,1]==0


def test_first_month_and_gaps_need_no_history_context():
    keys=np.array(['2005-01','2019-01','2021-01'])
    np.testing.assert_array_equal(history_window_indices(keys,np.array([0,2])),[[0],[2]])


@pytest.mark.parametrize('keys,targets,message',[
    (['2018-12','2018-12'],[0],'unique'),
    (['2018-12','2018-11'],[1],'chronological'),
    (['2019-01'],[-1],'outside'),
    (['2019-01'],[1],'outside'),
    (['2019-01'],[[0]],'one-dimensional'),
])
def test_bad_target_indices_fail(keys,targets,message):
    with pytest.raises((ValueError,IndexError),match=message):
        history_window_indices(np.array(keys),np.array(targets))


@pytest.mark.parametrize('old_channels',[21,36])
def test_contract_rejects_history_models(old_channels):
    validate_model_data_contract({'in_channels':11,'out_depths':27})
    with pytest.raises(ValueError,match='must be 11'):
        validate_model_data_contract({'in_channels':old_channels,'out_depths':27})


def test_reader_reads_only_current_index_and_rejects_history_window():
    class SurfaceSpy:
        def __init__(self): self.calls=[]
        def isel(self,**kwargs):
            self.calls.append(kwargs)
            assert kwargs=={'time':0}, 'Reader requested another month or a history slice'
            return type('Field',(),{'values':np.zeros((8,2,3),dtype='f4')})()
    surface=SurfaceSpy(); ocean=np.ones((1,2,3),dtype='f4')
    result,valid=read_model_input(surface,np.array([0]),ocean,1)
    assert result.shape==(11,2,3) and valid.shape==(5,2,3)
    assert surface.calls==[{'time':0}]
    with pytest.raises(ValueError,match='single target'):
        read_model_input(surface,np.array([5,4,3,2,1,0]),ocean,1)


def test_dataset_current_target_without_previous_months(tmp_path):
    dates=np.array(['2005-01','2005-02'],dtype='datetime64[M]')
    raw=np.broadcast_to(np.arange(2,dtype='f4')[:,None,None,None],(2,8,2,3)).copy()
    raw[0,2,0,0]=np.nan
    surface=xr.Dataset({'input':(('time','channel','lat','lon'),raw)},
                       coords={'time':dates,'channel':list(SURFACE_CHANNELS),'lat':[0,1],'lon':[0,1,2]})
    labels=xr.Dataset({'temperature':(('time','depth','lat','lon'),np.full((2,2,2,3),10.,'f4')),
                       'valid_label_mask':(('time','depth','lat','lon'),np.ones((2,2,2,3),'u1'))},
                      coords={'time':dates,'depth':[0.,100.],'lat':[0,1],'lon':[0,1,2]})
    domain=xr.Dataset({'valid_ocean_mask':(('depth','lat','lon'),np.ones((2,2,3),'u1'))},
                      coords={'depth':[0.,100.],'lat':[0,1],'lon':[0,1,2]})
    for name,ds in [('surface',surface),('labels',labels),('domain',domain)]:
        ds.to_netcdf(tmp_path/f'{name}.nc')
    spec=FullFieldSpec(str(tmp_path/'surface.nc'),str(tmp_path/'labels.nc'),str(tmp_path/'domain.nc'))
    dataset=FullFieldDataset(spec,np.array([0]))
    try:
        sample=dataset[0]
        assert sample['time_index'].item()==0
        assert sample['surface'].shape==(11,2,3)
        assert sample['surface_valid'][2,0,0]==0
        inferred,valid=read_model_input(dataset.surface,np.array([0]),dataset.surface_ocean_mask,1)
        np.testing.assert_array_equal(inferred,sample['surface'])
        np.testing.assert_array_equal(valid,sample['surface_valid'])
        # 改变另一个月的海表及全部标签，不应改变本月输入。
        dataset.surface.values[1]=999
        dataset.label.values[:]=-999
        np.testing.assert_array_equal(dataset[0]['surface'],sample['surface'])
    finally:
        dataset.close()
