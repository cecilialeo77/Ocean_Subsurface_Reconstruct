"""用小网格真实CLI覆盖两阶段训练、统计切换、续训及最终评估。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from glorys_swin_reconstruction.data import SURFACE_CHANNELS
from glorys_swin_reconstruction.train import build_model
from glorys_swin_reconstruction.transfer import (
    initialize_weights, validate_stage, validate_labels,
)


def test_stage_rejects_leakage_and_fake_test():
    cfg = dict(training=dict(stage='iprc_pretrain'), data=dict(
        train_range=['2005-01','2018-12'], val_range=['2019-01','2020-04'],test_range=None))
    validate_stage(cfg)
    bad=copy.deepcopy(cfg); bad['data']['train_range'][1]='2019-01'
    with pytest.raises(ValueError,match='overlap'): validate_stage(bad)
    bad=copy.deepcopy(cfg); bad['data']['test_range']=['2021-01','2024-12']
    with pytest.raises(ValueError,match='no test'): validate_stage(bad)


def test_label_guards(tmp_path):
    stats=tmp_path/'stats.json'; stats.write_text('{}')
    cfg=dict(training=dict(stage='iprc_pretrain'),data=dict(label_norm_path=str(stats)))
    ds=xr.Dataset(dict(temperature=(('time','depth'),[[0.],[0.]]),label_available=('time',[1,0])))
    ds.temperature.attrs['source_variable']='PTEMP'
    validate_labels(cfg,ds,np.array([0]))
    with pytest.raises(ValueError,match='placeholder'): validate_labels(cfg,ds,np.array([1]))
    ds.attrs['standardization_sha256']='wrong'
    with pytest.raises(ValueError,match='SHA256'): validate_labels(cfg,ds,np.array([0]))


@pytest.mark.parametrize('grouped,front', [(False, False), (True, False), (True, True)])
def test_complete_two_stage_cli(tmp_path, grouped, front):
    project=Path(__file__).resolve().parents[1]
    rng=np.random.default_rng(2026)
    # 两阶段均不提供历史上下文月份，验证当月输入的完整迁移链路。
    dates=np.array(['2005-01','2005-02','2019-01','2019-02','2021-01','2021-02'],dtype='datetime64[M]')
    keys=dates.astype(str)
    days=(dates.astype('datetime64[D]')-np.datetime64('1993-01-01')).astype('i4')
    coords=dict(time=days,lat=np.arange(16,dtype='f4'),lon=np.arange(16,dtype='f4'))
    surface=xr.Dataset(dict(input=(('time','channel','lat','lon'),
                                  rng.normal(size=(len(days),8,16,16)).astype('f4'))),
                       coords={**coords,'channel':list(SURFACE_CHANNELS)})
    surface.time.attrs.update(units='days since 1993-01-01',calendar='proleptic_gregorian')
    surface.to_netcdf(tmp_path/'surface.nc',engine='scipy')
    depth=[0.,100.,300.]
    mask=np.ones((3,16,16),'u1'); mask[:,0]=0
    domain=xr.Dataset(dict(valid_ocean_mask=(('depth','lat','lon'),mask)),
                      coords=dict(depth=depth,lat=coords['lat'],lon=coords['lon']))
    domain.to_netcdf(tmp_path/'domain.nc',engine='scipy')
    stats=tmp_path/'norm.json'
    stats.write_text(json.dumps(dict(mean=[18.,10.,5.],std=[9.,5.,2.],raw_units='degree_C')))
    surface_stats=tmp_path/'surface_norm.json'
    surface_stats.write_text(json.dumps(dict(variables={
        v:dict(mean=0.,std=1.) for v in SURFACE_CHANNELS[:5]})))
    for name,values in [('iprc',[.3,.2,.1]),('glorys',[.6,.2,-.5])]:
        target=np.broadcast_to(np.array(values,'f4')[None,:,None,None],(len(days),3,16,16)).copy()
        valid=np.broadcast_to(mask,target.shape).copy()
        available=np.ones(len(days),'u1')
        if name=='iprc':
            available=np.array([int('2005-01'<=k<='2020-04') for k in keys],'u1')
            valid[available==0]=0; target[available==0]=0
        labels=xr.Dataset(dict(temperature=(('time','depth','lat','lon'),target),
                               valid_label_mask=(('time','depth','lat','lon'),valid),
                               label_available=('time',available)),coords={**coords,'depth':depth})
        labels.time.attrs.update(surface.time.attrs)
        labels.temperature.attrs['source_variable']='PTEMP' if name=='iprc' else 'thetao'
        labels.attrs['standardization_sha256']=hashlib.sha256(stats.read_bytes()).hexdigest()
        labels.to_netcdf(tmp_path/f'{name}.nc',engine='scipy')
    configs={}
    for name in ['iprc_pretrain','glorys_finetune']:
        filename = 'glorys_finetune_grouped_lr' if grouped and name == 'glorys_finetune' else name
        if front:
            filename += '_front'
        cfg=yaml.safe_load((project/f'configs/{filename}.yaml').read_text(encoding='utf-8'))
        cfg['data'].update(surface_path=str(tmp_path/'surface.nc'),
                           label_path=str(tmp_path/('iprc.nc' if name=='iprc_pretrain' else 'glorys.nc')),
                           domain_mask_path=str(tmp_path/'domain.nc'),label_norm_path=str(stats),
                           surface_norm_path=str(surface_stats),time_text_var=None,
                           train_range=['2005-01','2005-02'],val_range=['2019-01','2019-02'],
                           test_range=None if name=='iprc_pretrain' else ['2021-01','2021-02'])
        cfg['model'].update(image_size=[16,16],out_depths=3,embed_dim=8,encoder_depths=[1,1],
                            decoder_depths=[1],num_heads=[1,2],window_size=[2,2],shift_size=[1,1],
                            drop_path_rate=0.,use_checkpoint=False)
        cfg['model']['moe'].update(width=4)
        cfg['training'].update(output_dir=str(tmp_path/name),device='cpu',epochs=1,val_every=1)
        path=tmp_path/f'{name}.yaml'; path.write_text(yaml.safe_dump(cfg))
        configs[name]=(cfg,path)
    env={**os.environ,'PYTHONPATH':str(project/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def run(module,name,*args,fail=None):
        result=subprocess.run([sys.executable,'-m',f'glorys_swin_reconstruction.{module}',
                               '--config',str(configs[name][1]),*args],env=env,capture_output=True,text=True,timeout=120)
        if fail:
            assert result.returncode!=0 and fail in result.stdout+result.stderr,result.stdout+result.stderr
        else:
            assert result.returncode==0,result.stdout+result.stderr
        return result
    run('inspect_data','iprc_pretrain')
    run('train','iprc_pretrain')
    pre=tmp_path/'iprc_pretrain'
    run('train','iprc_pretrain','--resume',str(pre/'last.pt'),'--epochs','2')
    before=torch.load(pre/'best.pt',map_location='cpu',weights_only=False)
    fine_cfg=configs['glorys_finetune'][0]
    model=build_model(fine_cfg['model'])
    initialize_weights(model,pre/'best.pt',fine_cfg,before['data_signature'])
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before['model'][key],rtol=0,atol=0)
    assert before['data_signature']['history_months']==1
    assert len(before['data_signature']['input_channels'])==11
    old=copy.deepcopy(before['data_signature']); old['history_months']=6
    with pytest.raises(ValueError,match='normalization'):
        initialize_weights(model,pre/'best.pt',fine_cfg,old)
    bad=copy.deepcopy(before['data_signature']); bad['std'][0]+=1
    with pytest.raises(ValueError,match='normalization'):
        initialize_weights(model,pre/'best.pt',fine_cfg,bad)
    run('train','glorys_finetune',fail='requires --init-checkpoint')
    run('reconstruct','iprc_pretrain','--split','test',fail='disabled')
    run('train','glorys_finetune','--init-checkpoint',str(pre/'best.pt'))
    fine=tmp_path/'glorys_finetune'
    state=torch.load(fine/'last.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==1 and len(state['history'])==1
    assert not state['initialization']['optimizer_restored']
    assert state['initialization']['moe_statistics_source']['sampled_month_count']==2
    assert max(float(v['step']) for v in state['optimizer']['state'].values())==2
    assert set(state['history'][0]['val']['depth_bands']) == {'0_150m', '200_2000m'}
    if grouped:
        assert [g['name'] for g in state['optimizer']['param_groups']] == ['encoder', 'decoder_head_moe']
        assert [g['lr'] for g in state['optimizer']['param_groups']] == [1e-5, 5e-5]
        assert state['scheduler']['min_lrs'] == [2e-7, 1e-6]
        assert state['history'][0]['lr_groups_used'] == {'encoder': 1e-5, 'decoder_head_moe': 5e-5}
        initialization=json.loads((fine/'initialization.json').read_text())
        assert len(initialization['optimizer_groups']) == 2
        original_lr=fine_cfg['training']['parameter_groups']['encoder']['learning_rate']
        fine_cfg['training']['parameter_groups']['encoder']['learning_rate']=2e-5
        configs['glorys_finetune'][1].write_text(yaml.safe_dump(fine_cfg))
        run('train','glorys_finetune','--resume',str(fine/'last.pt'),fail='identical parameter_groups')
        fine_cfg['training']['parameter_groups']['encoder']['learning_rate']=original_lr
        configs['glorys_finetune'][1].write_text(yaml.safe_dump(fine_cfg))
    assert not torch.equal(state['model']['moe.clim_gradient'],before['model']['moe.clim_gradient'])
    # GLORYS normalized [0.6,0.2,-0.5] -> physical [23.4,11,4].
    expected=np.maximum(0,-np.gradient(np.array([23.4,11.,4.]),depth))
    np.testing.assert_allclose(state['model']['moe.clim_gradient'][0,:,2,2],expected,rtol=1e-5)
    run('train','glorys_finetune','--resume',str(fine/'last.pt'),'--epochs','2')
    assert [r['epoch'] for r in json.loads((fine/'history.json').read_text())]==[1,2]
    if grouped:
        resumed=torch.load(fine/'last.pt',map_location='cpu',weights_only=False)
        assert max(float(v['step']) for v in resumed['optimizer']['state'].values())==4
        assert [g['name'] for g in resumed['optimizer']['param_groups']] == ['encoder', 'decoder_head_moe']
    run('reconstruct','glorys_finetune','--checkpoint',str(fine/'best.pt'),'--split','test')
    run('evaluate_reconstruction','glorys_finetune','--reconstruction',str(fine/'reconstruction_test.nc'),'--split','test')
    metrics=json.loads((fine/'metrics_test_reevaluated.json').read_text())
    assert metrics['historical_comparison']['passed']
    assert metrics['month_count']==2 and metrics['is_complete']
    # 验证顺序入口确实串联两个阶段，并将小样本结果隔离到smoke目录。
    for name,(cfg,path) in configs.items():
        cfg['training']['output_dir']=str(tmp_path/('pipeline_'+name))
        path.write_text(yaml.safe_dump(cfg))
    result=subprocess.run([sys.executable,'-m','glorys_swin_reconstruction.two_stage',
                           '--pretrain-config',str(configs['iprc_pretrain'][1]),
                           '--finetune-config',str(configs['glorys_finetune'][1]),
                           '--max-samples','1','--epochs','1'],env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    smoke=tmp_path/'pipeline_glorys_finetune/smoke/last.pt'
    assert smoke.is_file()
    smoke_pre=tmp_path/'pipeline_iprc_pretrain/smoke/best.pt'
    with pytest.raises(ValueError,match='smoke checkpoint'):
        initialize_weights(model,smoke_pre,fine_cfg,before['data_signature'])
