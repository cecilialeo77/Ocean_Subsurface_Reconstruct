import csv
import json

import numpy as np
import pytest
import torch

from glorys_swin_reconstruction.metrics import MaskedMetricAccumulator
from glorys_swin_reconstruction.metric_io import coverage_metadata, monthly_record, write_monthly
from glorys_swin_reconstruction.norm import DepthNorm


def accumulator():
    return MaskedMetricAccumulator(DepthNorm(np.array([10., 20.], dtype='float32'),
                                           np.array([2., 4.], dtype='float32'), 'degree_C'), np.array([0., 100.]))


def test_merge_matches_direct_and_hand_computed():
    y = torch.tensor([[[[1., 3.]], [[2., 6.]]], [[[4., 8.]], [[3., 9.]]]])
    p = y + torch.tensor([1., 3.]).view(2, 1, 1, 1)
    mask = torch.ones_like(y)
    mask[1, 1, 0, 1] = 0
    direct = accumulator()
    direct.update_physical(p, y, mask)
    merged = accumulator()
    monthly = []
    for i in range(2):
        part = accumulator()
        part.update_physical(p[i:i+1], y[i:i+1], mask[i:i+1])
        merged.merge(part)
        monthly.append(part.compute()['pooled_rmse'])
    for name in ('count', 'sse', 'sae', 'sum_y', 'sum_y2'):
        np.testing.assert_array_equal(getattr(direct, name), getattr(merged, name))
    result = merged.compute()
    assert result['count'] == 7
    assert result['pooled_rmse'] == pytest.approx(np.sqrt(31/7))
    assert result['pooled_mae'] == pytest.approx(13/7)
    valid_y = y[mask > 0].double().numpy()
    assert result['pooled_r2'] == pytest.approx(1 - 31/np.sum((valid_y-valid_y.mean())**2))
    assert not np.isclose(np.mean(monthly), result['pooled_rmse'])
    stats = merged.statistics()
    assert sum(row['sse'] for row in stats) == 31


def test_normalized_and_physical_entries_match():
    target = torch.arange(8, dtype=torch.float32).reshape(1,2,2,2)
    pred = target + 0.5
    mask = torch.ones_like(target)
    normalized, physical = accumulator(), accumulator()
    mean = torch.tensor([10.,20.]).view(1,2,1,1)
    std = torch.tensor([2.,4.]).view(1,2,1,1)
    normalized.update(pred, target, mask)
    physical.update_physical(pred*std+mean, target*std+mean, mask)
    assert normalized.compute() == physical.compute()
    assert normalized.compute()['per_depth'][0]['rmse'] == 1.
    assert normalized.compute()['per_depth'][1]['rmse'] == 2.


def test_empty_constant_and_strict_export(tmp_path):
    acc = accumulator()
    y = torch.ones((1,2,1,2))
    mask = torch.tensor([[[[1.,1.]], [[0.,0.]]]])
    acc.update_physical(y+1, y, mask)
    result = acc.compute()
    assert np.isnan(result['per_depth'][0]['r2'])
    assert result['per_depth'][1]['count'] == 0
    cfg = {'test_range': ['2021-01', '2021-01']}
    record = monthly_record('2021-01', 'test', acc)
    paths = write_monthly(tmp_path, 'test', [record], coverage_metadata(['2021-01'],cfg,'test'), result)
    text = paths[0].read_text()
    assert 'NaN' not in text and 'Infinity' not in text
    payload = json.loads(text)
    assert payload['records'][0]['per_depth'][1]['rmse'] is None
    with paths[2].open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and rows[1]['rmse'] == '' and rows[1]['count'] == '0'
    empty = accumulator().compute()
    assert empty['count'] == 0 and np.isnan(empty['pooled_rmse'])
    bad = y.clone()
    bad[0,0,0,0] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        acc.update_physical(bad,y,mask)
    different = accumulator()
    different.depth_values[1] = 200
    with pytest.raises(ValueError, match='Cannot merge'):
        acc.merge(different)


@pytest.mark.parametrize('keys,message', [(['2021-01','2021-01'],'Duplicate'),
                                         (['2021-02','2021-01'],'chronological'),
                                         (['2021-01'],'Missing'),
                                         (['2020-12'],'outside')])
def test_invalid_months(keys,message):
    with pytest.raises(ValueError, match=message):
        coverage_metadata(keys, {'test_range':['2021-01','2021-02']}, 'test')
