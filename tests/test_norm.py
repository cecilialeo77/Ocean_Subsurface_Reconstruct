import json

import numpy as np
import pytest

from glorys_swin_reconstruction.norm import DepthNorm


def test_glorys_flat_statistics_and_physical_inverse(tmp_path):
    path = tmp_path / "norm.json"
    path.write_text(json.dumps({"mean": [18., 4.], "std": [9., 2.],
                                "raw_units": "degree_C", "standardized_units": "1"}))
    norm = DepthNorm.from_config({"label_norm_path": str(path)})
    mean, std = norm.numpy_mean_std()
    np.testing.assert_allclose((np.ones((2, 1, 1)) * std + mean).ravel(), [27., 6.])
    assert norm.units == "degree_C"


def test_nonpositive_std_is_rejected(tmp_path):
    path = tmp_path / "norm.json"
    path.write_text(json.dumps({"mean": [0.], "std": [0.]}))
    with pytest.raises(ValueError, match="positive"):
        DepthNorm.from_config({"label_norm_path": str(path)})
