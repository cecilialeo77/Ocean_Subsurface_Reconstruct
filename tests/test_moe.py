import torch

from glorys_swin_reconstruction.model import GeoSwinUNet
from glorys_swin_reconstruction.moe import gradient_100km


def make_model(moe):
    return GeoSwinUNet(
        image_size=(64, 96), in_channels=11, out_depths=3,
        patch_size=4, embed_dim=8, encoder_depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 8), decoder_depths=(1, 1, 1),
        window_size=(2, 3), shift_size=(1, 1), use_checkpoint=False,
        reconstruction_head="periodic_bilinear_geo_refine",
        moe={"enabled": True, "width": 4} if moe else None,
    )


def test_zero_residual_matches_shared_baseline_and_routes_are_normalized():
    torch.manual_seed(14)
    baseline = make_model(False)
    torch.manual_seed(14)
    model = make_model(True)
    missing = model.load_state_dict(baseline.state_dict(), strict=False).missing_keys
    assert missing and all(name.startswith("moe.") for name in missing)
    baseline.eval()
    model.eval()
    x = torch.randn(1, 11, 64, 96)
    x[:, 8] = 1
    with torch.no_grad():
        expected = baseline(x)
        actual, alpha, beta = model(x, return_routes=True)
    assert torch.equal(actual, expected)
    assert alpha.shape == beta.shape == (1, 3, 4, 16, 24)
    assert torch.allclose(alpha.sum(2), torch.ones_like(alpha[:, :, 0]), atol=1e-6)
    assert torch.allclose(beta.sum(2), torch.ones_like(beta[:, :, 0]), atol=1e-6)


def test_masked_gradient_has_no_missing_data_edge():
    field = torch.zeros(1, 1, 4, 8)
    field[..., 2:4] = 10
    valid = torch.ones_like(field)
    valid[..., 2:4] = 0
    latitude = torch.zeros_like(field)
    gradient = gradient_100km(field, valid, latitude)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 0



def test_nonzero_residual_trains_both_routers():
    model = make_model(True)
    with torch.no_grad():
        model.moe.residual_scale.fill_(0.1)
    x = torch.randn(1, 11, 64, 96)
    x[:, 8] = 1
    model(x).square().mean().backward()
    assert model.moe.variable_router[0].weight.grad is not None
    assert model.moe.process_router.weight.grad is not None
    assert model.moe.heave.weight.grad is not None



def test_local_cooling_gradient_tracks_predicted_thermocline():
    model = make_model(True)
    model.moe.set_statistics(
        depth_m=torch.tensor([0.0, 50.0, 100.0]),
        temperature_mean=torch.tensor([10.0, 10.0, 10.0]),
        temperature_std=torch.tensor([2.0, 1.0, 0.5]),
    )
    ocean = torch.ones(1, 1, 64, 96)
    shallow = torch.tensor([0.0, -1.0, -2.0]).view(1, 3, 1, 1).expand(1, 3, 64, 96)
    deep = torch.tensor([0.0, 0.0, -2.0]).view(1, 3, 1, 1).expand(1, 3, 64, 96)
    shallow_gradient = model.moe.local_cooling_gradient(shallow, ocean)
    deep_gradient = model.moe.local_cooling_gradient(deep, ocean)
    assert shallow_gradient.shape == (1, 3, 16, 24)
    assert shallow_gradient[0, 0, 5, 5] > shallow_gradient[0, 2, 5, 5]
    assert deep_gradient[0, 2, 5, 5] > deep_gradient[0, 0, 5, 5]

    ocean[..., :4, :4] = 0
    masked = model.moe.local_cooling_gradient(shallow, ocean)
    assert torch.count_nonzero(masked[..., 0, 0]) == 0
    assert torch.isfinite(masked).all()


def test_process_route_responds_to_current_baseline():
    model = make_model(True)
    model.moe.local_gradient_weight = 1.0
    model.moe.set_statistics(
        depth_m=torch.tensor([0.0, 50.0, 100.0]),
        temperature_mean=torch.zeros(3),
        temperature_std=torch.ones(3),
    )
    feature = torch.zeros(1, 8, 16, 24)
    surface = torch.zeros(1, 11, 64, 96)
    surface[:, 8] = 1
    surface[:, 10] = 1
    shallow = torch.tensor([10.0, 9.0, 9.0]).view(1, 3, 1, 1).expand(1, 3, 64, 96)
    deep = torch.tensor([10.0, 10.0, 9.0]).view(1, 3, 1, 1).expand(1, 3, 64, 96)
    with torch.no_grad():
        _, _, route_shallow = model.moe(feature, surface, shallow, return_routes=True)
        _, _, route_deep = model.moe(feature, surface, deep, return_routes=True)
    assert not torch.allclose(route_shallow, route_deep)
    assert torch.allclose(route_shallow.sum(2), torch.ones_like(route_shallow[:, :, 0]))


def test_old_climatology_only_checkpoint_loads_strictly():
    model = make_model(True)
    old_state = model.state_dict()
    del old_state["moe.temperature_mean"]
    make_model(True).load_state_dict(old_state, strict=True)
