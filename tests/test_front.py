"""Geometry, boundary, gradient and MoE integration checks for FRONT."""
import copy
import pytest
import torch
from torch.nn import functional as F

from glorys_swin_reconstruction.front import AnisotropicFront, periodic_bilinear_sample
from glorys_swin_reconstruction.moe import ProcessMoE


def surface_field(h=8, w=12, scale=2):
    x = torch.zeros(1, 11, h * scale, w * scale)
    x[:, 0] = torch.arange(w * scale).view(1, 1, -1)
    x[:, 8] = 1
    x[:, 10] = 1
    return x


def test_sampler_matches_periodic_grid_sample_and_has_finite_gradients():
    torch.manual_seed(7)
    source = torch.randn(2, 3, 5, 7, requires_grad=True)
    dx = (torch.rand(2, 4, 5, 7) * 5 - 2.5).requires_grad_()
    dy = (torch.rand(2, 4, 5, 7) * 5 - 2.5).requires_grad_()
    actual = periodic_bilinear_sample(source, dx, dy)
    yy, xx = torch.meshgrid(torch.arange(5), torch.arange(7), indexing="ij")
    references = []
    tiled = torch.cat((source, source, source), -1)
    for k in range(4):
        gx = 2 * (xx + 7 + dx[:, k]) / 20 - 1
        gy = 2 * (yy + dy[:, k]) / 4 - 1
        references.append(F.grid_sample(tiled, torch.stack((gx, gy), -1),
                                       padding_mode="border", align_corners=True))
    torch.testing.assert_close(actual, torch.stack(references, 2), atol=3e-6, rtol=3e-6)
    actual.square().mean().backward()
    for value in (source, dx, dy):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_sampler_wraps_longitude_and_replicates_latitude():
    field = torch.arange(12.).reshape(1, 1, 3, 4)
    dx, dy = torch.ones(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)
    torch.testing.assert_close(periodic_bilinear_sample(field, dx, dy)[:, :, 0], field.roll(-1, -1))
    actual = periodic_bilinear_sample(field, dx * 0, dy + 20)[:, :, 0]
    torch.testing.assert_close(actual, field[:, :, -1:].expand_as(field))


def test_frame_direction_flat_field_and_missing_stencils():
    module = AnisotropicFront(4, 2)
    surface = surface_field()
    nx, ny, confidence, *_ = module.frame(surface, None)
    assert nx[0, 0, 3, 5] > 0.99 and abs(ny[0, 0, 3, 5]) < 1e-6
    assert confidence[0, 0, 3, 5] > 0
    flat = surface.clone(); flat[:, 0] = 3
    assert torch.count_nonzero(module.frame(flat, None)[2]) == 0
    valid = torch.ones(1, 5, 16, 24); valid[:, 0, :, 8:12] = 0
    surface[:, 0, :, 8:12] = float("nan")
    frame = module.frame(surface, valid)
    assert all(torch.isfinite(t).all() for t in frame)
    assert torch.count_nonzero(frame[2][..., 3:7]) == 0


def test_geometry_changes_with_depth_and_ablation_removes_change():
    module = AnisotropicFront(4, 2, tangent_radius_km=30, normal_radius_km=20)
    frame = module.frame(surface_field(), None)
    coefficients = torch.zeros(1, 15, 8, 12)
    coefficients[:, 1] = 1.0  # tangent-radius linear depth coefficient
    coefficients[:, 7] = 0.5  # angle linear depth coefficient
    near = module.sampling_geometry(coefficients, torch.tensor(0.), frame)
    deep = module.sampling_geometry(coefficients, torch.tensor(1.), frame)
    assert not torch.allclose(near[0], deep[0])
    assert not torch.allclose(near[1], deep[1])
    module.depth_conditioning = False
    a = module.sampling_geometry(coefficients, torch.tensor(0.), frame)
    b = module.sampling_geometry(coefficients, torch.tensor(1.), frame)
    torch.testing.assert_close(a[0], b[0]); torch.testing.assert_close(a[1], b[1])


def test_high_latitude_offsets_bounded_and_coast_values_excluded():
    module = AnisotropicFront(4, 2)
    surface = surface_field(); surface[:, 5] = 0.99999
    frame = module.frame(surface, None)
    coeff = torch.full((1, 15, 8, 12), 100.)
    dx, dy, _ = module.sampling_geometry(coeff, torch.tensor(1.), frame)
    assert dx.abs().max() <= 4.00001 and dy.abs().max() <= 4.00001
    # Fully masked coarse cells never contribute their values to ocean samples.
    frame = list(module.frame(surface_field(), None))
    frame[-1][..., 5:7] = 0
    with torch.no_grad():
        module.projection.weight.normal_()
    feature = torch.randn(1, 4, 8, 12)
    changed = feature.clone(); changed[..., 5:7] = 1e5
    a = module(feature, coeff * 0, torch.tensor(.5), frame)
    b = module(changed, coeff * 0, torch.tensor(.5), frame)
    ocean = frame[-1].expand_as(a) > 0
    torch.testing.assert_close(a[ocean], b[ocean])


def test_anisotropic_operator_is_longitude_shift_equivariant():
    module = AnisotropicFront(4, 2)
    with torch.no_grad():
        module.projection.weight.normal_(std=.1)
        module.geometry_head.weight.normal_(std=.01)
    surface, feature = surface_field(), torch.randn(1, 4, 8, 12)
    context = torch.randn_like(feature)
    out = module(feature, module.coefficients(context), torch.tensor(.5), module.frame(surface, None))
    shifted = module(feature.roll(2, -1), module.coefficients(context.roll(2, -1)), torch.tensor(.5),
                     module.frame(surface.roll(4, -1), None))
    torch.testing.assert_close(out.roll(2, -1), shifted, atol=3e-6, rtol=3e-6)


def make_moe(front):
    module = ProcessMoE(4, 3, (8, 12), 2, width=4, front=front)
    with torch.no_grad():
        module.residual_scale.fill_(0.2)
    return module


def test_zero_projection_preserves_fixed_front_and_routes():
    fixed = make_moe(None).eval()
    model = make_moe(dict(type="anisotropic")).eval()
    missing = model.load_state_dict(fixed.state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith("front_anisotropic.") for k in missing)
    feature, surface = torch.randn(1, 4, 8, 12), surface_field()
    baseline = torch.zeros(1, 3, 16, 24)
    with torch.no_grad():
        expected = fixed(feature, surface, baseline, return_routes=True)
        actual = model(feature, surface, baseline, return_routes=True)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Strict loading must reject a legacy checkpoint rather than hide new keys.
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(fixed.state_dict(), strict=True)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_branch_and_geometry_train_with_checkpointing_and_autocast(checkpointing):
    model = make_moe(dict(type="anisotropic", checkpoint=checkpointing))
    with torch.no_grad():
        model.front_anisotropic.projection.weight.normal_(std=.1)
    feature = torch.randn(1, 4, 8, 12, requires_grad=True)
    surface = surface_field()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(feature, surface, torch.zeros(1, 3, 16, 24))
        loss = output.float().square().mean()
    loss.backward()
    for value in (feature, model.front_anisotropic.geometry_head.weight,
                  model.front_anisotropic.projection.weight):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    depth_grad = model.front_anisotropic.geometry_head.weight.grad.reshape(5, 3, 4, 1, 1)
    assert depth_grad[:, 1:].abs().sum() > 0


def test_checkpointed_branch_matches_plain_gradients():
    torch.manual_seed(14)
    a = make_moe(dict(type="anisotropic", checkpoint=False))
    with torch.no_grad():
        a.front_anisotropic.projection.weight.normal_(std=.1)
    b = copy.deepcopy(a); b.front_checkpoint = True
    feature = torch.randn(1, 4, 8, 12)
    surface, baseline = surface_field(), torch.zeros(1, 3, 16, 24)
    ya, yb = a(feature, surface, baseline), b(feature, surface, baseline)
    torch.testing.assert_close(ya, yb)
    ya.square().mean().backward(); yb.square().mean().backward()
    for (ka, pa), (kb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert ka == kb
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad)


@pytest.mark.parametrize("option", [dict(tangent_radius_km=0), dict(max_shift_km=-1),
                                    dict(max_offset_cells=float("nan"))])
def test_invalid_geometry_options_rejected(option):
    with pytest.raises(ValueError):
        AnisotropicFront(4, 2, **option)


def test_subpixel_negative_longitude_at_seam_does_not_round_out_of_bounds():
    field = torch.arange(16.).reshape(1, 1, 4, 4)
    dx = torch.full((1, 1, 4, 4), -1e-8, requires_grad=True)
    dy = torch.zeros_like(dx)
    sampled = periodic_bilinear_sample(field, dx, dy)
    torch.testing.assert_close(sampled[:, :, 0], field, atol=1e-6, rtol=1e-6)
    sampled.sum().backward()
    assert torch.isfinite(dx.grad).all()
