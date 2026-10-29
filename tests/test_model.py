import pytest
import torch
from torch import nn

from glorys_swin_reconstruction.model import (
    ContinuousGeoReconstructionHead,
    GeoSwinUNet,
    GeoSwinTransformerBlock,
    PatchEmbed,
    PatchExpand,
    PatchMerging,
    PeriodicBilinearUpsample,
    window_partition,
    window_reverse,
)


def tiny_model(reconstruction_head: str = "patch_expand") -> GeoSwinUNet:
    return GeoSwinUNet(
        image_size=(64, 96),
        in_channels=11,
        out_depths=27,
        patch_size=4,
        embed_dim=8,
        encoder_depths=(2, 2, 2, 2),
        num_heads=(1, 2, 4, 8),
        decoder_depths=(2, 2, 2),
        window_size=(2, 3),
        shift_size=(1, 1),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        reconstruction_head=reconstruction_head,
    )


def test_model_forward_shape_and_backward():
    model = tiny_model()
    x = torch.randn(1, 11, 64, 96, requires_grad=True)
    output = model(x)
    assert output.shape == (1, 27, 64, 96)
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_parameter_count_is_positive():
    assert tiny_model().parameter_count() > 0


def test_continuous_head_model_forward_shape_and_backward():
    model = tiny_model("periodic_bilinear_geo_refine")
    x = torch.randn(1, 11, 64, 96, requires_grad=True)
    output = model(x)
    assert output.shape == (1, 27, 64, 96)
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_periodic_bilinear_upsample_is_longitude_shift_equivariant():
    upsample = PeriodicBilinearUpsample(scale=4)
    x = torch.randn(1, 3, 4, 12)
    shifted_low_resolution = torch.roll(x, shifts=1, dims=-1)
    shifted_high_resolution = torch.roll(upsample(x), shifts=4, dims=-1)
    assert torch.allclose(
        upsample(shifted_low_resolution),
        shifted_high_resolution,
        atol=1e-6,
        rtol=1e-6,
    )


def test_periodic_bilinear_upsample_preserves_constant_field():
    x = torch.full((1, 2, 3, 5), 7.25)
    output = PeriodicBilinearUpsample(scale=4)(x)
    assert output.shape == (1, 2, 12, 20)
    assert torch.equal(output, torch.full_like(output, 7.25))


def test_continuous_head_refinement_starts_as_zero_residual():
    head = ContinuousGeoReconstructionHead(
        input_resolution=(4, 6),
        dim=8,
        out_depths=3,
        scale=4,
        norm_eps=1e-5,
    )
    tokens = torch.randn(1, 4 * 6, 8)
    with torch.no_grad():
        feature = tokens.transpose(1, 2).reshape(1, 8, 4, 6)
        expected = head.base_projection(head.upsample(feature))
        actual = head(tokens)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(head.delta_projection.weight) == 0
    assert torch.count_nonzero(head.delta_projection.bias) == 0


def test_shifted_mask_blocks_latitude_but_not_longitude():
    block = GeoSwinTransformerBlock(
        dim=8,
        input_resolution=(4, 6),
        num_heads=1,
        window_size=(4, 3),
        shift_size=(2, 1),
        mlp_ratio=2.0,
        qkv_bias=True,
        drop=0.0,
        attention_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-5,
    )
    mask = block.attention_mask
    assert mask is not None
    assert torch.any(mask < 0)
    x = torch.randn(1, 4 * 6, 8)
    assert block(x).shape == x.shape


def test_rectangular_window_partition_roundtrip_is_exact():
    x = torch.arange(2 * 8 * 18 * 3).view(2, 8, 18, 3)
    windows = window_partition(x, (4, 9))
    restored = window_reverse(windows, (4, 9), 8, 18)
    assert torch.equal(restored, x)


@pytest.mark.parametrize("width", (360, 180, 90, 45))
def test_production_shifted_longitude_windows_have_no_duplicate_tokens(width):
    origins = torch.arange(width).view(1, 1, width, 1).expand(1, 8, width, 1)
    shifted = torch.roll(origins, shifts=(-4, -4), dims=(1, 2))
    windows = window_partition(shifted, (8, 9))
    seam_origins = windows[-1, 0, :, 0].tolist()
    expected = list(range(width - 5, width)) + [0, 1, 2, 3]
    assert seam_origins == expected
    assert len(set(seam_origins)) == 9


def test_nondivisible_window_is_rejected_instead_of_padded():
    with pytest.raises(ValueError, match="not divisible"):
        GeoSwinTransformerBlock(
            dim=8,
            input_resolution=(32, 90),
            num_heads=1,
            window_size=(8, 8),
            shift_size=(4, 4),
            mlp_ratio=2.0,
            qkv_bias=True,
            drop=0.0,
            attention_drop=0.0,
            drop_path=0.0,
            norm_eps=1e-5,
        )


def test_patch_embed_and_merge_preserve_expected_spatial_order():
    embed = PatchEmbed((4, 8), 4, 1, 1, 1e-5)
    embed.norm = nn.Identity()
    with torch.no_grad():
        embed.projection.weight.fill_(1.0)
        embed.projection.bias.zero_()
    image = torch.cat(
        (torch.ones(1, 1, 4, 4), torch.full((1, 1, 4, 4), 2.0)),
        dim=3,
    )
    assert embed(image).flatten().tolist() == [16.0, 32.0]

    merge = PatchMerging((2, 4), 1, 1e-5)
    merge.norm = nn.Identity()
    with torch.no_grad():
        merge.reduction.weight.zero_()
        merge.reduction.weight[0, 0] = 1.0
        merge.reduction.weight[1, 3] = 1.0
    tokens = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]]]
    )
    assert merge(tokens).squeeze(0).tolist() == [[0.0, 11.0], [2.0, 13.0]]


def test_patch_expand_has_expected_two_by_two_spatial_order():
    expand = PatchExpand((1, 2), 2, 1e-5)
    expand.norm = nn.Identity()
    with torch.no_grad():
        expand.expand.weight.zero_()
        expand.expand.weight[0, 0] = 1.0
        expand.expand.weight[1, 1] = 1.0
        expand.expand.weight[2, 0] = 2.0
        expand.expand.weight[3, 1] = 2.0
    tokens = torch.tensor([[[1.0, 10.0], [2.0, 20.0]]])
    grid = expand(tokens).view(1, 2, 4, 1)[0, :, :, 0]
    expected = torch.tensor(
        [[1.0, 10.0, 2.0, 20.0], [2.0, 20.0, 4.0, 40.0]]
    )
    assert torch.equal(grid, expected)
