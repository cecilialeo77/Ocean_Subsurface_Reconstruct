from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint as checkpoint_utils

from .moe import ProcessMoE


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError("Expected a scalar or a pair.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.activation(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(
    x: torch.Tensor,
    window_size: int | Sequence[int],
) -> torch.Tensor:
    """Convert B,H,W,C into non-overlapping rectangular windows."""
    batch, height, width, channels = x.shape
    window_height, window_width = _pair(window_size)
    if height % window_height or width % window_width:
        raise ValueError(
            f"Resolution {(height, width)} must be divisible by "
            f"window_size {(window_height, window_width)}."
        )
    x = x.view(
        batch,
        height // window_height,
        window_height,
        width // window_width,
        window_width,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_height, window_width, channels
    )


def window_reverse(
    windows: torch.Tensor,
    window_size: int | Sequence[int],
    height: int,
    width: int,
) -> torch.Tensor:
    window_height, window_width = _pair(window_size)
    if height % window_height or width % window_width:
        raise ValueError(
            f"Resolution {(height, width)} must be divisible by "
            f"window_size {(window_height, window_width)}."
        )
    windows_per_sample = (
        (height // window_height) * (width // window_width)
    )
    if windows.shape[0] % windows_per_sample:
        raise ValueError("Window count is incompatible with the requested resolution.")
    batch = windows.shape[0] // windows_per_sample
    x = windows.view(
        batch,
        height // window_height,
        width // window_width,
        window_height,
        window_width,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        batch, height, width, -1
    )


class WindowAttention(nn.Module):
    """Window multi-head self-attention with learned relative-position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int | Sequence[int],
        num_heads: int,
        qkv_bias: bool,
        attention_drop: float,
        projection_drop: float,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.window_size = _pair(window_size)
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        window_height, window_width = self.window_size
        table_size = (2 * window_height - 1) * (2 * window_width - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_height),
                torch.arange(window_width),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_height - 1
        relative[:, :, 1] += window_width - 1
        relative[:, :, 0] *= 2 * window_width - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attention_drop)
        self.projection = nn.Linear(dim, dim)
        self.projection_drop = nn.Dropout(projection_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        expected_tokens = self.window_size[0] * self.window_size[1]
        if tokens != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} tokens per window, got {tokens}."
            )
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attention = attention.view(
                batch_windows // windows, windows, self.num_heads, tokens, tokens
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = self.attention_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection_drop(self.projection(x))


class GeoSwinTransformerBlock(nn.Module):
    """Swin block with periodic longitude and non-periodic latitude."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int | Sequence[int],
        shift_size: int | Sequence[int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attention_drop: float,
        drop_path: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.input_resolution = _pair(input_resolution)
        self.window_size = _pair(window_size)
        self.shift_size = _pair(shift_size)
        if any(size <= 0 for size in self.window_size):
            raise ValueError("Both window dimensions must be positive.")
        if any(
            shift < 0 or shift >= size
            for shift, size in zip(self.shift_size, self.window_size)
        ):
            raise ValueError("Each shift dimension must lie in [0, window_size).")
        height, width = self.input_resolution
        window_height, window_width = self.window_size
        if height % window_height or width % window_width:
            raise ValueError(
                f"Resolution {self.input_resolution} is not divisible by rectangular "
                f"window {self.window_size}; choose an exact window instead of padding."
            )
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attention = WindowAttention(
            dim, window_size, num_heads, qkv_bias, attention_drop, drop
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.register_buffer("attention_mask", self._make_latitude_mask(), persistent=False)

    def _make_latitude_mask(self) -> torch.Tensor | None:
        """Block north/south roll wrap while deliberately allowing longitude wrap."""
        shift_height, _ = self.shift_size
        if shift_height == 0:
            return None
        height, width = self.input_resolution
        window_height, window_width = self.window_size
        region = torch.zeros((1, height, width, 1))
        slices = (
            slice(0, -window_height),
            slice(-window_height, -shift_height),
            slice(-shift_height, None),
        )
        for label, latitude_slice in enumerate(slices):
            region[:, latitude_slice, :, :] = label
        mask_windows = window_partition(region, self.window_size).view(
            -1, window_height * window_width
        )
        difference = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return difference.masked_fill(difference != 0, -100.0).masked_fill(
            difference == 0, 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError(
                f"Expected {height * width} tokens for {self.input_resolution}, got {tokens}."
            )
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        shift_height, shift_width = self.shift_size
        window_height, window_width = self.window_size
        if shift_height or shift_width:
            x = torch.roll(
                x,
                shifts=(-shift_height, -shift_width),
                dims=(1, 2),
            )
        windows = window_partition(x, self.window_size).view(
            -1, window_height * window_width, channels
        )
        windows = self.attention(windows, self.attention_mask)
        x = window_reverse(
            windows.view(-1, window_height, window_width, channels),
            self.window_size,
            height,
            width,
        )
        if shift_height or shift_width:
            x = torch.roll(
                x,
                shifts=(shift_height, shift_width),
                dims=(1, 2),
            )
        x = x.contiguous().view(batch, tokens, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int, norm_eps: float) -> None:
        super().__init__()
        self.input_resolution = _pair(input_resolution)
        self.dim = int(dim)
        self.norm = nn.LayerNorm(4 * dim, eps=norm_eps)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width or channels != self.dim:
            raise ValueError("PatchMerging received an incompatible tensor shape.")
        if height % 2 or width % 2:
            raise ValueError(f"PatchMerging requires even resolution, got {(height, width)}.")
        x = x.view(batch, height, width, channels)
        x = torch.cat(
            (x[:, 0::2, 0::2], x[:, 1::2, 0::2],
             x[:, 0::2, 1::2], x[:, 1::2, 1::2]),
            dim=-1,
        ).view(batch, -1, 4 * channels)
        return self.reduction(self.norm(x))


class PatchExpand(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int, norm_eps: float) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("PatchExpand input dim must be even.")
        self.input_resolution = _pair(input_resolution)
        self.dim = int(dim)
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width or channels != self.dim:
            raise ValueError("PatchExpand received an incompatible tensor shape.")
        x = self.expand(x).view(batch, height, width, 2, 2, channels // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            batch, height * 2 * width * 2, channels // 2
        )
        return self.norm(x)


class FinalPatchExpand(nn.Module):
    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        scale: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_resolution = _pair(input_resolution)
        self.dim = int(dim)
        self.scale = int(scale)
        self.expand = nn.Linear(dim, scale * scale * dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width or channels != self.dim:
            raise ValueError("FinalPatchExpand received an incompatible tensor shape.")
        scale = self.scale
        x = self.expand(x).view(batch, height, width, scale, scale, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            batch, height * scale, width * scale, channels
        )
        return self.norm(x)


class PeriodicBilinearUpsample(nn.Module):
    """Bilinear upsampling with cyclic longitude and bounded latitude.

    PyTorch's regular bilinear interpolation treats both image axes as
    non-periodic. Here latitude uses that bounded behavior, while longitude
    uses center-aligned linear interpolation with indices wrapped modulo W.
    """

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = int(scale)
        if self.scale <= 0:
            raise ValueError("Upsampling scale must be positive.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}.")
        _, _, height, width = x.shape
        if height <= 0 or width <= 0:
            raise ValueError("Spatial dimensions must be positive.")

        # Interpolate only latitude here. Keeping the longitude size unchanged
        # makes this exactly the ordinary non-periodic vertical interpolation.
        x = F.interpolate(
            x,
            size=(height * self.scale, width),
            mode="bilinear",
            align_corners=False,
        )

        target_width = width * self.scale
        positions = torch.arange(
            target_width, device=x.device, dtype=torch.float32
        )
        source = (positions + 0.5) / self.scale - 0.5
        left_unwrapped = torch.floor(source)
        longitude_weight = (source - left_unwrapped).to(dtype=x.dtype)
        left = left_unwrapped.to(dtype=torch.long).remainder(width)
        right = (left + 1).remainder(width)

        left_values = x.index_select(-1, left)
        right_values = x.index_select(-1, right)
        weight_shape = (1, 1, 1, target_width)
        weight = longitude_weight.view(weight_shape)
        return left_values * (1.0 - weight) + right_values * weight


class GeographicDepthwiseConv2d(nn.Module):
    """Depthwise 3x3 mixing with cyclic lon and reflected lat boundaries."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.convolution = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=0,
            groups=self.channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(x.shape)}."
            )
        if x.shape[-2] < 2:
            raise ValueError("Reflected latitude padding requires height >= 2.")
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1), mode="reflect")
        return self.convolution(x)


class ContinuousGeoReconstructionHead(nn.Module):
    """Continuous x4 field reconstruction without phase-specific subpixels."""

    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        out_depths: int,
        scale: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_resolution = _pair(input_resolution)
        self.dim = int(dim)
        self.upsample = PeriodicBilinearUpsample(scale)
        self.base_projection = nn.Conv2d(self.dim, out_depths, kernel_size=1)
        self.refine = GeographicDepthwiseConv2d(self.dim)
        group_count = min(8, self.dim)
        while self.dim % group_count:
            group_count -= 1
        self.refine_norm = nn.GroupNorm(group_count, self.dim, eps=norm_eps)
        self.refine_activation = nn.GELU()
        self.delta_projection = nn.Conv2d(self.dim, out_depths, kernel_size=1)
        nn.init.zeros_(self.delta_projection.weight)
        nn.init.zeros_(self.delta_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width or channels != self.dim:
            raise ValueError(
                "ContinuousGeoReconstructionHead received an incompatible tensor shape."
            )
        feature = x.transpose(1, 2).reshape(batch, channels, height, width)
        feature = self.upsample(feature)
        base = self.base_projection(feature)
        detail = self.refine_activation(self.refine_norm(self.refine(feature)))
        return base + self.delta_projection(detail)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int | Sequence[int],
        shift_size: int | Sequence[int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attention_drop: float,
        drop_paths: Sequence[float],
        norm_eps: float,
        downsample: bool,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        if len(drop_paths) != depth:
            raise ValueError("drop_paths length must equal layer depth.")
        self.use_checkpoint = bool(use_checkpoint)
        zero_shift = (0, 0)
        self.blocks = nn.ModuleList(
            GeoSwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=zero_shift if index % 2 == 0 else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attention_drop=attention_drop,
                drop_path=float(drop_paths[index]),
                norm_eps=norm_eps,
            )
            for index in range(depth)
        )
        self.downsample = (
            PatchMerging(input_resolution, dim, norm_eps) if downsample else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = checkpoint_utils.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        output = x
        if self.downsample is not None:
            x = self.downsample(x)
        return output, x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_resolution: tuple[int, int],
        output_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int | Sequence[int],
        shift_size: int | Sequence[int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attention_drop: float,
        drop_paths: Sequence[float],
        norm_eps: float,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        if output_dim * 2 != input_dim:
            raise ValueError("Decoder dimensions must halve at each expansion.")
        self.expand = PatchExpand(input_resolution, input_dim, norm_eps)
        self.output_resolution = _pair(output_resolution)
        self.concat_norm = nn.LayerNorm(2 * output_dim, eps=norm_eps)
        self.concat_projection = nn.Linear(2 * output_dim, output_dim, bias=False)
        self.layer = BasicLayer(
            dim=output_dim,
            input_resolution=output_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attention_drop=attention_drop,
            drop_paths=drop_paths,
            norm_eps=norm_eps,
            downsample=False,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        if x.shape[:2] != skip.shape[:2]:
            raise ValueError("Expanded decoder feature and encoder skip do not align.")
        x = self.concat_projection(self.concat_norm(torch.cat((x, skip), dim=-1)))
        x, _ = self.layer(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.image_size = _pair(image_size)
        self.patch_size = int(patch_size)
        if any(size % patch_size for size in self.image_size):
            raise ValueError("image_size must be divisible by patch_size.")
        self.patch_resolution = (
            self.image_size[0] // patch_size,
            self.image_size[1] // patch_size,
        )
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Expected input resolution {self.image_size}, got {tuple(x.shape[-2:])}."
            )
        return self.norm(self.projection(x).flatten(2).transpose(1, 2))


class GeoSwinUNet(nn.Module):
    """U-shaped Swin Transformer for full-field depth regression.

    All encoder and decoder feature mixing is performed by shifted-window
    Transformer blocks. The recommended task head reconstructs a continuous
    grid by periodic bilinear upsampling and lightweight geographic refinement.
    The legacy phase-specific patch-expansion head remains available solely for
    checkpoint and experiment compatibility.
    """

    def __init__(
        self,
        image_size: Sequence[int] = (512, 1440),
        in_channels: int = 11,
        out_depths: int = 27,
        patch_size: int = 4,
        embed_dim: int = 96,
        encoder_depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        decoder_depths: Sequence[int] = (6, 2, 2),
        window_size: int | Sequence[int] = (8, 9),
        shift_size: int | Sequence[int] | None = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attention_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_eps: float = 1e-5,
        use_checkpoint: bool = True,
        reconstruction_head: str = "patch_expand",
        moe: dict | None = None,
    ) -> None:
        super().__init__()
        image_size = _pair(image_size)
        encoder_depths = tuple(map(int, encoder_depths))
        num_heads = tuple(map(int, num_heads))
        decoder_depths = tuple(map(int, decoder_depths))
        window_size = _pair(window_size)
        shift_size = (
            (window_size[0] // 2, window_size[1] // 2)
            if shift_size is None
            else _pair(shift_size)
        )
        levels = len(encoder_depths)
        if levels < 2 or len(num_heads) != levels or len(decoder_depths) != levels - 1:
            raise ValueError("Encoder/head/decoder depth lists have incompatible lengths.")
        total_stride = int(patch_size) * 2 ** (levels - 1)
        if image_size[0] % total_stride or image_size[1] % total_stride:
            raise ValueError(
                f"image_size must be divisible by total stride {total_stride}."
            )
        if any(size <= 0 for size in window_size):
            raise ValueError("Both window dimensions must be positive.")
        if any(
            shift < 0 or shift >= size
            for shift, size in zip(shift_size, window_size)
        ):
            raise ValueError("Each shift dimension must lie in [0, window_size).")

        self.image_size = image_size
        self.in_channels = int(in_channels)
        self.out_depths = int(out_depths)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.window_size = window_size
        self.shift_size = shift_size
        self.reconstruction_head_name = str(reconstruction_head)
        valid_heads = {"patch_expand", "periodic_bilinear_geo_refine"}
        if self.reconstruction_head_name not in valid_heads:
            raise ValueError(
                f"reconstruction_head must be one of {sorted(valid_heads)}, "
                f"got {self.reconstruction_head_name!r}."
            )
        self.patch_embed = PatchEmbed(
            image_size, patch_size, in_channels, embed_dim, norm_eps
        )
        self.position_drop = nn.Dropout(drop_rate)

        base_resolution = self.patch_embed.patch_resolution
        resolutions = [
            (base_resolution[0] // 2 ** level, base_resolution[1] // 2 ** level)
            for level in range(levels)
        ]
        dimensions = [embed_dim * 2 ** level for level in range(levels)]
        total_blocks = sum(encoder_depths) + sum(decoder_depths)
        path_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cursor = 0

        self.encoder = nn.ModuleList()
        for level in range(levels):
            depth = encoder_depths[level]
            rates = path_rates[cursor: cursor + depth]
            cursor += depth
            self.encoder.append(
                BasicLayer(
                    dim=dimensions[level],
                    input_resolution=resolutions[level],
                    depth=depth,
                    num_heads=num_heads[level],
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attention_drop=attention_drop_rate,
                    drop_paths=rates,
                    norm_eps=norm_eps,
                    downsample=level < levels - 1,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.bottleneck_norm = nn.LayerNorm(dimensions[-1], eps=norm_eps)
        self.decoder = nn.ModuleList()
        for decoder_index, target_level in enumerate(range(levels - 2, -1, -1)):
            depth = decoder_depths[decoder_index]
            rates = path_rates[cursor: cursor + depth]
            cursor += depth
            self.decoder.append(
                DecoderLayer(
                    input_dim=dimensions[target_level + 1],
                    output_dim=dimensions[target_level],
                    input_resolution=resolutions[target_level + 1],
                    output_resolution=resolutions[target_level],
                    depth=depth,
                    num_heads=num_heads[target_level],
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attention_drop=attention_drop_rate,
                    drop_paths=rates,
                    norm_eps=norm_eps,
                    use_checkpoint=use_checkpoint,
                )
            )

        if self.reconstruction_head_name == "patch_expand":
            self.final_expand: FinalPatchExpand | None = FinalPatchExpand(
                resolutions[0], embed_dim, patch_size, norm_eps
            )
            self.output_projection: nn.Linear | None = nn.Linear(
                embed_dim, out_depths
            )
            self.continuous_head: ContinuousGeoReconstructionHead | None = None
        else:
            self.final_expand = None
            self.output_projection = None
            self.continuous_head = ContinuousGeoReconstructionHead(
                input_resolution=resolutions[0],
                dim=embed_dim,
                out_depths=out_depths,
                scale=patch_size,
                norm_eps=norm_eps,
            )
        moe_cfg = dict(moe or {})
        if moe_cfg.get("enabled", False) and self.continuous_head is None:
            raise ValueError("MoE requires periodic_bilinear_geo_refine head.")
        self.moe = (
            ProcessMoE(embed_dim, out_depths, resolutions[0], patch_size,
                       width=int(moe_cfg.get("width", 16)),
                       prior_strength=float(moe_cfg.get("prior_strength", 0.5)),
                       local_gradient_weight=float(moe_cfg.get("local_gradient_weight", 0.0)),
                       front=moe_cfg.get("front"))
            if moe_cfg.get("enabled", False) else None
        )

        self.apply(self._initialize_weights)
        if self.continuous_head is not None:
            # Keep the refinement branch inactive at initialization. The base
            # path therefore starts as a continuous interpolation baseline.
            nn.init.zeros_(self.continuous_head.delta_projection.weight)
            nn.init.zeros_(self.continuous_head.delta_projection.bias)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, surface_valid: torch.Tensor | None = None,
                return_routes: bool = False):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input [B,{self.in_channels},H,W], got {tuple(x.shape)}.")
        surface = x
        x = self.position_drop(self.patch_embed(x))
        skips: list[torch.Tensor] = []
        for layer in self.encoder:
            output, x = layer(x)
            skips.append(output)
        x = self.bottleneck_norm(skips[-1])
        for decoder, skip in zip(self.decoder, reversed(skips[:-1])):
            x = decoder(x, skip)
        if self.reconstruction_head_name == "patch_expand":
            if self.final_expand is None or self.output_projection is None:
                raise RuntimeError("Legacy reconstruction head is not initialized.")
            x = self.final_expand(x)
            x = self.output_projection(x)
            return x.permute(0, 3, 1, 2).contiguous()
        if self.continuous_head is None:
            raise RuntimeError("Continuous reconstruction head is not initialized.")
        baseline = self.continuous_head(x)
        if self.moe is None:
            return baseline
        feature = x.transpose(1, 2).reshape(x.shape[0], self.embed_dim,
                                             *self.moe.low_size)
        result = self.moe(feature, surface, baseline, surface_valid, return_routes)
        if return_routes:
            residual, alpha, beta = result
            return baseline + residual, alpha, beta
        return baseline + result

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
