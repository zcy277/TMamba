"""Cross-modal guidance and fusion modules used by TMamba.

This file contains only the final scanning strategy described in the paper:
Wide-Angle Spatial Scanning (WAS) for intra-modal guidance and
Modality-Interwoven Scanning (MIS) for inter-modal fusion.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

from .utils.s6 import selective_scan_fn
from .utils.ss2d_utils import LayerNorm, Linear
from .vmamba import LSR, mamba_init


# The first four entries are used by MFNet and the latter four by PST900.
# They preserve the window schedule of the final experimental implementation.
_WINDOW_SIZE_BY_HEIGHT = {
    120: 10,
    60: 10,
    30: 5,
    15: 5,
    180: 10,
    90: 10,
    45: 5,
    23: 8,
}


def _window_size(height: int) -> int:
    try:
        return _WINDOW_SIZE_BY_HEIGHT[height]
    except KeyError as error:
        supported = ", ".join(str(value) for value in _WINDOW_SIZE_BY_HEIGHT)
        raise ValueError(
            f"Unsupported feature-map height {height}; expected one of: {supported}."
        ) from error


def _padded_shape(height: int, width: int, window_size: int) -> tuple[int, int]:
    padded_height = height + (window_size - height % window_size) % window_size
    padded_width = width + (window_size - width % window_size) % window_size
    return padded_height, padded_width


def _pad_to_window(x: torch.Tensor, window_size: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    padded_height, padded_width = _padded_shape(height, width, window_size)
    return F.pad(x, (0, padded_width - width, 0, padded_height - height))


def wide_angle_spatial_scan(x: torch.Tensor) -> torch.Tensor:
    """Create the four WAS sequences used by WAS-SS2D."""
    _, _, height, width = x.shape
    window_size = _window_size(height)
    x = _pad_to_window(x, window_size)

    windows = rearrange(
        x,
        "b c (nh ph) (nw pw) -> b nh nw c ph pw",
        ph=window_size,
        pw=window_size,
    )
    row_scan = rearrange(windows, "b nh nw c ph pw -> b c (nh nw ph pw)")
    column_scan = rearrange(windows, "b nh nw c ph pw -> b c (nw nh ph pw)")

    return torch.stack(
        [
            row_scan,
            column_scan,
            torch.flip(row_scan, dims=[-1]),
            torch.flip(column_scan, dims=[-1]),
        ],
        dim=1,
    )


def wide_angle_spatial_merge(
    sequences: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Merge the four WAS sequences back into a spatial feature map."""
    window_size = _window_size(height)
    padded_height, padded_width = _padded_shape(height, width, window_size)
    windows_high = padded_height // window_size
    windows_wide = padded_width // window_size
    num_windows = windows_high * windows_wide
    window_area = window_size * window_size

    row = sequences[:, 0] + torch.flip(sequences[:, 2], dims=[-1])
    column = sequences[:, 1] + torch.flip(sequences[:, 3], dims=[-1])

    row_windows = rearrange(
        row,
        "b c (num_windows window_area) -> b c num_windows window_area",
        num_windows=num_windows,
        window_area=window_area,
    )
    column_windows = rearrange(
        column,
        "b c (num_windows window_area) -> b c num_windows window_area",
        num_windows=num_windows,
        window_area=window_area,
    )
    column_windows = rearrange(
        column_windows,
        "b c (nw nh) window_area -> b c (nh nw) window_area",
        nh=windows_high,
        nw=windows_wide,
    )

    merged_windows = row_windows + column_windows
    output = rearrange(
        merged_windows,
        "b c (nh nw) (ph pw) -> b c (nh ph) (nw pw)",
        nh=windows_high,
        nw=windows_wide,
        ph=window_size,
        pw=window_size,
    )
    return output[:, :, :height, :width]


def modality_interwoven_scan(
    rgb: torch.Tensor, thermal: torch.Tensor
) -> torch.Tensor:
    """Create four modality-interwoven sequences from RGB and thermal features."""
    _, _, height, _ = rgb.shape
    window_size = _window_size(height)
    rgb = _pad_to_window(rgb, window_size)
    thermal = _pad_to_window(thermal, window_size)

    rgb_windows = rearrange(
        rgb,
        "b c (nh ph) (nw pw) -> b nh nw c ph pw",
        ph=window_size,
        pw=window_size,
    )
    thermal_windows = rearrange(
        thermal,
        "b c (nh ph) (nw pw) -> b nh nw c ph pw",
        ph=window_size,
        pw=window_size,
    )

    rgb_row = rearrange(rgb_windows, "b nh nw c ph pw -> b (nh nw) c ph pw")
    thermal_row = rearrange(
        thermal_windows, "b nh nw c ph pw -> b (nh nw) c ph pw"
    )
    rgb_column = rearrange(
        rgb_windows, "b nh nw c ph pw -> b (nw nh) c ph pw"
    )
    thermal_column = rearrange(
        thermal_windows, "b nh nw c ph pw -> b (nw nh) c ph pw"
    )

    row_scan = rearrange(
        torch.stack([rgb_row, thermal_row], dim=2),
        "b num_windows modalities c ph pw -> b c (num_windows modalities ph pw)",
    )
    column_scan = rearrange(
        torch.stack([rgb_column, thermal_column], dim=2),
        "b num_windows modalities c ph pw -> b c (num_windows modalities ph pw)",
    )

    return torch.stack(
        [
            row_scan,
            column_scan,
            torch.flip(row_scan, dims=[-1]),
            torch.flip(column_scan, dims=[-1]),
        ],
        dim=1,
    )


def modality_interwoven_merge(
    sequences: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Merge modality-interwoven sequences into one fused feature map."""
    window_size = _window_size(height)
    padded_height, padded_width = _padded_shape(height, width, window_size)
    windows_high = padded_height // window_size
    windows_wide = padded_width // window_size
    num_windows = windows_high * windows_wide
    window_area = window_size * window_size

    row = sequences[:, 0] + torch.flip(sequences[:, 2], dims=[-1])
    column = sequences[:, 1] + torch.flip(sequences[:, 3], dims=[-1])

    row_modalities = rearrange(
        row,
        "b c (num_windows modalities window_area) -> modalities b c num_windows window_area",
        modalities=2,
        num_windows=num_windows,
        window_area=window_area,
    )
    column_modalities = rearrange(
        column,
        "b c (num_windows modalities window_area) -> modalities b c num_windows window_area",
        modalities=2,
        num_windows=num_windows,
        window_area=window_area,
    )

    rgb_row, thermal_row = row_modalities[0], row_modalities[1]
    rgb_column, thermal_column = column_modalities[0], column_modalities[1]
    rgb_column = rearrange(
        rgb_column,
        "b c (nw nh) window_area -> b c (nh nw) window_area",
        nh=windows_high,
        nw=windows_wide,
    )
    thermal_column = rearrange(
        thermal_column,
        "b c (nw nh) window_area -> b c (nh nw) window_area",
        nh=windows_high,
        nw=windows_wide,
    )

    fused_windows = rgb_row + rgb_column + thermal_row + thermal_column
    output = rearrange(
        fused_windows,
        "b c (nh nw) (ph pw) -> b c (nh ph) (nw pw)",
        nh=windows_high,
        nw=windows_wide,
        ph=window_size,
        pw=window_size,
    )
    return output[:, :, :height, :width]


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.project_in = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.activation = nn.GELU()
        self.project_out = nn.Conv2d(channels // ratio, channels, 1, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.project_out(self.activation(self.project_in(self.max_pool(x))))
        return self.gate(attention)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("Spatial-attention kernel size must be 3 or 7.")
        padding = 3 if kernel_size == 7 else 1
        self.projection = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maximum = torch.max(x, dim=1, keepdim=True)[0]
        return self.gate(self.projection(maximum))


class AlternatingPurificationAttention(nn.Module):
    """Alternately purify thermal and RGB features with channel/spatial attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.rgb_channel_attention = ChannelAttention(channels)
        self.rgb_spatial_attention = SpatialAttention()
        self.thermal_channel_attention = ChannelAttention(channels)
        self.thermal_spatial_attention = SpatialAttention()

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        rgb_attention = rgb * self.rgb_channel_attention(rgb)
        refined_thermal = thermal + thermal * self.rgb_spatial_attention(rgb_attention)

        thermal_attention = refined_thermal * self.thermal_channel_attention(
            refined_thermal
        )
        refined_rgb = rgb + rgb * self.thermal_spatial_attention(thermal_attention)
        return refined_rgb


class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = rearrange(self.avg_pool(x), "b c 1 1 -> b 1 c")
        attention = self.projection(attention)
        attention = rearrange(attention, "b 1 c -> b c 1 1")
        return x * self.gate(attention).expand_as(x)


class _SelectiveScan2D(nn.Module):
    """Shared S6 parameterization for the two paper-specific SS2D modules."""

    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        ssm_ratio: float,
        dt_rank: Any,
        dropout: float,
        bias: bool,
        dt_min: float,
        dt_max: float,
        dt_init: str,
        dt_scale: float,
        dt_init_floor: float,
    ):
        super().__init__()
        self.num_directions = 4
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.inner_dim = int(ssm_ratio * model_dim)
        self.dt_rank = int(
            math.ceil(self.model_dim / 16) if dt_rank == "auto" else dt_rank
        )

        self.x_projection = Linear(
            self.inner_dim,
            self.num_directions * (self.dt_rank + self.state_dim * 2),
            groups=self.num_directions,
            bias=False,
            channel_first=True,
        )
        self.dt_projection = Linear(
            self.dt_rank,
            self.num_directions * self.inner_dim,
            groups=self.num_directions,
            bias=False,
            channel_first=True,
        )
        (
            self.A_logs,
            self.Ds,
            dt_projection_weight,
            self.dt_projs_bias,
        ) = mamba_init.init_dt_A_D(
            self.state_dim,
            self.dt_rank,
            self.inner_dim,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
            k_group=self.num_directions,
        )
        self.dt_projection.weight.data = dt_projection_weight.data.view(
            self.dt_projection.weight.shape
        )
        self.output_norm = LayerNorm(self.inner_dim, channel_first=True)
        self.output_projection = Linear(
            self.inner_dim, self.model_dim, bias=bias, channel_first=True
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _run_selective_scan(self, sequences: torch.Tensor) -> torch.Tensor:
        batch, _, _, length = sequences.shape
        projected = self.x_projection(sequences.view(batch, -1, length))
        dts, Bs, Cs = torch.split(
            projected.view(batch, self.num_directions, -1, length),
            [self.dt_rank, self.state_dim, self.state_dim],
            dim=2,
        )
        dts = self.dt_projection(dts.contiguous().view(batch, -1, length))

        xs = sequences.contiguous().view(batch, -1, length)
        dts = dts.contiguous().view(batch, -1, length)
        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        Bs = Bs.contiguous().view(
            batch, self.num_directions, self.state_dim, length
        )
        Cs = Cs.contiguous().view(
            batch, self.num_directions, self.state_dim, length
        )
        delta_bias = self.dt_projs_bias.view(-1).float()

        return selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            delta_bias=delta_bias,
            delta_softplus=True,
        ).view(batch, self.num_directions, -1, length)


class WideAngleSpatialSS2D(_SelectiveScan2D):
    """Local Spatial SSM with Wide-Angle Spatial Scanning."""

    def __init__(
        self,
        model_dim: int,
        state_dim: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: Any = "auto",
        activation: type[nn.Module] = nn.SiLU,
        conv_kernel: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__(
            model_dim,
            state_dim,
            ssm_ratio,
            dt_rank,
            dropout,
            bias,
            dt_min,
            dt_max,
            dt_init,
            dt_scale,
            dt_init_floor,
        )
        self.input_projection = Linear(
            self.model_dim, self.inner_dim * 2, bias=bias, channel_first=True
        )
        self.purification_attention = AlternatingPurificationAttention(
            self.inner_dim
        )
        self.activation = activation()
        self.depthwise_conv = (
            nn.Conv2d(
                self.inner_dim,
                self.inner_dim,
                groups=self.inner_dim,
                bias=conv_bias,
                kernel_size=conv_kernel,
                padding=(conv_kernel - 1) // 2,
            )
            if conv_kernel > 1
            else None
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        aggregated = self.purification_attention(rgb, thermal) + rgb + thermal
        content, gate = self.input_projection(aggregated).chunk(2, dim=1)
        gate = self.activation(gate)

        if self.depthwise_conv is not None:
            content = self.activation(self.depthwise_conv(content))

        _, _, height, width = content.shape
        sequences = wide_angle_spatial_scan(content)
        output = self._run_selective_scan(sequences)
        output = wide_angle_spatial_merge(output, height, width)
        output = self.output_norm(output) * gate
        return self.dropout(self.output_projection(output))


class ModalityInterwovenSS2D(_SelectiveScan2D):
    """Cross-Modal Fusion SSM with Modality-Interwoven Scanning."""

    def __init__(
        self,
        model_dim: int,
        state_dim: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: Any = "auto",
        activation: type[nn.Module] = nn.SiLU,
        conv_kernel: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__(
            model_dim,
            state_dim,
            ssm_ratio,
            dt_rank,
            dropout,
            bias,
            dt_min,
            dt_max,
            dt_init,
            dt_scale,
            dt_init_floor,
        )
        self.rgb_projection = Linear(
            self.model_dim, self.inner_dim * 2, bias=bias, channel_first=True
        )
        self.thermal_projection = Linear(
            self.model_dim, self.inner_dim * 2, bias=bias, channel_first=True
        )
        self.channel_attention = ChannelAttention(self.inner_dim)
        self.spatial_attention = SpatialAttention()
        self.activation = activation()
        self.depthwise_conv = (
            nn.Conv2d(
                self.inner_dim,
                self.inner_dim,
                groups=self.inner_dim,
                bias=conv_bias,
                kernel_size=conv_kernel,
                padding=(conv_kernel - 1) // 2,
            )
            if conv_kernel > 1
            else None
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        rgb_content, rgb_gate = self.rgb_projection(rgb).chunk(2, dim=1)
        thermal_content, thermal_gate = self.thermal_projection(thermal).chunk(
            2, dim=1
        )

        joint_gate = rgb_gate + thermal_gate
        joint_gate = joint_gate * self.channel_attention(joint_gate)
        joint_gate = joint_gate * self.spatial_attention(joint_gate)
        rgb_gate = self.activation(rgb_gate * joint_gate)
        thermal_gate = self.activation(thermal_gate * joint_gate)

        if self.depthwise_conv is not None:
            rgb_content = self.activation(self.depthwise_conv(rgb_content))
            thermal_content = self.activation(self.depthwise_conv(thermal_content))

        _, _, height, width = rgb_content.shape
        sequences = modality_interwoven_scan(rgb_content, thermal_content)
        output = self._run_selective_scan(sequences)
        output = modality_interwoven_merge(output, height, width)
        output = self.output_norm(output)
        output = output * rgb_gate + output * thermal_gate
        return self.dropout(self.output_projection(output))


class CrossModalGuidanceModule(nn.Module):
    """Generate consistent guidance with APA, WAS-SS2D, and ECA."""

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        state_dim: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: Any = "auto",
        conv_kernel: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rgb_norm = LayerNorm(hidden_dim, channel_first=True)
        self.thermal_norm = LayerNorm(hidden_dim, channel_first=True)
        self.wide_angle_ss2d = WideAngleSpatialSS2D(
            model_dim=hidden_dim,
            state_dim=state_dim,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
            conv_bias=conv_bias,
            dropout=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.output_norm = LayerNorm(hidden_dim, channel_first=True)
        self.channel_attention = EfficientChannelAttention()

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        guidance = self.wide_angle_ss2d(
            self.rgb_norm(rgb), self.thermal_norm(thermal)
        )
        guidance = rgb + thermal + self.drop_path(guidance)
        attention = self.channel_attention(self.output_norm(guidance))
        return guidance + self.drop_path(attention)


class InterModalFusionModule(nn.Module):
    """Full CGM-to-IFM fusion path used at one TMamba encoder stage."""

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        state_dim: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: Any = "auto",
        conv_kernel: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        lsr_ratio: float = 4.0,
        lsr_activation: type[nn.Module] = nn.GELU,
        lsr_dropout: float = 0.0,
    ):
        super().__init__()
        self.guidance_module = CrossModalGuidanceModule(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            state_dim=state_dim,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
            conv_bias=conv_bias,
            dropout=dropout,
        )
        self.rgb_norm = LayerNorm(hidden_dim, channel_first=True)
        self.thermal_norm = LayerNorm(hidden_dim, channel_first=True)
        self.modality_interwoven_ss2d = ModalityInterwovenSS2D(
            model_dim=hidden_dim,
            state_dim=state_dim,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
            conv_bias=conv_bias,
            dropout=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.lsr_norm = LayerNorm(hidden_dim, channel_first=True)
        self.lsr = LSR(
            dim=hidden_dim,
            hidden_dim=int(hidden_dim * lsr_ratio),
            act_layer=lsr_activation,
            drop=lsr_dropout,
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        guidance = self.guidance_module(rgb, thermal)
        refined_rgb = rgb + guidance
        refined_thermal = thermal + guidance

        fused = refined_rgb + refined_thermal
        fused = fused + self.drop_path(
            self.modality_interwoven_ss2d(
                self.rgb_norm(refined_rgb), self.thermal_norm(refined_thermal)
            )
        )

        normalized = self.lsr_norm(fused)
        batch, channels, height, width = normalized.shape
        sequence = normalized.flatten(2).transpose(1, 2)
        regularized = self.lsr(sequence, img_size=(height, width))
        regularized = regularized.permute(0, 3, 1, 2).contiguous()
        return fused + self.drop_path(regularized)
