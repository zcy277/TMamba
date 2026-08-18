"""Dual Three-Stream Adapter VMamba encoder for TMamba."""

import torch
import torch.nn as nn

from .fusion import InterModalFusionModule
from .utils.ss2d_utils import LayerNorm, PatchMerging2D
from .vmamba import VSSM


VMAMBA_TINY_DIM = 96
VMAMBA_TINY_CONFIG = {
    "depths": [2, 2, 9, 2],
    "dims": VMAMBA_TINY_DIM,
    "mlp_ratio": 0.0,
    "downsample_version": "v1",
    "drop_path_rate": 0.2,
    "norm_layer": "ln2d",
}
VMAMBA_TINY_PRETRAINED = "pretrained/vssmtiny_dp01_ckpt_epoch_292.pth"


class AdapterBranch(nn.Module):
    """Bottleneck adapter branch: Linear-DWConv-Linear with GeLU."""

    def __init__(self, input_dim: int):
        super().__init__()
        hidden_dim = input_dim // 2
        self.down_projection = nn.Linear(input_dim, hidden_dim)
        self.depthwise_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
            bias=True,
        )
        self.up_projection = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

        nn.init.zeros_(self.depthwise_conv.weight)
        if self.depthwise_conv.bias is not None:
            nn.init.zeros_(self.depthwise_conv.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = features.permute(0, 2, 3, 1)
        output = self.activation(self.down_projection(output))
        output = output.permute(0, 3, 1, 2)
        output = self.depthwise_conv(output)
        output = output.permute(0, 2, 3, 1)
        output = self.activation(output)
        output = self.activation(self.up_projection(output))
        return output.permute(0, 3, 1, 2)


class DualTAVMambaBackbone(VSSM):
    """Shared VMamba-Tiny with RGB, thermal, and shared adapter branches."""

    def __init__(
        self,
        pretrained: str | None = VMAMBA_TINY_PRETRAINED,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        apply_pst_patchmerge_fix: bool = False,
    ):
        super().__init__(**VMAMBA_TINY_CONFIG)
        self.out_indices = out_indices
        self.pst_patchmerge_applied = apply_pst_patchmerge_fix

        self.shared_adapters = nn.ModuleList()
        self.rgb_adapters = nn.ModuleList()
        self.thermal_adapters = nn.ModuleList()
        for stage_index, stage_depth in enumerate(VMAMBA_TINY_CONFIG["depths"]):
            stage_dim = self.dims[stage_index]
            self.shared_adapters.append(
                nn.ModuleList(AdapterBranch(stage_dim) for _ in range(stage_depth))
            )
            self.rgb_adapters.append(
                nn.ModuleList(AdapterBranch(stage_dim) for _ in range(stage_depth))
            )
            self.thermal_adapters.append(
                nn.ModuleList(AdapterBranch(stage_dim) for _ in range(stage_depth))
            )

        if self.pst_patchmerge_applied:
            # Samba was designed for single-modal 2D inputs. PST900 reaches an odd
            # spatial size before the last stage, so retain the explicit padded
            # PatchMerging2D adaptation used by the final experimental path.
            target_stage = self.num_layers - 2
            self.layers[target_stage].downsample = PatchMerging2D(
                dim=self.dims[target_stage],
                out_dim=self.dims[target_stage + 1],
                norm_layer=LayerNorm,
                channel_first=self.channel_first,
            )

        for stage_index in self.out_indices:
            self.add_module(
                f"outnorm{stage_index}",
                LayerNorm(
                    self.dims[stage_index], channel_first=self.channel_first
                ),
            )

        self.fusion_modules = nn.ModuleList(
            InterModalFusionModule(
                hidden_dim=self.dims[stage_index],
                drop_path=0.0,
                state_dim=16,
                ssm_ratio=1.0,
                dt_rank="auto",
                conv_kernel=3,
                conv_bias=True,
                dropout=0.0,
                lsr_ratio=4.0,
                lsr_dropout=0.0,
            )
            for stage_index in range(self.num_layers)
        )

        if hasattr(self, "classifier"):
            del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is None:
            return

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu")
            )
            incompatible_keys = self.load_state_dict(
                checkpoint["model"], strict=False
            )
            print(f"Loaded VMamba-Tiny weights from {checkpoint_path}")
            print(f"Incompatible pretrained keys: {incompatible_keys}")
        except Exception as error:
            print(f"Could not load VMamba-Tiny weights from {checkpoint_path}: {error}")

    def forward(
        self, rgb: torch.Tensor, thermal: torch.Tensor
    ) -> list[torch.Tensor]:
        rgb = self.patch_embed(rgb)
        thermal = self.patch_embed(thermal)

        if self.pos_embed is not None:
            position = (
                self.pos_embed
                if self.channel_first
                else self.pos_embed.permute(0, 2, 3, 1)
            )
            rgb = rgb + position
            thermal = thermal + position

        fused_features = []
        for stage_index in range(self.num_layers):
            for block_index, vss_block in enumerate(
                self.layers[stage_index].blocks
            ):
                rgb_channels_first = (
                    rgb
                    if self.channel_first
                    else rgb.permute(0, 3, 1, 2).contiguous()
                )
                thermal_channels_first = (
                    thermal
                    if self.channel_first
                    else thermal.permute(0, 3, 1, 2).contiguous()
                )

                shared_adapter = self.shared_adapters[stage_index][block_index]
                rgb_adapter = self.rgb_adapters[stage_index][block_index]
                thermal_adapter = self.thermal_adapters[stage_index][block_index]

                rgb_channels_first = (
                    rgb_channels_first
                    + shared_adapter(rgb_channels_first)
                    + rgb_adapter(rgb_channels_first)
                )
                thermal_channels_first = (
                    thermal_channels_first
                    + shared_adapter(thermal_channels_first)
                    + thermal_adapter(thermal_channels_first)
                )

                rgb = (
                    rgb_channels_first
                    if self.channel_first
                    else rgb_channels_first.permute(0, 2, 3, 1).contiguous()
                )
                thermal = (
                    thermal_channels_first
                    if self.channel_first
                    else thermal_channels_first.permute(0, 2, 3, 1).contiguous()
                )
                rgb, thermal = vss_block(rgb), vss_block(thermal)

            rgb_for_fusion = (
                rgb
                if self.channel_first
                else rgb.permute(0, 3, 1, 2).contiguous()
            )
            thermal_for_fusion = (
                thermal
                if self.channel_first
                else thermal.permute(0, 3, 1, 2).contiguous()
            )

            if stage_index in self.out_indices:
                output_norm = getattr(self, f"outnorm{stage_index}")
                rgb_for_fusion = output_norm(rgb_for_fusion)
                thermal_for_fusion = output_norm(thermal_for_fusion)
                fused_features.append(
                    self.fusion_modules[stage_index](
                        rgb_for_fusion, thermal_for_fusion
                    )
                )

            rgb = self.layers[stage_index].downsample(rgb)
            thermal = self.layers[stage_index].downsample(thermal)

        return fused_features


class DualTAVMambaEncoder(nn.Module):
    """Paper-facing encoder with a frozen shared VMamba-Tiny backbone."""

    def __init__(
        self,
        load_pretrained_backbone: bool = True,
        apply_pst_patchmerge_fix: bool = False,
    ):
        super().__init__()
        pretrained = VMAMBA_TINY_PRETRAINED if load_pretrained_backbone else None
        self.backbone = DualTAVMambaBackbone(
            pretrained=pretrained,
            apply_pst_patchmerge_fix=apply_pst_patchmerge_fix,
        )
        self._freeze_parameters()

    def _freeze_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        trainable_modules = (
            self.backbone.shared_adapters,
            self.backbone.rgb_adapters,
            self.backbone.thermal_adapters,
            self.backbone.fusion_modules,
        )
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

        if self.backbone.pst_patchmerge_applied:
            target_stage = self.backbone.num_layers - 2
            for parameter in self.backbone.layers[
                target_stage
            ].downsample.parameters():
                parameter.requires_grad = True

    def forward(
        self, rgb: torch.Tensor, thermal: torch.Tensor
    ) -> list[torch.Tensor]:
        return self.backbone(rgb, thermal)
