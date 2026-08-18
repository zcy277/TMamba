"""TMamba model for RGB-thermal semantic segmentation."""

import torch
import torch.nn as nn

from .decoders.MambaDecoder import MambaDecoder
from .encoders.dual_ta_vmamba import DualTAVMambaEncoder, VMAMBA_TINY_DIM


class TMamba(nn.Module):
    """Final TMamba architecture described in the paper."""

    def __init__(
        self,
        num_classes: int = 9,
        ignore_index: int = -1,
        load_pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.segmentation_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

        apply_pst_patchmerge_fix = num_classes == 5
        self.encoder = DualTAVMambaEncoder(
            load_pretrained_backbone=load_pretrained_backbone,
            apply_pst_patchmerge_fix=apply_pst_patchmerge_fix,
        )

        decoder_channels = [VMAMBA_TINY_DIM * (2**stage) for stage in range(4)]
        self.decoder = MambaDecoder(
            num_classes=num_classes,
            in_channels=decoder_channels,
            embed_dim=VMAMBA_TINY_DIM,
            deep_supervision=False,
        )
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        for module in self.decoder.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.1
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def re_freeze_backbone(self) -> None:
        """Restore the intended frozen-backbone training state after loading."""
        self.encoder._freeze_parameters()

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        label: torch.Tensor | None = None,
    ):
        fused_features = self.encoder(rgb, thermal)
        logits = self.decoder(fused_features)

        if label is None:
            return logits

        loss = self.segmentation_criterion(logits, label)
        return logits, loss
