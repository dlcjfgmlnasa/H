# -*- coding: utf-8 -*-
"""Multi-modal 1D PSPNet (Pyramid Scene Parsing Network) for signals."""

from __future__ import annotations

from collections import OrderedDict as ODict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def pad_to_multiple_1d(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    """Right-pad (B, C, T) so that T % multiple == 0."""
    t_len = x.size(-1)
    pad = (multiple - (t_len % multiple)) % multiple
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


def right_unpad_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Remove right-side padding that was added by pad_to_multiple_1d."""
    return x[..., :-pad] if pad else x


# -----------------------------------------------------------------------------
# Lightweight Residual Backbone for 1D
# -----------------------------------------------------------------------------
class DSConv1d(nn.Module):
    """Depthwise-Separable 1D Conv."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        groups = in_ch if groups is None else groups
        pad = (k - 1) // 2
        self.dw = nn.Conv1d(
            in_ch, in_ch, k, stride=s, padding=pad, groups=groups, bias=False
        )
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class ResidualBlock1D(nn.Module):
    """Residual block using depthwise-separable convs."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = DSConv1d(in_ch, out_ch, k=3, s=stride)
        self.conv2 = DSConv1d(out_ch, out_ch, k=3, s=1)
        self.down: Optional[nn.Module]
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x + identity, inplace=True)


class Backbone1D(nn.Module):
    """1D backbone producing C2..C5.

    Input (B, C, T) -> stem( /2 ) -> C2(/4), C3(/8), C4(/16), C5(/32)
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 64,
        stage_channels: Tuple[int, int, int, int] = (128, 256, 512, 512),
        stage_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, stem_channels, kernel_size=7, stride=2, padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.layer2 = self._make_layer(
            stem_channels, stage_channels[0], stage_blocks[0], stride=2
        )
        self.layer3 = self._make_layer(
            stage_channels[0], stage_channels[1], stage_blocks[1], stride=2
        )
        self.layer4 = self._make_layer(
            stage_channels[1], stage_channels[2], stage_blocks[2], stride=2
        )
        self.layer5 = self._make_layer(
            stage_channels[2], stage_channels[3], stage_blocks[3], stride=2
        )

        self.out_ch = {
            "C2": stage_channels[0],
            "C3": stage_channels[1],
            "C4": stage_channels[2],
            "C5": stage_channels[3],
        }

    @staticmethod
    def _make_layer(
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks: List[nn.Module] = [ResidualBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(ResidualBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> ODict[str, torch.Tensor]:
        x = self.stem(x)      # /2
        c2 = self.layer2(x)   # /4
        c3 = self.layer3(c2)  # /8
        c4 = self.layer4(c3)  # /16
        c5 = self.layer5(c4)  # /32
        return ODict([("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)])


# -----------------------------------------------------------------------------
# Pyramid Pooling Module (1D)
# -----------------------------------------------------------------------------
class PPM1D(nn.Module):
    """1D Pyramid Pooling Module as in PSPNet.

    For each bin size, we AdaptiveAvgPool1d to that bin, then 1x1 conv to
    reduce channels, upsample back to the input length, and concatenate.
    """

    def __init__(
        self,
        in_ch: int,
        branch_out_ch: int = 64,
        bins: Sequence[int] = (1, 2, 3, 6),
        use_bn: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.branches = nn.ModuleList()
        for _ in bins:
            layers: List[nn.Module] = [nn.Conv1d(in_ch, branch_out_ch, 1, bias=False)]
            if use_bn:
                layers.append(nn.BatchNorm1d(branch_out_ch))
            layers.append(nn.ReLU(inplace=True))
            self.branches.append(nn.Sequential(*layers))

        bottleneck_in = in_ch + len(bins) * branch_out_ch
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_in, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(in_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_len = x.size(-1)
        pyramids = [x]
        for bin_size, branch in zip(self.bins, self.branches):
            pooled = F.adaptive_avg_pool1d(x, bin_size)     # (B, C, bin)
            reduced = branch(pooled)                         # (B, Bch, bin)
            up = F.interpolate(
                reduced, size=t_len, mode="linear", align_corners=False
            )
            pyramids.append(up)
        x = torch.cat(pyramids, dim=1)
        x = self.bottleneck(x)
        return x


# -----------------------------------------------------------------------------
# PSP Head (logits to num_classes, then upsample to original length)
# -----------------------------------------------------------------------------
class PSPHead1D(nn.Module):
    """Classifier head for PSPNet-1D."""

    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        use_bn: bool = True,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.cls = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(in_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Conv1d(in_ch, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, out_len: int) -> torch.Tensor:
        x = self.cls(x)
        x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        return x


# -----------------------------------------------------------------------------
# Multi-modal PSPNet-1D
# -----------------------------------------------------------------------------
class MultiModalPSPNet1D(nn.Module):
    """PSPNet for multi-modal 1D signals.

    Pipeline:
      - Per-modality backbone -> pick feature level (e.g., C5)
      - 1x1 project to common dim -> fuse ("sum" | "concat" | "attn")
      - PPM1D -> PSPHead1D -> upsample to original length

    Notes:
      - Input tensors may have different T; internally padded and aligned.
      - Overall stride is 32 when using C5 (stem + 4 stages).
    """

    def __init__(
        self,
        modalities: Dict[str, int],
        backbone_cfg: Optional[Dict] = None,
        feature_level: str = "C5",           # "C3" | "C4" | "C5"
        fused_channels: int = 256,
        fusion: str = "sum",                 # "sum" | "concat" | "attn"
        ppm_bins: Sequence[int] = (1, 2, 3, 6),
        ppm_branch_out: int = 64,
        num_classes: int = 1,
        head_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert feature_level in {"C3", "C4", "C5"}
        assert fusion in {"sum", "concat", "attn"}

        self.modalities = list(modalities.keys())
        self.feature_level = feature_level
        self.fusion = fusion
        self.num_classes = num_classes

        backbone_cfg = backbone_cfg or {}
        self.backbones = nn.ModuleDict(
            {
                m: Backbone1D(in_channels=cin, **backbone_cfg)
                for m, cin in modalities.items()
            }
        )

        # Determine in-channels for the chosen level (assume identical per bb)
        any_bb = next(iter(self.backbones.values()))
        level_ch = any_bb.out_ch[feature_level]

        # Per-modality 1x1 adapters to unify to fused_channels
        self.adapters = nn.ModuleDict(
            {m: nn.Conv1d(level_ch, fused_channels, 1) for m in self.modalities}
        )

        # Concat fusion -> reducer
        if self.fusion == "concat":
            self.concat_reducer = nn.Conv1d(
                fused_channels * len(self.modalities), fused_channels, 1
            )
        else:
            self.concat_reducer = None

        # Attention fusion -> small MLP to get modality weights
        if self.fusion == "attn":
            self.attn_mlp = nn.Sequential(
                nn.Linear(fused_channels, fused_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(fused_channels // 4, len(self.modalities)),
            )
        else:
            self.attn_mlp = None

        # Pyramid Pooling + Head
        self.ppm = PPM1D(
            in_ch=fused_channels,
            branch_out_ch=ppm_branch_out,
            bins=ppm_bins,
            use_bn=True,
            dropout_p=0.0,
        )
        self.head = PSPHead1D(
            in_ch=fused_channels, num_classes=num_classes,
            use_bn=True, dropout_p=head_dropout_p
        )

    # ------------------------- helpers -------------------------
    def _pick_level(self, cdict: ODict[str, torch.Tensor]) -> torch.Tensor:
        return cdict[self.feature_level]

    # ------------------------- forward -------------------------
    def forward(self, xdict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(xdict, dict) and len(xdict) > 0
        # Align batch and lengths; pad to multiple of 32 for C5
        batch = None
        t_max = 0
        for x in xdict.values():
            assert x.dim() == 3, "Each modality must be (B, C, T)."
            batch = x.size(0) if batch is None else batch
            assert x.size(0) == batch, "All modalities must share batch size."
            t_max = max(t_max, x.size(-1))

        padded: Dict[str, torch.Tensor] = {}
        pad_lens: Dict[str, int] = {}
        for m, x in xdict.items():
            if x.size(-1) < t_max:
                x = F.pad(x, (0, t_max - x.size(-1)))
            x, pad = pad_to_multiple_1d(x, multiple=32)
            padded[m] = x
            pad_lens[m] = pad
        original_len = t_max  # final upsample target

        # Per-modality backbone -> pick feature level -> 1x1 adapt
        feats: List[torch.Tensor] = []
        for m in self.modalities:
            cdict = self.backbones[m](padded[m])
            f = self._pick_level(cdict)           # (B, C_level, T')
            f = self.adapters[m](f)                # (B, Fused, T')
            feats.append(f)

        # Modality fusion
        if self.fusion == "sum":
            fused = torch.stack(feats, dim=0).sum(dim=0)  # (B, F, T')
        elif self.fusion == "concat":
            fused = torch.cat(feats, dim=1)               # (B, M*F, T')
            fused = self.concat_reducer(fused)            # (B, F, T')
        else:  # "attn"
            # Global average on channels to get a per-modality descriptor
            # (B, F, T') -> (B, F)
            descs = [f.mean(dim=-1) for f in feats]
            # Use the first modality as query (simple heuristic)
            query = descs[0]                              # (B, F)
            weights = self.attn_mlp(query)                # (B, M)
            weights = F.softmax(weights, dim=-1)          # (B, M)
            weights = weights.unsqueeze(-1).unsqueeze(-1) # (B, M, 1, 1)
            stacked = torch.stack(feats, dim=1)           # (B, M, F, T')
            fused = (stacked * weights).sum(dim=1)        # (B, F, T')

        # PPM + Head -> logits at original length
        fused = self.ppm(fused)                            # (B, F, T')
        logits = self.head(fused, out_len=original_len)    # (B, C, T)
        return logits


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, 5000
    xdict = {
        "ecg": torch.randn(B, 1, T),
        "ppg": torch.randn(B, 1, T),
        "imu": torch.randn(B, 6, T),
    }

    model = MultiModalPSPNet1D(
        modalities={"ecg": 1, "ppg": 1, "imu": 6},
        backbone_cfg=dict(
            stem_channels=64,
            stage_channels=(128, 256, 512, 512),
            stage_blocks=(2, 2, 2, 2),
        ),
        feature_level="C5",          # or "C4"/"C3"
        fused_channels=192,
        fusion="attn",               # "sum" | "concat" | "attn"
        ppm_bins=(1, 2, 3, 6),
        ppm_branch_out=64,
        num_classes=1,
        head_dropout_p=0.1,
    )

    y = model(xdict)                 # (B, 1, T)
    print("logits:", y.shape)
