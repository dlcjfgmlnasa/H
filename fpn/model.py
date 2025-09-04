# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict as ODict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def pad_to_multiple_1d(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    t_len = x.size(-1)
    pad = (multiple - (t_len % multiple)) % multiple
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


def right_unpad_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    return x[..., :-pad] if pad else x


# -----------------------------------------------------------------------------
# Basic Blocks
# -----------------------------------------------------------------------------
class DSConv1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        g: Optional[int] = None,
    ) -> None:
        super().__init__()
        groups = in_ch if g is None else g
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


# -----------------------------------------------------------------------------
# Backbone: C2 ~ C5
# -----------------------------------------------------------------------------
class Backbone1D(nn.Module):
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
        x = self.stem(x)      # T/2
        c2 = self.layer2(x)   # T/4
        c3 = self.layer3(c2)  # T/8
        c4 = self.layer4(c3)  # T/16
        c5 = self.layer5(c4)  # T/32
        return ODict([("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)])


# -----------------------------------------------------------------------------
# FPN
# -----------------------------------------------------------------------------
class FPN1D(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        use_smooth: bool = True,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv1d(c, out_channels, 1) for c in in_channels_list]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Conv1d(out_channels, out_channels, 3, padding=1)
                if use_smooth
                else nn.Identity()
                for _ in in_channels_list
            ]
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 2, -1, -1):
            up = F.interpolate(
                lat[i + 1], size=lat[i].shape[-1], mode="linear",
                align_corners=False,
            )
            lat[i] = lat[i] + up
        outs = [s(x) for s, x in zip(self.smooth, lat)]
        return outs


# -----------------------------------------------------------------------------
# Multi-modal Wrapper
# -----------------------------------------------------------------------------
class MultiModalFPN1D(nn.Module):
    """
    ...
    use_c2_in_fpn: If True, build P2~P5; else P3~P5.
    fusion: "sum" | "concat" | "attn" modality fusion.
    """
    def __init__(
        self,
        modalities: Dict[str, int],
        backbone_cfg: Optional[Dict] = None,
        fpn_out_channels: int = 256,
        use_c2_in_fpn: bool = False,
        fusion: str = "sum",
    ) -> None:
        super().__init__()
        self.modalities = list(modalities.keys())
        self.use_c2 = use_c2_in_fpn
        self.fusion = fusion

        backbone_cfg = backbone_cfg or {}
        self.backbones = nn.ModuleDict(
            {
                m: Backbone1D(in_channels=cin, **backbone_cfg)
                for m, cin in modalities.items()
            }
        )

        any_bb = next(iter(self.backbones.values()))
        if self.use_c2:
            self.fpn_in_list = [
                any_bb.out_ch["C2"],
                any_bb.out_ch["C3"],
                any_bb.out_ch["C4"],
                any_bb.out_ch["C5"],
            ]
            self.p_names = ["P2", "P3", "P4", "P5"]
        else:
            self.fpn_in_list = [
                any_bb.out_ch["C3"],
                any_bb.out_ch["C4"],
                any_bb.out_ch["C5"],
            ]
            self.p_names = ["P3", "P4", "P5"]

        # Modality-specific 1x1 to unify dims
        self.mod_laterals = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    [nn.Conv1d(cin, fpn_out_channels, 1) for cin in self.fpn_in_list]
                )
                for m in self.modalities
            }
        )

        # Concat fusion -> reducer per level
        if fusion == "concat":
            self.reducers = nn.ModuleList(
                [
                    nn.Conv1d(
                        fpn_out_channels * len(self.modalities),
                        fpn_out_channels,
                        1,
                    )
                    for _ in self.fpn_in_list
                ]
            )
        else:
            self.reducers = None

        # Attention fusion -> level-wise modality weights
        if fusion == "attn":
            self.attn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),  # (B, F, 1)
                        nn.Flatten(1),            # (B, F)
                        nn.Linear(fpn_out_channels, fpn_out_channels // 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(
                            fpn_out_channels // 4, len(self.modalities)
                        ),
                    )
                    for _ in self.fpn_in_list
                ]
            )
        else:
            self.attn = None

        # FPN on fused C-levels
        self.fpn = FPN1D(
            [fpn_out_channels] * len(self.fpn_in_list),
            out_channels=fpn_out_channels,
        )

    def _select_levels(
        self, cdict: ODict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        if self.use_c2:
            keys = ["C2", "C3", "C4", "C5"]
        else:
            keys = ["C3", "C4", "C5"]
        return [cdict[k] for k in keys]

    def forward(self, xdict: Dict[str, torch.Tensor]) -> ODict[str, torch.Tensor]:
        assert isinstance(xdict, dict) and len(xdict) > 0

        # Align batch/length and pad to multiple of 32
        batch = None
        t_max = 0
        for x in xdict.values():
            assert x.dim() == 3, "Each modality must be a (B, C, T) tensor."
            batch = x.size(0) if batch is None else batch
            assert x.size(0) == batch, "All modalities must share batch size."
            t_max = max(t_max, x.size(-1))

        padded: Dict[str, torch.Tensor] = {}
        pad_info: Dict[str, int] = {}
        for m, x in xdict.items():
            if x.size(-1) < t_max:
                x = F.pad(x, (0, t_max - x.size(-1)))
            x, pad = pad_to_multiple_1d(x, multiple=32)
            padded[m] = x
            pad_info[m] = pad  # For potential restoration in heads

        # Per-modality backbones
        c_feats: Dict[str, ODict[str, torch.Tensor]] = {
            m: self.backbones[m](padded[m]) for m in self.modalities
        }

        # Project and prepare per-level lists
        n_levels = len(self.fpn_in_list)
        per_level_proj: List[List[torch.Tensor]] = [[] for _ in range(n_levels)]
        for m in self.modalities:
            levels = self._select_levels(c_feats[m])
            adapters = self.mod_laterals[m]
            for i, (feat, adapt) in enumerate(zip(levels, adapters)):
                per_level_proj[i].append(adapt(feat))  # (B, F, T_i)

        # Fuse across modalities
        fused: List[torch.Tensor] = []
        for li, feats in enumerate(per_level_proj):
            if self.fusion == "sum":
                out = torch.stack(feats, dim=0).sum(dim=0)
            elif self.fusion == "concat":
                cat = torch.cat(feats, dim=1)          # (B, M*F, T)
                out = self.reducers[li](cat)           # (B, F, T)
            elif self.fusion == "attn":
                # Simple modality attention via pooled query
                stack = torch.stack(feats, dim=1)      # (B, M, F, T)
                query = feats[0]                       # (B, F, T)
                qv = self.attn[li][0](query)           # GAP -> (B, F, 1)
                qv = self.attn[li][1](qv)              # (B, F)
                qv = self.attn[li][2](qv)              # FC
                qv = self.attn[li][3](qv)
                weights = self.attn[li][4](qv)         # (B, M)
                weights = F.softmax(weights, dim=-1)   # (B, M)
                weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1)
                out = (stack * weights).sum(dim=1)     # (B, F, T)
            else:
                raise ValueError("fusion must be 'sum', 'concat', or 'attn'.")
            fused.append(out)

        # Top-down FPN
        p_list = self.fpn(fused)
        out = ODict((name, feat) for name, feat in zip(self.p_names, p_list))
        return out


# -----------------------------------------------------------------------------
# Optional Segmentation Head
# -----------------------------------------------------------------------------
class PyramidSegHead1D(nn.Module):
    """Simple head merging P-levels then upsampling to original length."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        up_to_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.out = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.up_to_length = up_to_length

    def forward(
        self,
        p_feats: ODict[str, torch.Tensor],
        original_len: int,
    ) -> torch.Tensor:
        keys = sorted(p_feats.keys())  # e.g., ["P3", "P4", "P5"]
        base = p_feats[keys[0]]
        merged = base
        for k in keys[1:]:
            merged = merged + F.interpolate(
                p_feats[k],
                size=base.size(-1),
                mode="linear",
                align_corners=False,
            )
        y = self.out(merged)
        y = F.interpolate(y, size=original_len, mode="linear", align_corners=False)
        return y


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    BATCH, T_LEN = 4, 5000
    inputs = {
        "ecg": torch.randn(BATCH, 2, T_LEN),
        # "ppg": torch.randn(BATCH, 1, T_LEN),
        # "imu": torch.randn(BATCH, 6, T_LEN),
    }

    model = MultiModalFPN1D(
        # modalities={"ecg": 1, "ppg": 1, "imu": 6},
        modalities={'ecg': 2},
        backbone_cfg=dict(
            stem_channels=64,
            stage_channels=(128, 256, 512, 512),
            stage_blocks=(2, 2, 2, 2),
        ),
        fpn_out_channels=192,
        use_c2_in_fpn=False,   # Build P3~P5
        fusion="attn",         # "sum" | "concat" | "attn"
    )

    pyramids = model(inputs)
    for name, tensor in pyramids.items():
        print(name, tensor.shape)

    head = PyramidSegHead1D(in_ch=192, out_ch=1)
    mask = head(pyramids, original_len=T_LEN)
    print("mask:", mask.shape)
