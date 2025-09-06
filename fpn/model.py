# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict as ODict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f


def _choose_gn_groups(c: int) -> int:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def _same_padding_1d(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2



class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 7) -> None:
        super().__init__()
        padding = _same_padding_1d(k)

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=False)
        self.norm1 = nn.GroupNorm(_choose_gn_groups(out_ch), out_ch)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=padding, bias=False)
        self.norm2 = nn.GroupNorm(_choose_gn_groups(out_ch), out_ch)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(_choose_gn_groups(out_ch), out_ch),
            )
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        return self.act_out(out)


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 9) -> None:
        super().__init__()
        pad = _same_padding_1d(kernel_size)
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.GroupNorm(_choose_gn_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False),
            nn.GroupNorm(_choose_gn_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)



class Backbone1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            stem_channels: int = 64,
            stage_channels: Tuple[int, int, int, int] = (128, 256, 512, 512),
            stage_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
            stage_strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
            stem_kernel: int = 15,
            block_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, stem_channels, kernel_size=stem_kernel,
                stride=2, padding=_same_padding_1d(stem_kernel), bias=False
            ),
            nn.GroupNorm(_choose_gn_groups(stem_channels), stem_channels),
            nn.SiLU(inplace=True),
        )
        self.layer2 = self._make_layer(stem_channels, stage_channels[0], stage_blocks[0], stride=stage_strides[0],
                                       k=block_kernel)
        self.layer3 = self._make_layer(stage_channels[0], stage_channels[1], stage_blocks[1], stride=stage_strides[1],
                                       k=block_kernel)
        self.layer4 = self._make_layer(stage_channels[1], stage_channels[2], stage_blocks[2], stride=stage_strides[2],
                                       k=block_kernel)
        self.layer5 = self._make_layer(stage_channels[2], stage_channels[3], stage_blocks[3], stride=stage_strides[3],
                                       k=block_kernel)

        self.out_ch = {
            "C2": stage_channels[0], "C3": stage_channels[1],
            "C4": stage_channels[2], "C5": stage_channels[3],
        }

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int, k: int) -> nn.Sequential:
        blocks: List[nn.Module] = [BasicBlock1D(in_ch, out_ch, stride=stride, k=k)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock1D(out_ch, out_ch, stride=1, k=k))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> ODict[str, torch.Tensor]:
        x = self.stem(x)        # T / 2
        c2 = self.layer2(x)     # T / 4
        c3 = self.layer3(c2)    # T / 8
        c4 = self.layer4(c3)    # T / 16
        c5 = self.layer5(c4)    # T / 32
        return ODict([("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)])


class FCN1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,

            stem_channels: int = 64,
            stage_channels: Tuple[int, int, int, int] = (128, 256, 512, 512),
            stage_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
            stage_strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
            stem_kernel: int = 15,
            block_kernel: int = 7,
    ):
        super().__init__()
        self.backbone = Backbone1D(
            in_channels=in_channels, stem_channels=stem_channels,
            stage_channels=stage_channels, stage_blocks=stage_blocks,
            stage_strides=stage_strides, stem_kernel=stem_kernel,
            block_kernel=block_kernel,
        )

        w2 = self.backbone.out_ch["C2"]
        w3 = self.backbone.out_ch["C3"]
        w4 = self.backbone.out_ch["C4"]
        w5 = self.backbone.out_ch["C5"]

        # Bottleneck
        self.b = ConvBlock1D(w5, w5, kernel_size=3)

        # Decoder
        self.up4 = nn.Conv1d(w5, w4, 1, bias=False)
        self.d4 = ConvBlock1D(w4, w4, kernel_size=3)

        self.up3 = nn.Conv1d(w4, w3, 1, bias=False)
        self.d3 = ConvBlock1D(w3, w3, kernel_size=3)

        self.up2 = nn.Conv1d(w3, w2, 1, bias=False)
        self.d2 = ConvBlock1D(w2, w2, kernel_size=3)

        self.out_head = nn.Conv1d(w2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        c2, c3, c4, c5 = features["C2"], features["C3"], features["C4"], features["C5"]

        z = self.b(c5)

        u4 = f.interpolate(z, size=c4.size(-1), mode="linear", align_corners=False)
        u4 = self.up4(u4) + c4
        u4 = self.d4(u4)

        u3 = f.interpolate(u4, size=c3.size(-1), mode="linear", align_corners=False)
        u3 = self.up3(u3) + c3
        u3 = self.d3(u3)

        u2 = f.interpolate(u3, size=c2.size(-1), mode="linear", align_corners=False)
        u2 = self.up2(u2) + c2
        u2 = self.d2(u2)

        y = self.out_head(u2)
        y = f.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)
        return y



if __name__ == "__main__":
    torch.manual_seed(0)
    BATCH, T_LEN, IN_CH = 4, 5000, 2

    T_LEN_LIGHT = 125 * 5  # 625

    model_light = FCN1D(
        in_channels=IN_CH,
        out_channels=6,
        stem_channels=32,
        stage_channels=(32, 64, 128, 256),
        stage_blocks=(2, 2, 2, 1),
        stem_kernel=11,
        block_kernel=5,
    )