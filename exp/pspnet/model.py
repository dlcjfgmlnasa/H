# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict as ODict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f


def _same_padding_1d(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 7) -> None:
        super().__init__()
        padding = _same_padding_1d(k)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=padding, bias=False)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
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
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
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
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.layer2 = self._make_layer(stem_channels, stage_channels[0], stage_blocks[0], stride=stage_strides[0], k=block_kernel)
        self.layer3 = self._make_layer(stage_channels[0], stage_channels[1], stage_blocks[1], stride=stage_strides[1], k=block_kernel)
        self.layer4 = self._make_layer(stage_channels[1], stage_channels[2], stage_blocks[2], stride=stage_strides[2], k=block_kernel)
        self.layer5 = self._make_layer(stage_channels[2], stage_channels[3], stage_blocks[3], stride=stage_strides[3], k=block_kernel)
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
        x = self.stem(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return ODict([("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)])


class PyramidPooling1D(nn.Module):
    def __init__(self, in_channels: int, pool_sizes: Tuple[int, ...]):
        super().__init__()
        self.in_channels = in_channels
        self.pool_sizes = pool_sizes
        reduction_channels = in_channels // len(pool_sizes)
        self.layers = nn.ModuleList()
        for size in pool_sizes:
            self.layers.append(nn.Sequential(
                nn.AdaptiveAvgPool1d(output_size=size),
                nn.Conv1d(in_channels, reduction_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(reduction_channels),
                nn.SiLU(inplace=True)
            ))
        self.out_channels = in_channels + (len(pool_sizes) * reduction_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.size(-1)
        features = [x]
        for layer in self.layers:
            pooled = layer(x)
            upsampled = f.interpolate(pooled, size=target_size, mode='linear', align_corners=False)
            features.append(upsampled)
        return torch.cat(features, dim=1)


class PSPNet1D(nn.Module):
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
        pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.backbone = Backbone1D(
            in_channels=in_channels, stem_channels=stem_channels,
            stage_channels=stage_channels, stage_blocks=stage_blocks,
            stage_strides=stage_strides, stem_kernel=stem_kernel,
            block_kernel=block_kernel,
        )
        ppm_in_channels = self.backbone.out_ch["C5"]
        self.ppm = PyramidPooling1D(in_channels=ppm_in_channels, pool_sizes=pool_sizes)
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.ppm.out_channels, ppm_in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ppm_in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(ppm_in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.size(-1)
        features = self.backbone(x)
        c5 = features["C5"]
        ppm_out = self.ppm(c5)
        y = self.final_conv(ppm_out)
        y = f.interpolate(y, size=input_size, mode='linear', align_corners=False)
        return y


# --- 테스트 ---
if __name__ == "__main__":
    torch.manual_seed(0)
    BATCH, T_LEN, IN_CH = 4, 5000, 2
    psp_model = PSPNet1D(in_channels=IN_CH, out_channels=6)

    output_train = psp_model(torch.randn(BATCH, IN_CH, T_LEN))
    print(f"PSPNet Train mode output shape: {output_train.shape}")
