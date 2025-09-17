# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict as ODict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            upsampled = F.interpolate(pooled, size=target_size, mode='linear', align_corners=False)
            features.append(upsampled)
        return torch.cat(features, dim=1)


class SEBlock1D(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = in_channels // reduction_ratio
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.se(x)
        return x * attention_weights


class DecoderBlock1D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attention: bool = True, kernel_size: int = 5):
        super().__init__()
        combined_ch = in_ch + skip_ch
        self.convs = ConvBlock1D(combined_ch, out_ch, kernel_size=kernel_size)
        self.attention = SEBlock1D(out_ch) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip_feature.size(-1), mode='linear', align_corners=False)
        x = torch.cat([x, skip_feature], dim=1)
        x = self.convs(x)
        x = self.attention(x)
        return x


class MANet1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stem_channels: int = 32,
            stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
            stage_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
            stage_strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
            decoder_channels: Tuple[int, int, int] = (256, 128, 64),
            pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
            stem_kernel: int = 15,
            block_kernel: int = 7,
            decoder_kernel: int = 5,
            bottleneck_kernel: int = 3,
    ):
        super().__init__()
        # --- Encoder ---
        self.backbone = Backbone1D(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            stage_strides=stage_strides,
            stem_kernel=stem_kernel,
            block_kernel=block_kernel,
        )

        # --- Bottleneck ---
        ppm_in_channels = self.backbone.out_ch["C5"]
        self.ppm = PyramidPooling1D(in_channels=ppm_in_channels, pool_sizes=pool_sizes)
        self.bottleneck_conv = ConvBlock1D(self.ppm.out_channels, ppm_in_channels, kernel_size=bottleneck_kernel)
        self.bottleneck_attention = SEBlock1D(ppm_in_channels)

        # --- Decoder ---
        self.decoder4 = DecoderBlock1D(ppm_in_channels, self.backbone.out_ch["C4"], decoder_channels[0], kernel_size=decoder_kernel)
        self.decoder3 = DecoderBlock1D(decoder_channels[0], self.backbone.out_ch["C3"], decoder_channels[1], kernel_size=decoder_kernel)
        self.decoder2 = DecoderBlock1D(decoder_channels[1], self.backbone.out_ch["C2"], decoder_channels[2], kernel_size=decoder_kernel)

        # --- Final Classifier ---
        self.final_conv = nn.Conv1d(decoder_channels[2], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.size(-1)
        features = self.backbone(x)
        c2, c3, c4, c5 = features["C2"], features["C3"], features["C4"], features["C5"]
        b = self.ppm(c5)
        b = self.bottleneck_conv(b)
        b = self.bottleneck_attention(b)
        d4 = self.decoder4(b, c4)
        d3 = self.decoder3(d4, c3)
        d2 = self.decoder2(d3, c2)
        y = self.final_conv(d2)
        y = F.interpolate(y, size=input_size, mode='linear', align_corners=False)
        return y


# --- 테스트 ---
if __name__ == "__main__":
    torch.manual_seed(0)
    BATCH, T_LEN, IN_CH, NUM_CLASSES = 4, 5000, 8, 5

    print("--- MA-Net Test (Kernel Size Customization) ---")
    ma_model = MANet1D(
        in_channels=IN_CH,
        out_channels=NUM_CLASSES,
        stem_channels=32,
        stage_channels=(64, 128, 256, 256),
        decoder_channels=(128, 64, 32),
        stem_kernel=50,
        block_kernel=25,
        decoder_kernel=10,
        bottleneck_kernel=5
    )

    output_train = ma_model(torch.randn(BATCH, IN_CH, T_LEN))
    print(f"MA-Net Train mode output shape: {output_train.shape}")

    num_params = sum(p.numel() for p in ma_model.parameters() if p.requires_grad)
    print(f"MA-Net total parameters: {num_params / 1e6:.2f}M")