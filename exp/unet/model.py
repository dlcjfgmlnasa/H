# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict as ODict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # F를 import 합니다.


def _same_padding_1d(kernel_size: int, dilation: int = 1) -> int:
    """Calculates padding for 'same' convolution in 1D."""
    return dilation * (kernel_size - 1) // 2


class BasicBlock1D(nn.Module):
    """A basic residual block with two 1D convolutions."""

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
    """A simple block of two 1D convolutions, batch norm, and SiLU activation."""

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
    """The ResNet-like encoder part of the network."""

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
            "C1": stem_channels, "C2": stage_channels[0], "C3": stage_channels[1],
            "C4": stage_channels[2], "C5": stage_channels[3],
        }

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int, k: int) -> nn.Sequential:
        blocks: List[nn.Module] = [BasicBlock1D(in_ch, out_ch, stride=stride, k=k)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock1D(out_ch, out_ch, stride=1, k=k))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> ODict[str, torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return ODict([("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)])


class UpBlock1D(nn.Module):
    """
    An up-sampling block that uses interpolation, concatenation with a skip
    connection, and a convolutional block.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 7) -> None:
        super().__init__()
        # The convolutional block will take the concatenated channels as input
        self.conv_block = ConvBlock1D(in_ch + skip_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample x to match the spatial dimension of the skip connection
        x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)
        # Pass through the convolutional block
        return self.conv_block(x)


class UNet1D(nn.Module):
    """
    A U-Net architecture for 1D sequence-to-sequence tasks that uses
    interpolation for up-sampling in the decoder.
    """

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
        # -- Encoder (Backbone) --
        self.backbone = Backbone1D(
            in_channels=in_channels, stem_channels=stem_channels,
            stage_channels=stage_channels, stage_blocks=stage_blocks,
            stage_strides=stage_strides, stem_kernel=stem_kernel,
            block_kernel=block_kernel,
        )

        # -- Decoder --
        # Replaced ConvTranspose1d and separate ConvBlock1D with a single UpBlock1D
        self.dec4 = UpBlock1D(stage_channels[3], stage_channels[2], stage_channels[2], kernel_size=block_kernel)
        self.dec3 = UpBlock1D(stage_channels[2], stage_channels[1], stage_channels[1], kernel_size=block_kernel)
        self.dec2 = UpBlock1D(stage_channels[1], stage_channels[0], stage_channels[0], kernel_size=block_kernel)
        self.dec1 = UpBlock1D(stage_channels[0], stem_channels, stem_channels, kernel_size=block_kernel)

        # Final up-sampling and convolution to restore original resolution
        # This part doesn't have a skip connection from the input
        self.dec0_conv = ConvBlock1D(stem_channels, stem_channels // 2, kernel_size=block_kernel)
        self.final_conv = nn.Conv1d(stem_channels // 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-1]

        # -- Encoder Path --
        features = self.backbone(x)
        c1, c2, c3, c4, c5 = features["C1"], features["C2"], features["C3"], features["C4"], features["C5"]

        # -- Decoder Path with Skip Connections --
        # The logic is now much cleaner, encapsulated in UpBlock1D
        d4 = self.dec4(c5, c4)
        d3 = self.dec3(d4, c3)
        d2 = self.dec2(d3, c2)
        d1 = self.dec1(d2, c1)

        # Stage 0 (to restore original resolution)
        u0 = F.interpolate(d1, size=input_size, mode='linear', align_corners=False)
        d0 = self.dec0_conv(u0)

        # Final output layer
        out = self.final_conv(d0)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    BATCH, T_LEN, IN_CH = 4, 125 * 5, 2

    # Instantiate the new U-Net model
    model = UNet1D(in_channels=IN_CH, out_channels=6)

    # Check the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {num_params:,}")

    # Test with a random tensor
    input_tensor = torch.randn(BATCH, IN_CH, T_LEN)
    output_train = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Train mode output shape: {output_train.shape}")

    # Verify that the output length matches the input length
    assert input_tensor.shape[-1] == output_train.shape[-1]
    print("✅ Output length matches input length.")