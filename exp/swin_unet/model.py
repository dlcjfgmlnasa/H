# -*- coding: utf-8 -*-
"""
1D Swin-Unet (PyTorch) with configurable patch_size & window_size.

This script implements a 1D version of the Swin-Unet architecture using PyTorch
for sequence segmentation tasks.
"""
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple_length(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    """Right-pad the last dimension of a tensor to a multiple of a given length."""
    input_length = x.size(-1)
    pad = (multiple - (input_length % multiple)) % multiple
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


def right_unpad_length(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Remove right padding from the last dimension of a tensor."""
    return x[..., :-pad] if pad else x


def window_partition_1d(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int]:
    """Partition a 1D sequence into non-overlapping windows."""
    batch_size, num_tokens, channels = x.shape
    pad_len = (window_size - (num_tokens % window_size)) % window_size
    if pad_len:
        # Pad tokens by repeating the last one
        pad_token = x[:, -1:, :].expand(batch_size, pad_len, channels)
        x = torch.cat([x, pad_token], dim=1)
        num_tokens += pad_len
    x = x.view(batch_size, num_tokens // window_size, window_size, channels)
    x = x.reshape(batch_size * (num_tokens // window_size), window_size, channels)
    return x, pad_len


def window_reverse_1d(
    windows: torch.Tensor, batch_size: int, num_tokens: int, window_size: int, pad_len: int
) -> torch.Tensor:
    """Reverse the window partitioning operation."""
    num_windows = (num_tokens + pad_len) // window_size
    x = windows.view(batch_size, num_windows, window_size, -1).reshape(
        batch_size, num_windows * window_size, -1
    )
    if pad_len:
        x = x[:, :-pad_len, :]
    return x


class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        x, pad = pad_to_multiple_length(x, self.patch_size)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x, pad


class WindowAttention1D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped_batch_size, window_len, channels = x.shape
        qkv = self.qkv(x).reshape(
            reshaped_batch_size, window_len, 3, self.num_heads, channels // self.num_heads
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(reshaped_batch_size, window_len, channels)
        return self.proj(out)


class SwinBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        window_size: int = 8,
        shift_size: int = 0,
    ):
        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)

        windows, pad_len = window_partition_1d(x, self.window_size)
        windows = self.attn(windows)
        x = window_reverse_1d(
            windows,
            batch_size=batch_size,
            num_tokens=num_tokens,
            window_size=self.window_size,
            pad_len=pad_len,
        )

        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        batch_size, num_tokens, channels = x.shape
        pad_len = num_tokens % 2
        if pad_len:
            x = torch.cat([x, x[:, -1:, :]], dim=1)

        x = x.reshape(batch_size, (num_tokens + pad_len) // 2, 2 * channels)
        x = self.norm(x)
        x = self.reduction(x)
        return x, pad_len


class SwinUnet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 64,
        depth: List[int] = [2, 2, 2, 2],
        num_heads: List[int] = [4, 4, 8, 8],
        patch_size: int = 4,
        window_size: int = 8,
        use_shift: bool = True,
    ):
        super().__init__()
        assert len(depth) == len(num_heads), "depth and num_heads must align per stage"
        self.patch_size = patch_size
        self.window_size = window_size
        self.shift_size = (window_size // 2) if use_shift and window_size > 1 else 0

        self.patch_embed = PatchEmbedding1D(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size
        )

        dims = [embed_dim * (2 ** i) for i in range(len(depth))]
        self.encoders = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        for i, d in enumerate(depth):
            blocks = []
            for b in range(d):
                shift = self.shift_size if (b % 2 == 1) else 0
                blocks.append(
                    SwinBlock1D(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=shift,
                    )
                )
            self.encoders.append(nn.Sequential(*blocks))
            if i < len(depth) - 1:
                self.merge_layers.append(PatchMerging1D(dim=dims[i]))

        self.bottleneck = nn.Sequential(
            SwinBlock1D(
                dim=dims[-1],
                num_heads=num_heads[-1],
                window_size=window_size,
                shift_size=0,
            ),
            SwinBlock1D(
                dim=dims[-1],
                num_heads=num_heads[-1],
                window_size=window_size,
                shift_size=self.shift_size,
            ),
        )

        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(len(depth) - 1)):
            self.upsamples.append(
                nn.ConvTranspose1d(dims[i + 1], dims[i], kernel_size=2, stride=2)
            )
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Linear(2 * dims[i], dims[i]),
                    SwinBlock1D(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=0,
                    ),
                    SwinBlock1D(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=self.shift_size,
                    ),
                )
            )

        self.final_proj = nn.Linear(2 * dims[0], dims[0])
        self.head = nn.ConvTranspose1d(
            dims[0], num_classes, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, time_pad = self.patch_embed(x)

        skips = []
        merge_pads = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            skips.append(x)
            if i < len(self.encoders) - 1:
                x, pad_token = self.merge_layers[i](x)
                merge_pads.append(pad_token)

        x = self.bottleneck(x)

        merge_pads.reverse()
        for i, (up, dec) in enumerate(zip(self.upsamples, self.dec_blocks)):
            x = x.transpose(1, 2)
            x = up(x)
            x = x.transpose(1, 2)

            pad_to_remove = merge_pads[i]
            if pad_to_remove > 0:
                x = x[:, :-pad_to_remove, :]

            skip_idx = len(skips) - 2 - i
            skip = skips[skip_idx]

            x = torch.cat([x, skip], dim=-1)
            x = dec(x)

        skip0 = skips[0]
        x = torch.cat([x, skip0], dim=-1)
        x = self.final_proj(x)

        x = x.transpose(1, 2)
        logits = self.head(x)

        logits = right_unpad_length(logits, time_pad)
        return logits


if __name__ == "__main__":
    batch_size, in_channels, time_length = 2, 1, 10000
    model = SwinUnet1D(
        in_channels=in_channels,
        num_classes=3,
        embed_dim=128,
        depth=[2, 2, 2, 2],
        num_heads=[4, 4, 8, 8],
        patch_size=2,
        window_size=64,
        use_shift=True,
    )
    x = torch.randn(batch_size, in_channels, time_length)
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape)
    assert y.shape == (batch_size, 3, time_length)
    print("Test passed!")