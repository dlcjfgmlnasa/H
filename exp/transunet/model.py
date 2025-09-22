# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, linear: bool = True):
        super().__init__()
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv1D(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv1D(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CL (channel, length)
        diff = x2.size()[-1] - x1.size()[-1]
        x1 = nn.functional.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransUNet1D(nn.Module):
    def __init__(
            self,
            *,
            length: int,
            segment_size: int,
            num_classes: int,
            dim: int,
            depth: int,
            heads: int,
            mlp_dim: int,
            in_channels: int = 1,
            cnn_channels: tuple[int, ...] = (64, 128, 256, 512),
            dim_head: int = 64,
            dropout: float = 0.,
            emb_dropout: float = 0.
    ):
        super().__init__()
        assert length % segment_size == 0, 'Signal length must be divisible by the segment size.'

        # 1D Segment Embedding
        self.segment_embedding = nn.Conv1d(in_channels, dim, kernel_size=segment_size, stride=segment_size)

        num_segments = length // segment_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_segments, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        # ======== 1. 1D CNN Encoder ========
        self.inc = DoubleConv1D(in_channels, cnn_channels[0])
        self.downs = nn.ModuleList()
        for i in range(len(cnn_channels) - 1):
            self.downs.append(Down1D(cnn_channels[i], cnn_channels[i + 1]))

        # ======== 2. 1D Decoder ========
        self.ups = nn.ModuleList()
        up_in_channels = [dim] + list(reversed(cnn_channels))

        for i in range(len(cnn_channels)):
            in_ch = up_in_channels[i]
            out_ch = up_in_channels[i + 1]
            self.ups.append(Up1D(in_ch, out_ch, linear=False))

        self.outc = OutConv1D(cnn_channels[0], num_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        x = self.inc(data)
        skip_connections.append(x)

        for down_layer in self.downs:
            x = down_layer(x)
            skip_connections.append(x)

        # ======== Transformer Encoder Path ========
        embedded = self.segment_embedding(data)
        embedded = embedded.transpose(1, 2)

        b, n, _ = embedded.shape
        embedded += self.pos_embedding[:, :n]
        embedded = self.dropout(embedded)

        # Transformer Output
        transformer_out = self.transformer(embedded)

        # ======== 1D Decoder Path ========
        x = transformer_out.transpose(1, 2)
        num_segments = data.shape[-1] // self.segment_embedding.kernel_size[0]
        x = x.view(b, -1, num_segments)

        for up_layer in self.ups:
            skip = skip_connections.pop()
            x = up_layer(x, skip)

        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    sig_length = 5000
    seg_size = 50
    n_classes = 5
    input_channels = 2

    model = TransUNet1D(
        length=sig_length,
        segment_size=seg_size,
        num_classes=n_classes,
        in_channels=input_channels,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
    ).cuda()  # GPU 사용

    dummy_signal = torch.randn(4, input_channels, sig_length).cuda()
    output = model(dummy_signal)
