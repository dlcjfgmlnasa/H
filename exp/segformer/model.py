# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Sequence, Tuple

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


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x / keep_prob * rand


# -----------------------------------------------------------------------------
# MiT (Mix Transformer Encoder) Building Blocks
# -----------------------------------------------------------------------------
class OverlapPatchEmbed1D(nn.Module):
    def __init__(
            self,
            in_ch: int,
            embed_dim: int,
            kernel_size: int,
            stride: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv1d(
            in_ch, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, E, T')
        x = self.proj(x)
        # (B, E, T') -> (B, T', E)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class EfficientAttention1D(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 시퀀스 길이를 줄이기 위한 Conv1d와 LayerNorm
            self.sr = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr, self.norm = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        q = self.q(x).reshape(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_kv = x.permute(0, 2, 1)  # (B, C, T)
            x_kv = self.sr(x_kv)  # (B, C, T//sr)
            x_kv = x_kv.transpose(1, 2)  # (B, T//sr, C)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        tk = x_kv.shape[1]
        k = self.k(x_kv).reshape(b, tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(b, tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, t, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MixFFN1D(nn.Module):
    """SegFormer's Mix-FFN."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.fc1(x)
        x = x.transpose(1, 2)  # (B, H, T)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, T, H)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock1D(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention1D(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN1D(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MiTBackbone1D(nn.Module):
    """Mix Transformer"""

    def __init__(
        self,
        in_channels: int,
        embed_dims: Sequence[int] = (64, 128, 256, 512),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur_dpr = 0

        self.stages = nn.ModuleList()
        in_ch = in_channels
        for i in range(4):
            patch_embed = OverlapPatchEmbed1D(
                in_ch=in_ch,
                embed_dim=embed_dims[i],
                kernel_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
            )

            blocks = nn.ModuleList([
                EncoderBlock1D(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur_dpr + j],
                    sr_ratio=sr_ratios[i],
                ) for j in range(depths[i])
            ])

            in_ch = embed_dims[i]
            cur_dpr += depths[i]
            self.stages.append(nn.ModuleDict({"patch_embed": patch_embed, "blocks": blocks}))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage["patch_embed"](x)

            for blk in stage["blocks"]:
                x = blk(x)

            # (B, T', E) -> (B, E, T')
            x = x.transpose(1, 2)
            features.append(x)
        return features


# -----------------------------------------------------------------------------
# SegFormer Decode Head
# -----------------------------------------------------------------------------
class SegFormerDecodeHead1D(nn.Module):
    def __init__(
            self,
            in_dims: Sequence[int],
            num_classes: int,
            decoder_dim: int = 256,
            dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv1d(c, decoder_dim, 1, bias=False) for c in in_dims
        ])

        self.fuse = nn.Sequential(
            nn.Conv1d(decoder_dim * len(in_dims), decoder_dim, 1, bias=False),
            nn.BatchNorm1d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        self.classifier = nn.Conv1d(decoder_dim, num_classes, 1)

    def forward(self, feats: List[torch.Tensor], out_len: int) -> torch.Tensor:
        ref_len = feats[0].shape[-1]

        upsampled = [
            F.interpolate(proj(f), size=ref_len, mode="linear", align_corners=False)
            for f, proj in zip(feats, self.proj)
        ]

        # Concat -> Fuse -> Classify
        x = torch.cat(upsampled, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)

        # 원래 입력 시퀀스 길이로 최종 upsample
        x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        return x


# -----------------------------------------------------------------------------
# Final SegFormer-1D Model
# -----------------------------------------------------------------------------
class SegFormer1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            embed_dims=(64, 64, 128, 128),
            depths=(2, 2, 2, 2),
            num_heads=(1, 2, 4, 8),
            sr_ratios=(8, 4, 2, 1),
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            decoder_dim=256,
    ) -> None:
        super().__init__()

        self.backbone = MiTBackbone1D(in_channels=in_channels,
                                      embed_dims=embed_dims,
                                      depths=depths,
                                      num_heads=num_heads,
                                      sr_ratios=sr_ratios,
                                      mlp_ratio=mlp_ratio,
                                      drop_rate=drop_path_rate)
        in_dims = embed_dims

        self.decode_head = SegFormerDecodeHead1D(
            in_dims=in_dims, num_classes=num_classes, decoder_dim=decoder_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        t_orig = x.shape[-1]

        x, pad = pad_to_multiple_1d(x, multiple=32)

        features = self.backbone(x)
        logits = self.decode_head(features, out_len=x.shape[-1])

        logits = right_unpad_1d(logits, pad)
        assert logits.shape[-1] == t_orig

        return logits


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    bsz, t_len, in_ch = 2, 5000, 1
    num_classes = 5

    model = SegFormer1D(
        in_channels=1,
        num_classes=num_classes,
        embed_dims=(64, 64, 128, 128),
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 4, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        decoder_dim=256,
    )

    x_ = torch.randn(bsz, in_ch, t_len)
    y = model(x_)

    print("Input shape:", x_.shape)
    print("Output logits shape:", y.shape)  # (B, num_classes, T)
