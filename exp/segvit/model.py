# -*- coding: utf-8 -*-
"""Single-modal 1D SegViT (ViT backbone + Token Pyramid + MLP decode head)."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple_1d(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    """Right-pad (B, C, T) so that T % multiple == 0."""
    t_len = x.size(-1)
    pad = (multiple - (t_len % multiple)) % multiple
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


def right_unpad_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Remove right-side padding added by pad_to_multiple_1d."""
    return x[..., :-pad] if pad else x


def get_1d_sincos_pos_embed(embed_dim: int, t_len: int) -> torch.Tensor:
    """Standard 1D sine-cos positional embedding. Returns (1, C, T)."""
    assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos."
    position = torch.arange(t_len).float()
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(t_len, embed_dim)
    pe[:, 0::2] = torch.sin(position[:, None] * div_term[None, :])
    pe[:, 1::2] = torch.cos(position[:, None] * div_term[None, :])
    pe = pe.t().unsqueeze(0)
    return pe


class DropPath(nn.Module):
    """Stochastic depth per sample (when applied in residual branch)."""

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
# ViT(1D) Components
# -----------------------------------------------------------------------------
class PatchEmbed1D(nn.Module):
    """Overlapping/non-overlapping patch embedding for 1D signals."""

    def __init__(
            self,
            in_ch: int,
            embed_dim: int,
            patch_size: int = 16,
            stride: int | None = None,
            padding: int | None = None,
            use_overlap: bool = True,
    ) -> None:
        super().__init__()
        stride = patch_size if stride is None else stride
        if use_overlap:
            padding = (patch_size // 2) if padding is None else padding
        else:
            padding = 0 if padding is None else padding

        self.proj = nn.Conv1d(
            in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class MSA1D(nn.Module):
    """Multi-head self-attention (no SR) for 1D tokens."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, t, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock1D(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MSA1D(dim, num_heads, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTBackbone1D(nn.Module):
    """Plain ViT backbone for 1D signals."""

    def __init__(
            self,
            in_channels: int,
            embed_dim: int = 256,
            depth: int = 8,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            patch_size: int = 16,
            patch_stride: int | None = None,
            use_overlap: bool = True,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed1D(
            in_ch=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride=patch_stride,
            use_overlap=use_overlap,
        )
        self.embed_dim = embed_dim
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                ViTBlock1D(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        t_len = x.shape[-1]
        pe = get_1d_sincos_pos_embed(self.embed_dim, t_len).to(x.device)
        x = x + pe
        x = x.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


# -----------------------------------------------------------------------------
# Neck and Head
# -----------------------------------------------------------------------------
class TokenPyramid1D(nn.Module):
    """Build S1..S4 by temporal pooling of ViT tokens + 1x1 conv projection."""

    def __init__(
            self,
            in_dim: int,
            stage_dims: Sequence[int] = (64, 128, 320, 512),
    ) -> None:
        super().__init__()
        self.stage_dims = stage_dims
        self.proj = nn.ModuleList(
            [nn.Conv1d(in_dim, d, kernel_size=1, bias=False) for d in stage_dims]
        )
        self.bn = nn.ModuleList([nn.BatchNorm1d(d) for d in stage_dims])
        self.act = nn.ModuleList([nn.ReLU(inplace=True) for _ in stage_dims])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, e, t = x.shape
        t2 = max(1, t // 2)
        t4 = max(1, t // 4)
        t8 = max(1, t // 8)
        s1 = x
        s2 = F.adaptive_avg_pool1d(x, t2)
        s3 = F.adaptive_avg_pool1d(x, t4)
        s4 = F.adaptive_avg_pool1d(x, t8)
        outs_raw = [s1, s2, s3, s4]
        outs: List[torch.Tensor] = []
        for i, f in enumerate(outs_raw):
            o = self.proj[i](f)
            o = self.bn[i](o)
            o = self.act[i](o)
            outs.append(o)
        return outs


class SegViTDecodeHead1D(nn.Module):
    """Project each stage -> same dim, upsample to finest, concat, fuse, classify."""

    def __init__(
            self,
            in_dims: Sequence[int],
            decoder_dim: int = 256,
            num_classes: int = 1,
            dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(c, decoder_dim, 1, bias=False),
                    nn.BatchNorm1d(decoder_dim),
                    nn.ReLU(inplace=True),
                )
                for c in in_dims
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(decoder_dim * len(in_dims), decoder_dim, 1, bias=False),
            nn.BatchNorm1d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.cls = nn.Conv1d(decoder_dim, num_classes, 1)

    def forward(self, feats: List[torch.Tensor], out_len: int) -> torch.Tensor:
        ref_len = feats[0].shape[-1]
        ups = []
        for f, p in zip(feats, self.proj):
            v = p(f)
            if v.shape[-1] != ref_len:
                v = F.interpolate(v, size=ref_len, mode="linear", align_corners=False)
            ups.append(v)
        x = torch.cat(ups, dim=1)
        x = self.fuse(x)
        x = self.cls(x)
        if x.shape[-1] != out_len:
            x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        return x


# -----------------------------------------------------------------------------
# SegViT 1D
# -----------------------------------------------------------------------------
class SegViT1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            embed_dim: int,
            depth: int,
            num_heads: int,
            mlp_ratio: float,
            drop_rate: float,
            attn_drop_rate: float,
            drop_path_rate: float,
            patch_size: int,
            use_overlap: bool,
            stage_dims: Sequence[int] = (64, 128, 128, 256),
            decoder_dim: int = 256,
            num_classes: int = 1,
    ) -> None:
        super().__init__()

        # 1. Backbone
        self.backbone = ViTBackbone1D(in_channels=in_channels,
                                      embed_dim=embed_dim,
                                      depth=depth,
                                      num_heads=num_heads,
                                      mlp_ratio=mlp_ratio,
                                      drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate,
                                      drop_path_rate=drop_path_rate,
                                      patch_size=patch_size,
                                      use_overlap=use_overlap)
        embed_dim = self.backbone.embed_dim

        # 2. Token Pyramid Neck
        self.tpa = TokenPyramid1D(in_dim=embed_dim, stage_dims=stage_dims)

        # 3. Decoder Head
        self.decode_head = SegViTDecodeHead1D(
            in_dims=stage_dims, decoder_dim=decoder_dim, num_classes=num_classes
        )
        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.norm(x)
        orig_len = x.shape[-1]

        # Pad input to be divisible by a safe multiple (e.g., 128)
        x, pad = pad_to_multiple_1d(x, multiple=128)

        # Backbone -> Tokens
        tokens = self.backbone(x)

        # Tokens -> Pyramid Features
        pyramid_features = self.tpa(tokens)

        # Decode pyramid to logits
        logits = self.decode_head(pyramid_features, out_len=x.shape[-1])

        # Unpad to match original input length
        logits = right_unpad_1d(logits, pad)

        assert logits.shape[-1] == orig_len, "Output length must match input length"
        return logits

