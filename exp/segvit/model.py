# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple

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
    """Standard 1D sine-cos positional embedding. Returns (1, T, C)."""
    assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos."
    position = torch.arange(t_len).float()
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(t_len, embed_dim)
    pe[:, 0::2] = torch.sin(position[:, None] * div_term[None, :])
    pe[:, 1::2] = torch.cos(position[:, None] * div_term[None, :])
    return pe.unsqueeze(0)


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
# ViT(1D) Components
# -----------------------------------------------------------------------------
class PatchEmbed1D(nn.Module):
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
        # x: (B, C, T) -> proj -> (B, embed_dim, T_new)
        x = self.proj(x)
        # (B, embed_dim, T_new) -> transpose -> (B, T_new, embed_dim) for LayerNorm
        x = x.transpose(1, 2)
        x = self.norm(x)
        # Keep (B, T_new, embed_dim) for Transformer blocks
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
        # x: (B, T, C)
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
        self.patch_size = patch_size
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
        # x: (B, C, T_original)
        x_patched = self.patch(x)
        t_len_after_patch = x_patched.shape[1]

        pe = get_1d_sincos_pos_embed(self.embed_dim, t_len_after_patch).to(x.device)
        x_patched = x_patched + pe

        for blk in self.blocks:
            x_patched = blk(x_patched)

        x_patched = self.norm(x_patched)  # (B, T_new, embed_dim)
        return x_patched


# -----------------------------------------------------------------------------
# ATM (Attention To Mask) Module for Query-based Segmentation
# -----------------------------------------------------------------------------
class CrossAttention1D(nn.Module):
    """Cross-attention module where Query is class embedding, Key/Value are image tokens."""

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 8, drop: float = 0.0) -> None:
        super().__init__()
        assert query_dim % num_heads == 0 and kv_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim, bias=True)
        self.k_proj = nn.Linear(kv_dim, kv_dim, bias=True)
        self.v_proj = nn.Linear(kv_dim, kv_dim, bias=True)

        self.proj_out = nn.Linear(query_dim, query_dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        b, n_cls, c_q = query.shape
        _, t_enc, _ = key.shape

        q = self.q_proj(query).reshape(b, n_cls, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(b, t_enc, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(b, t_enc, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(b, n_cls, c_q)
        out = self.proj_out(out)
        out = self.proj_drop(out)
        return out


class ATMModule(nn.Module):
    """Attention To Mask module as described in the image."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.class_prediction_head = nn.Linear(embed_dim, 1)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention1D(
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            drop=attn_drop_rate
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, drop_rate)

    def forward(self, class_queries: torch.Tensor, encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self.norm1(class_queries)
        attn_output = self.cross_attn(queries, encoder_features, encoder_features)
        class_queries = class_queries + attn_output

        ff_output = self.mlp(self.norm2(class_queries))
        class_queries = class_queries + ff_output

        # 1. Class Predictions (FC in diagram)
        class_logits = self.class_prediction_head(class_queries).squeeze(-1)

        # 2. Generate Masks
        norm_encoder_features = F.normalize(encoder_features, p=2, dim=-1)
        norm_class_queries = F.normalize(class_queries, p=2, dim=-1)

        mask_logits = norm_class_queries @ norm_encoder_features.transpose(1, 2)

        return class_logits, mask_logits


# -----------------------------------------------------------------------------
# SegViT 1D (Modified with ATM)
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
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # 1. Backbone (ViT Encoder)
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

        # 2. Class Queries (Learnable embeddings for each class)
        self.class_queries = nn.Parameter(torch.randn(num_classes, embed_dim))

        # 3. ATM Decoder Head
        self.atm_module = ATMModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, pad = pad_to_multiple_1d(x, multiple=self.backbone.patch_size)

        encoder_features = self.backbone(x)

        batch_class_queries = self.class_queries.unsqueeze(0).repeat(x.shape[0], 1, 1)

        class_logits, masks = self.atm_module(batch_class_queries, encoder_features)

        if masks.shape[-1] != x.shape[-1]:  # Use padded length for interpolation
            masks = F.interpolate(masks, size=x.shape[-1], mode="linear", align_corners=False)

        masks = right_unpad_1d(masks, pad)
        return masks
