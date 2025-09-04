# -*- coding: utf-8 -*-
"""Multi-modal 1D SegFormer (MiT backbone + MLP decode head)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
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
# MiT building blocks (1D)
# -----------------------------------------------------------------------------
class OverlapPatchEmbed1D(nn.Module):
    """Overlapping patch embedding for 1D signals.

    Conv1d with stride>1 + LayerNorm on channel dimension (applied in (B, T, C)).
    """

    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        pad = (kernel_size // 2) if padding is None else padding
        self.proj = nn.Conv1d(
            in_ch, embed_dim, kernel_size=kernel_size, stride=stride, padding=pad
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, E, T')
        x = self.proj(x)
        x = x.transpose(1, 2)      # (B, T', E)
        x = self.norm(x)
        x = x.transpose(1, 2)      # (B, E, T')
        return x


class Attention1D(nn.Module):
    """Multi-head self-attention with spatial reduction (SR) for 1D."""

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
            # depthwise conv to reduce temporal length
            self.sr = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio,
                                groups=dim, bias=False)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        q = self.q(x).reshape(b, t, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, H, T, Dh)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)              # (B, C, T)
            x_ = self.sr(x_)                    # (B, C, T//sr)
            x_ = x_.transpose(1, 2)            # (B, T//sr, C)
            x_ = self.norm(x_)
            k = self.k(x_)
            v = self.v(x_)
            tk = x_.shape[1]
        else:
            k = self.k(x)
            v = self.v(x)
            tk = t

        k = k.reshape(b, tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, Tk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                   # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(b, t, c)       # (B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MixFFN1D(nn.Module):
    """FFN with depthwise Conv to mix local 1D context."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                                groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        x = self.fc1(x)                    # (B, T, H)
        x = x.transpose(1, 2)              # (B, H, T)
        x = self.dwconv(x)                 # (B, H, T)
        x = x.transpose(1, 2)              # (B, T, H)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)                    # (B, T, C)
        x = self.drop(x)
        return x


class EncoderBlock1D(nn.Module):
    """Pre-norm Transformer block (Attn + Mix-FFN) for 1D."""

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
        self.attn = Attention1D(
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
        # x: (B, T, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MiTBackbone1D(nn.Module):
    """MiT backbone (SegFormer) for 1D signals.

    Overlapping patch embeddings + hierarchical Transformer encoder.
    Outputs 4 stages: S1(/4), S2(/8), S3(/16), S4(/32)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dims: Sequence[int] = (64, 128, 320, 512),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        patch_kernel_sizes: Sequence[int] = (7, 3, 3, 3),
        patch_strides: Sequence[int] = (4, 2, 2, 2),
    ) -> None:
        super().__init__()
        assert len({len(embed_dims), len(depths), len(num_heads), len(sr_ratios),
                    len(patch_kernel_sizes), len(patch_strides)}) == 1

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        dpr_i = 0

        self.stages = nn.ModuleList()
        self.blocks = nn.ModuleList()

        in_ch = in_channels
        for i in range(4):
            # Overlapping patch embed
            stage = OverlapPatchEmbed1D(
                in_ch=in_ch,
                embed_dim=embed_dims[i],
                kernel_size=patch_kernel_sizes[i],
                stride=patch_strides[i],
            )
            self.stages.append(stage)
            in_ch = embed_dims[i]

            # Transformer blocks
            blk_list: List[nn.Module] = []
            for _ in range(depths[i]):
                blk_list.append(
                    EncoderBlock1D(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[dpr_i],
                        sr_ratio=sr_ratios[i],
                    )
                )
                dpr_i += 1
            self.blocks.append(nn.ModuleList(blk_list))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, C, T)
        feats: List[torch.Tensor] = []
        h = x
        for stage, blk_list in zip(self.stages, self.blocks):
            h = stage(h)                   # (B, E, T')
            h_ln = h.transpose(1, 2)       # (B, T', E)
            for blk in blk_list:
                h_ln = blk(h_ln)           # (B, T', E)
            h = h_ln.transpose(1, 2)       # (B, E, T')
            feats.append(h)
        # feats: [S1(/4), S2(/8), S3(/16), S4(/32)]
        return feats


# -----------------------------------------------------------------------------
# SegFormer decode head (1D)
# -----------------------------------------------------------------------------
class SegFormerDecodeHead1D(nn.Module):
    """SegFormer MLP-style decode head (1D).

    - Project each stage to 'decoder_dim' with 1x1 conv
    - Upsample all to the finest stage length (S1)
    - Concatenate and fuse -> classifier -> upsample to original length
    """

    def __init__(
        self,
        in_dims: Sequence[int],
        decoder_dim: int = 256,
        num_classes: int = 1,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(c, decoder_dim, 1, bias=False),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(inplace=True),
            ) for c in in_dims]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(decoder_dim * len(in_dims), decoder_dim, 1, bias=False),
            nn.BatchNorm1d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.classifier = nn.Conv1d(decoder_dim, num_classes, 1)

    def forward(
        self,
        feats: List[torch.Tensor],
        out_len: int,
    ) -> torch.Tensor:
        # feats: [S1,S2,S3,S4] each (B, C_i, T_i)
        assert len(feats) >= 2
        ref_len = feats[0].shape[-1]
        ups = []
        for f, proj in zip(feats, self.proj):
            p = proj(f)
            if p.shape[-1] != ref_len:
                p = F.interpolate(p, size=ref_len, mode="linear", align_corners=False)
            ups.append(p)
        x = torch.cat(ups, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        if x.shape[-1] != out_len:
            x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        return x


# -----------------------------------------------------------------------------
# Multi-modal SegFormer-1D (fusion at each stage)
# -----------------------------------------------------------------------------
class MultiModalSegFormer1D(nn.Module):
    """Multi-modal 1D SegFormer with stage-wise fusion ('sum' | 'concat' | 'attn')."""

    def __init__(
        self,
        modalities: Dict[str, int],
        backbone_cfg: Optional[Dict] = None,
        fusion: str = "sum",
        decoder_dim: int = 256,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        assert fusion in {"sum", "concat", "attn"}
        self.modalities = list(modalities.keys())
        self.fusion = fusion

        backbone_cfg = backbone_cfg or {}
        self.backbones = nn.ModuleDict({
            m: MiTBackbone1D(in_channels=cin, **backbone_cfg)
            for m, cin in modalities.items()
        })

        # Infer stage dims from any backbone (assumed identical configs)
        any_bb = next(iter(self.backbones.values()))
        with torch.no_grad():
            # Fake pass to infer dims robustly? Avoid heavy op; trust cfg
            embed_dims = tuple(backbone_cfg.get("embed_dims", (64, 128, 320, 512)))
        self.stage_dims = embed_dims  # (d1, d2, d3, d4)

        # For fusion=concat, reduce back to stage dim per stage
        if self.fusion == "concat":
            self.reducers = nn.ModuleList([
                nn.Conv1d(len(self.modalities) * d, d, 1, bias=False)
                for d in self.stage_dims
            ])
        else:
            self.reducers = None

        # For fusion=attn, learn weights per stage from pooled descriptors
        if self.fusion == "attn":
            self.attn_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, d // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(d // 4, len(self.modalities)),
                )
                for d in self.stage_dims
            ])
        else:
            self.attn_mlps = None

        self.decode_head = SegFormerDecodeHead1D(
            in_dims=self.stage_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

    def _fuse_stage(self, feats_m: List[torch.Tensor], stage_idx: int) -> torch.Tensor:
        """Fuse same-stage features from multiple modalities."""
        if self.fusion == "sum":
            out = torch.stack(feats_m, dim=0).sum(dim=0)
        elif self.fusion == "concat":
            cat = torch.cat(feats_m, dim=1)
            out = self.reducers[stage_idx](cat)
        else:  # "attn"
            # weights from first modality descriptor as query (simple, efficient)
            desc = [f.mean(dim=-1) for f in feats_m]      # each (B, C)
            query = desc[0]                                # (B, C)
            weights = self.attn_mlps[stage_idx](query)     # (B, M)
            weights = F.softmax(weights, dim=-1).unsqueeze(-1).unsqueeze(-1)
            stack = torch.stack(feats_m, dim=1)           # (B, M, C, T)
            out = (stack * weights).sum(dim=1)            # (B, C, T)
        return out

    def forward(self, xdict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(xdict, dict) and len(xdict) > 0

        # Align batch & lengths; pad to multiple of 32 (MiT uses /32 at S4)
        batch = None
        t_max = 0
        for x in xdict.values():
            assert x.dim() == 3, "Each modality must be (B, C, T)."
            batch = x.size(0) if batch is None else batch
            assert x.size(0) == batch, "All modalities must share batch size."
            t_max = max(t_max, x.size(-1))

        padded: Dict[str, torch.Tensor] = {}
        for m, x in xdict.items():
            if x.shape[-1] < t_max:
                x = F.pad(x, (0, t_max - x.shape[-1]))
            x, _ = pad_to_multiple_1d(x, multiple=32)
            padded[m] = x
        out_len = t_max

        # Per-modality backbone features
        feats_per_mod = {m: self.backbones[m](padded[m]) for m in self.modalities}
        # feats_per_mod[m] : [S1, S2, S3, S4] each (B, C_i, T_i)

        # Stage-wise fusion across modalities
        fused_feats: List[torch.Tensor] = []
        for s_idx in range(4):
            feats_m = [feats_per_mod[m][s_idx] for m in self.modalities]
            fused = self._fuse_stage(feats_m, s_idx)
            fused_feats.append(fused)

        # Decode to logits at original length
        logits = self.decode_head(fused_feats, out_len=out_len)  # (B, C, T)
        return logits


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    bsz, t_len = 2, 5000
    xdict = {
        "ecg": torch.randn(bsz, 1, t_len),
        "ppg": torch.randn(bsz, 1, t_len - 111),
        "imu": torch.randn(bsz, 6, t_len + 17),
    }

    model = MultiModalSegFormer1D(
        modalities={"ecg": 1, "ppg": 1, "imu": 6},
        backbone_cfg=dict(
            embed_dims=(64, 128, 320, 512),
            depths=(2, 2, 2, 2),
            num_heads=(1, 2, 5, 8),
            sr_ratios=(8, 4, 2, 1),
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            patch_kernel_sizes=(7, 3, 3, 3),
            patch_strides=(4, 2, 2, 2),
        ),
        fusion="attn",          # "sum" | "concat" | "attn"
        decoder_dim=256,
        num_classes=1,
    )

    y = model(xdict)
    print("logits:", y.shape)  # (B, 1, T)
