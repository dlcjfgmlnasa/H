# -*- coding: utf-8 -*-
"""Multi-modal 1D SegViT (ViT backbone + Token Pyramid + MLP decode head)."""

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


def get_1d_sincos_pos_embed(embed_dim: int, t_len: int) -> torch.Tensor:
    """Standard 1D sine-cos positional embedding. Returns (1, C, T)."""
    assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos."
    position = torch.arange(t_len).float()  # (T,)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
    )  # (C/2,)

    pe = torch.zeros(t_len, embed_dim)
    pe[:, 0::2] = torch.sin(position[:, None] * div_term[None, :])
    pe[:, 1::2] = torch.cos(position[:, None] * div_term[None, :])
    pe = pe.t().unsqueeze(0)  # (1, C, T)
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
# ViT(1D): Patch Embedding + Transformer Blocks
# -----------------------------------------------------------------------------
class PatchEmbed1D(nn.Module):
    """Overlapping/non-overlapping patch embedding for 1D signals."""

    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        patch_size: int = 16,
        stride: Optional[int] = None,
        padding: Optional[int] = None,
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
        # x: (B, C, T) -> (B, E, T')
        x = self.proj(x)
        x = x.transpose(1, 2)  # (B, T', E)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, E, T')
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
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(b, t, c)  # (B, T, C)
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
        patch_stride: Optional[int] = None,
        use_overlap: bool = True,
        use_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed1D(
            in_ch=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride=patch_stride,
            use_overlap=use_overlap,
        )
        self.use_pos = use_pos_embed
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
        # x: (B, C, T)
        x = self.patch(x)  # (B, E, T')
        t_len = x.shape[-1]
        if self.use_pos:
            pe = get_1d_sincos_pos_embed(self.embed_dim, t_len).to(x.device)
            x = x + pe  # broadcast (1, E, T')
        x = x.transpose(1, 2)  # (B, T', E)
        for blk in self.blocks:
            x = blk(x)  # (B, T', E)
        x = self.norm(x)  # (B, T', E)
        x = x.transpose(1, 2)  # (B, E, T')
        return x  # tokens as (B, E, T')


# -----------------------------------------------------------------------------
# Token Pyramid Aggregation (TPA, 1D)
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
        # x: (B, E, T')
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
        return outs  # [S1, S2, S3, S4]


# -----------------------------------------------------------------------------
# Decode head (SegViT-style lightweight MLP head)
# -----------------------------------------------------------------------------
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
        # feats: [S1..S4] each (B, C_i, T_i)
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
# Multi-modal SegViT (1D)
# -----------------------------------------------------------------------------
class MultiModalSegViT1D(nn.Module):
    """Multi-modal 1D SegViT with stage-wise fusion ('sum' | 'concat' | 'attn')."""

    def __init__(
        self,
        modalities: Dict[str, int],
        backbone_cfg: Optional[Dict] = None,
        stage_dims: Sequence[int] = (64, 128, 320, 512),
        fusion: str = "sum",
        decoder_dim: int = 256,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        assert fusion in {"sum", "concat", "attn"}
        self.modalities = list(modalities.keys())
        self.fusion = fusion
        self.stage_dims = stage_dims

        # --- fix: avoid passing duplicated in_channels ---
        backbone_cfg = backbone_cfg or {}
        cfg = dict(backbone_cfg)          # copy to avoid side-effects
        cfg.pop("in_channels", None)      # remove if exists

        self.backbones = nn.ModuleDict(
            {
                m: ViTBackbone1D(in_channels=cin, **cfg)
                for m, cin in modalities.items()
            }
        )
        embed_dim = next(iter(self.backbones.values())).embed_dim

        # TPA per modality (separate 1x1 projections)
        self.tpas = nn.ModuleDict(
            {m: TokenPyramid1D(in_dim=embed_dim, stage_dims=stage_dims)
             for m in self.modalities}
        )

        # concat fusion reducers per stage
        if self.fusion == "concat":
            self.reducers = nn.ModuleList(
                [
                    nn.Conv1d(
                        len(self.modalities) * d, d, kernel_size=1, bias=False
                    )
                    for d in stage_dims
                ]
            )
        else:
            self.reducers = None

        # attention fusion mlps per stage (descriptor -> weights for M modalities)
        if self.fusion == "attn":
            self.attn_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d, d // 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(d // 4, len(self.modalities)),
                    )
                    for d in stage_dims
                ]
            )
        else:
            self.attn_mlps = None

        self.decode_head = SegViTDecodeHead1D(
            in_dims=stage_dims, decoder_dim=decoder_dim, num_classes=num_classes
        )

    def _fuse_stage(self, feats_m: List[torch.Tensor], s_idx: int) -> torch.Tensor:
        """Fuse same-stage features from multiple modalities."""
        if self.fusion == "sum":
            out = torch.stack(feats_m, dim=0).sum(dim=0)
        elif self.fusion == "concat":
            cat = torch.cat(feats_m, dim=1)
            out = self.reducers[s_idx](cat)
        else:  # "attn"
            desc = [f.mean(dim=-1) for f in feats_m]  # (B, C)
            query = desc[0]                           # simple/effective
            weights = self.attn_mlps[s_idx](query)    # (B, M)
            weights = F.softmax(weights, dim=-1).unsqueeze(-1).unsqueeze(-1)
            stack = torch.stack(feats_m, dim=1)       # (B, M, C, T)
            out = (stack * weights).sum(dim=1)        # (B, C, T)
        return out

    def forward(self, xdict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(xdict, dict) and len(xdict) > 0

        # Align batch/length; pad to multiple of 128 (safe for patch/TPA chain)
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
            x, _ = pad_to_multiple_1d(x, multiple=128)
            padded[m] = x
        out_len = t_max

        # Per-modality: ViT tokens -> TPA stages
        stages_per_mod: Dict[str, List[torch.Tensor]] = {}
        for m in self.modalities:
            tokens = self.backbones[m](padded[m])  # (B, E, T')
            stages = self.tpas[m](tokens)          # [S1..S4]
            stages_per_mod[m] = stages

        # Stage-wise fusion
        fused: List[torch.Tensor] = []
        for s_idx in range(4):
            feats_m = [stages_per_mod[m][s_idx] for m in self.modalities]
            fused.append(self._fuse_stage(feats_m, s_idx))

        # Decode to original length
        logits = self.decode_head(fused, out_len=out_len)  # (B, C, T)
        return logits


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    bsz, t_len = 2, 5000
    xdict = {
        "ecg": torch.randn(bsz, 1, t_len),
        "ppg": torch.randn(bsz, 1, t_len - 77),
        "imu": torch.randn(bsz, 6, t_len + 33),
    }

    model = MultiModalSegViT1D(
        modalities={"ecg": 1, "ppg": 1, "imu": 6},
        backbone_cfg=dict(
            # in_channels는 모달리티별로 다르므로 여기 넣지 않습니다!
            embed_dim=256,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            patch_size=16,
            patch_stride=None,   # None -> stride=patch_size
            use_overlap=True,
            use_pos_embed=True,
        ),
        stage_dims=(64, 128, 320, 512),
        fusion="attn",           # "sum" | "concat" | "attn"
        decoder_dim=256,
        num_classes=1,
    )

    y = model(xdict)
    print("logits:", y.shape)  # (B, 1, T)
