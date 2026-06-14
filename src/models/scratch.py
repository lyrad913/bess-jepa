"""Most of Codes are from https://github.com/lucas-maes/le-wm
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
} 
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def apply_mask(x, masks, concat=True, masked=True):
    """https://github.com/Sennadir/TS_JEPA/blob/main/src/models/utils/mask_utils.py
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding with lazy buffer growth."""

    def __init__(self, dim: int, max_len: int = 0, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"sinusoidal PE needs an even dim, got {dim}")
        self.dim = dim
        self.base = base
        self.register_buffer("pe", self._build(max_len), persistent=False)

    def _build(self, length: int, device: torch.device | None = None) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        omega = torch.arange(self.dim // 2, dtype=torch.float32, device=device) / (self.dim / 2.0)
        omega = 1.0 / (self.base ** omega)
        out = pos * omega.unsqueeze(0)
        return torch.cat([out.sin(), out.cos()], dim=-1).unsqueeze(0)

    def _ensure_len(self, length: int, device: torch.device) -> None:
        if length <= self.pe.shape[1] and self.pe.device == device:
            return
        self.pe = self._build(length, device=device)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, D)
        positions: optional (N,) or (B, N) original patch indices.
        """
        if positions is None:
            self._ensure_len(x.shape[1], x.device)
            pe = self.pe[:, : x.shape[1]]
        else:
            positions = positions.to(device=x.device, dtype=torch.long)
            self._ensure_len(int(positions.max().item()) + 1, x.device)
            if positions.ndim == 1:
                pe = self.pe[:, positions]
            elif positions.ndim == 2:
                pe = self.pe[0, positions]
            else:
                raise ValueError(f"positions must have shape (N,) or (B, N), got {tuple(positions.shape)}")
        return x + pe.to(dtype=x.dtype)

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, is_casual = False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, is_causal = False):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        self.is_causal = is_causal

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), causal = self.is_causal)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, is_causal=False):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.is_causal = is_causal

    def forward(self, x):
        x = x + self.attn(self.norm1(x), causal = self.is_causal)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        is_causal = False
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, is_causal)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x
    
    
class GapEmbedder(nn.Module):
    def __init__(self, input_dim=2, emb_dim=128, mlp_scale=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, gap):
        # gap: (B,)
        gap = gap.float().view(-1, 1)
        x = torch.cat([gap, torch.log1p(gap)], dim=-1)  # (B, 2)
        x = self.embed(x)                               # (B, D)
        return x[:, None, :]                            # (B, 1, D)

# class Embedder(nn.Module):
#     def __init__(
#         self,
#         input_dim=10,
#         smoothed_dim=10,
#         emb_dim=10,
#         mlp_scale=4,
#     ):
#         super().__init__()
#         self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
#         self.embed = nn.Sequential(
#             nn.Linear(smoothed_dim, mlp_scale * emb_dim),
#             nn.SiLU(),
#             nn.Linear(mlp_scale * emb_dim, emb_dim),
#         )

#     def forward(self, x):
#         """
#         x: (B, T, D)
#         """
#         x = x.float()
#         x = x.permute(0, 2, 1)
#         x = self.patch_embed(x)
#         x = x.permute(0, 2, 1)
#         x = self.embed(x)
#         return x


# class MLP(nn.Module):
#     """Simple MLP with optional normalization and activation"""

#     def __init__(
#         self,
#         input_dim,
#         hidden_dim,
#         output_dim=None,
#         norm_fn=nn.LayerNorm,
#         act_fn=nn.GELU,
#     ):
#         super().__init__()
#         norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             norm_fn,
#             act_fn(),
#             nn.Linear(hidden_dim, output_dim or input_dim),
#         )

#     def forward(self, x):
#         """
#         x: (B*T, D)
#         """
#         return self.net(x)


# class ARPredictor(nn.Module):
#     """Autoregressive predictor for next-step embedding prediction."""

#     def __init__(
#         self,
#         *,
#         num_frames,
#         depth,
#         heads,
#         mlp_dim,
#         input_dim,
#         hidden_dim,
#         output_dim=None,
#         dim_head=64,
#         dropout=0.0,
#         emb_dropout=0.0,
#     ):
#         super().__init__()
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
#         self.dropout = nn.Dropout(emb_dropout)
#         self.transformer = Transformer(
#             input_dim,
#             hidden_dim,
#             output_dim or input_dim,
#             depth,
#             heads,
#             dim_head,
#             mlp_dim,
#             dropout,
#             block_class=ConditionalBlock,
#         )

#     def forward(self, x, c):
#         """
#         x: (B, T, d)
#         c: (B, T, act_dim)
#         """
#         T = x.size(1)
#         x = x + self.pos_embedding[:, :T]
#         x = self.dropout(x)
#         x = self.transformer(x, c)
#         return x
