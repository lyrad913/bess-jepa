import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """Feed-forward network used in Transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product self-attention."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        attn_mask = None
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask[:, None, None, :].to(device=x.device, dtype=torch.bool)
            attn_mask = torch.zeros_like(mask, dtype=x.dtype).masked_fill(mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Block(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0, is_causal: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), src_key_padding_mask, is_causal=self.is_causal)
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalBlock(nn.Module):
    """Pre-norm Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0, is_causal: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.is_causal = is_causal

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            src_key_padding_mask,
            is_causal=self.is_causal,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    """Transformer stack supporting standard and AdaLN-zero conditional blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class: type[nn.Module] = Block,
        is_causal: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([
            block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, is_causal)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            if isinstance(block, ConditionalBlock):
                if c is None:
                    raise ValueError("ConditionalBlock requires conditioning tensor c")
                x = block(x, c, src_key_padding_mask)
            else:
                x = block(x, src_key_padding_mask)
        return self.output_proj(self.norm(x))


class GapEmbedder(nn.Module):
    """Embed scalar window gap into a broadcastable conditioning token."""

    def __init__(self, input_dim: int = 2, emb_dim: int = 128, mlp_scale: int = 4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, gap: torch.Tensor) -> torch.Tensor:
        gap = gap.float().view(-1, 1)
        x = torch.cat([gap, torch.log1p(gap)], dim=-1)
        return self.embed(x)[:, None, :]


class Tokenizer(nn.Module):
    """
    PatchTST Style Tokenizer except channel(feature) independent
    (batch_size, n_features, seq_len) 
    -> (batch_size, n_features, patch_len, n_patches) 
    -> (batch_size, n_patches, embed_dim)
    Default
    - seq_len : 96
    - patch_len : 16
    - strides : 8
    - n_patches = (seq_len - patch_len) / strides + 1 = 11

    """
    def __init__(
        self,
        seq_len:int, patch_len:int, strides:int,
        n_features:int, embed_dim:int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.strides = strides
        self.n_patches = (seq_len - patch_len) // strides + 1
        self.projection = nn.Linear(patch_len * n_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_feature, seq_len)
        """
        if x.shape[-1] < self.patch_len:
            raise ValueError(f"expected length >= patch_len={self.patch_len}, got {x.shape[-1]}")

        x = x.unfold(-1, self.patch_len, self.strides)  # (batch, n_features, n_patches, patch_len)
        x = rearrange(x, "b c n p -> b n (p c)")        # (batch, n_patches, patch_len * n_features)
        x = self.projection(x)  # (batch, n_patches, embed_dim)
        return x

class Encoder(nn.Module):
    """
    Multivariate patch encoder.

    Args:
        x: (B, N, D) – tokenized patches
    Returns:
        (B, N, D)
    """

    def __init__(self, embed_dim: int, nhead: int, num_layers: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.pos = SinusoidalPE(embed_dim)
        self.transformer = Transformer(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=num_layers,
            heads=nhead,
            dim_head=embed_dim // nhead,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
            block_class=Block,
            is_causal=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos(x, positions=positions)
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class Predictor(nn.Module):
    """
    Predicts target patch embeddings from context embeddings.
    <Flow>
    - context_patch = tokenizer(time_series[start: start+seq_len])
    - target_patch = tokenizer(time_series[start+seq_len+delta_t: start+2*seq_len+delta_t])
    - context_representation = encoder(context_patch)
    - target representation = encoder(target_patch)
    - predicted representation = predictor(context_representation, delta_t)
    """

    def __init__(self, embed_dim: int, nhead: int, num_layers: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.pos = SinusoidalPE(embed_dim)
        self.gap_embed = GapEmbedder(input_dim=2, emb_dim=embed_dim)
        self.transformer = Transformer(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=num_layers,
            heads=nhead,
            dim_head=embed_dim // nhead,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
            block_class=ConditionalBlock,
            is_causal=False,
        )

    def forward(
        self,
        ctx: torch.Tensor,
        gap: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = ctx.shape[0]
        if gap is not None:
            gap = gap.to(device=ctx.device, dtype=ctx.dtype).view(B, 1)
            c = self.gap_embed(gap).to(device=ctx.device, dtype=ctx.dtype)
        else:
            c = torch.zeros(B, 1, ctx.shape[-1], device=ctx.device, dtype=ctx.dtype)
        ctx = self.pos(ctx, positions=positions)
        return self.transformer(ctx, c=c)


class SIGReg(torch.nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer (single-GPU!)
    https://github.com/lucas-maes/le-wm/blob/main/module.py#L10
    """

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
        proj: (V, B, D), where V is a view/time axis and B is the sample axis.
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        t = self.t.to(dtype=proj.dtype)
        phi = self.phi.to(dtype=proj.dtype)
        weights = self.weights.to(dtype=proj.dtype)
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class Decoder(nn.Module):
    """Linear reconstruction head: 
    (batch_size, n_patches, embed_dim)
    -> (batch_size, n_features, seq_len)
    """
    def __init__(self, n_patches, embed_dim, n_features, seq_len):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.fc = nn.Linear(n_patches * embed_dim, n_features*seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b n d -> b (n d)")
        x = self.fc(x)
        return rearrange(x, "b (n c) -> b n c", n=self.n_features, c=self.seq_len)    
    
    
class RoPE(nn.Module):
    """
    Rotary Positioinal Encoding
    """
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        # dim은 head_dim이어야 합니다 (예: 64, 128)
        self.dim = dim
        
        # 1. 주파수 생성: theta_i = 10000^(-2(i-1)/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 2. 미리 최대 길이만큼의 위치(m) 배열 생성
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # 3. 외적(Outer product)을 통해 m * theta_i 계산
        freqss = torch.outer(t, self.inv_freq) # [max_seq_len, dim // 2]
        
        # 4. [theta_0..theta_{d/2-1}, theta_0..theta_{d/2-1}] 형태로 복사 (HuggingFace half-half 스타일)
        emb = torch.cat((freqss, freqss), dim=-1) # [max_seq_len, dim]
        
        # 5. cos, sin 값 저장
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :] # type:ignore
    
def rotate_half(x):
    # x의 마지막 차원(head_dim)을 절반으로 나눔
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    # [-x2, x1] 형태로 결합
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shape: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin shape: [seq_len, head_dim] -> 브로드캐스팅을 위해 차원 확장 필요
    cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim]
    
    # RoPE 공식 적용
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """
    RoPE가 적용된 Multi-Head Self-Attention.

    일반 nn.MultiheadAttention은 Q/K를 외부에서 수정할 수 없어서
    RoPE를 주입하려면 Attention 내부를 직접 구현해야 합니다.

    흐름:
        1. Q, K, V 선형 투영
        2. (B, N, D) → (B, heads, N, head_dim) 으로 reshape
        3. Q, K 에 RoPE 적용 → 위치 정보가 attention score에 반영됨
        4. F.scaled_dot_product_attention (FlashAttention 백엔드 자동 사용)
        5. (B, heads, N, head_dim) → (B, N, D) 로 합친 후 출력 투영
    """

    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = embed_dim // nhead  # 각 헤드가 담당하는 차원 크기

        # Q, K, V 를 각각 별도 선형층으로 투영 (bias 포함)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

        # RoPE: head_dim 기준으로 생성 (embed_dim 전체가 아님에 주의)
        self.rope = RoPE(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        # ── 1. Q, K, V 투영 ──────────────────────────────────────────────
        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ── 2. Multi-head 형태로 reshape ─────────────────────────────────
        # (B, N, D) → (B, N, heads, head_dim) → (B, heads, N, head_dim)
        q = q.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        # ── 3. RoPE 적용 ─────────────────────────────────────────────────
        # cos, sin: (N, head_dim) → apply_rotary_pos_emb 내부에서 (1,1,N,head_dim)으로 확장
        cos, sin = self.rope(N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── 4. Scaled dot-product attention ──────────────────────────────
        # src_key_padding_mask: (B, N), True인 위치는 무시(패딩)
        # SDPA가 기대하는 attn_mask 형태: (B, heads, N, N) 또는 (B, 1, 1, N) + additive (-inf)
        attn_mask = None
        if src_key_padding_mask is not None:
            # (B, N) → (B, 1, 1, N): key 방향으로만 마스킹
            attn_mask = src_key_padding_mask[:, None, None, :].float()
            attn_mask = attn_mask.masked_fill(src_key_padding_mask[:, None, None, :], float("-inf"))

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, heads, N, head_dim)

        # ── 5. 다시 합쳐서 출력 투영 ──────────────────────────────────────
        # (B, heads, N, head_dim) → (B, N, heads*head_dim) = (B, N, D)
        x = x.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-Zero: LayerNorm 출력에 gap-conditioned scale/shift 적용"""
    return x * (1 + scale) + shift


class TransformerEncoderBlock(nn.Module):
    """
    Pre-Norm Transformer Encoder 블록 (조건 없음).

        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, embed_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, nhead, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), src_key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


class ConditionalTransformerEncoderBlock(nn.Module):
    """
    AdaLN-Zero Transformer Encoder 블록 (gap 조건부).

    DiT 방식: shift/scale 외에 gate도 gap으로 제어.
    gate는 residual 기여도를 조절해 표현력을 높임.

        x = x + gate_attn * Attention(modulate(LayerNorm(x), shift, scale))
        x = x + gate_ffn  * FFN(modulate(LayerNorm(x), shift, scale))

    adaLN_modulation: zero init → 학습 초기에 identity처럼 동작
    """

    def __init__(self, embed_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, nhead, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )
        # gap_hidden → (shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)
        linear = nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        nn.init.constant_(linear.weight, 0)
        nn.init.constant_(linear.bias, 0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), linear)

    def forward(self, x: torch.Tensor, c: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # c: (B, D) gap hidden representation
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_a[:, None] * self.attn(modulate(self.norm1(x), shift_a[:, None], scale_a[:, None]), src_key_padding_mask)
        x = x + gate_f[:, None] * self.ff(modulate(self.norm2(x), shift_f[:, None], scale_f[:, None]))
        return x


class TransformerEncoder(nn.Module):
    """
    block_class로 TransformerEncoderBlock 또는 ConditionalTransformerEncoderBlock 선택.

    - Encoder: TransformerEncoderBlock (조건 없음)
    - Predictor: ConditionalTransformerEncoderBlock (gap 조건부)
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        block_class: type = TransformerEncoderBlock,
    ):
        super().__init__()
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            block_class(embed_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.layers:
            if isinstance(block, ConditionalTransformerEncoderBlock):
                assert c is not None
                x = block(x, c, src_key_padding_mask)
            else:
                x = block(x, src_key_padding_mask)
        return self.norm(x)
