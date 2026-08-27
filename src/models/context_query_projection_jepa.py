import torch
import torch.nn as nn
import torch.nn.functional as F

from projection_jepa import ProjectionJEPA
from sub_models import ConditionalBlock, GapEmbedder, SinusoidalPE, Transformer
from ts_jepa import gather_tokens, random_mask_indices


class ContextQueryPredictor(nn.Module):
    """Predict a full future window from visible context tokens and mask queries."""

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
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
        keep_idx: torch.Tensor,
        mask_idx: torch.Tensor,
        gap: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, embed_dim = ctx.shape
        mask_queries = self.mask_token.expand(batch_size, mask_idx.shape[1], embed_dim)

        positions = torch.cat([keep_idx, mask_idx], dim=1)
        x = torch.cat([ctx, mask_queries], dim=1)
        x = self.pos(x, positions=positions)

        gap = gap.to(device=ctx.device, dtype=ctx.dtype).view(batch_size, 1)
        condition = self.gap_embed(gap).to(device=ctx.device, dtype=ctx.dtype)
        x = self.transformer(x, c=condition)

        # Restore [keep | mask] outputs to the target window's temporal order.
        ordered = torch.empty_like(x)
        ordered.scatter_(
            dim=1,
            index=positions.unsqueeze(-1).expand(-1, -1, embed_dim),
            src=x,
        )
        return ordered


class ContextQueryProjectionJEPA(ProjectionJEPA):
    """
    Future JEPA with a visible-context bottleneck and explicit mask queries.

    The online encoder sees only unmasked context tokens. The predictor combines
    those representations with learned queries at the removed positions and
    predicts the complete future target window. Context and target use the same
    trainable tokenizer/encoder; the target path is not detached or stop-grad.
    """

    def __init__(
        self,
        *args,
        future_context_mask_ratio: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.future_context_mask_ratio = float(future_context_mask_ratio)
        self.hparams.future_context_mask_ratio = self.future_context_mask_ratio
        self.predictor = ContextQueryPredictor(
            embed_dim=self.hparams.embed_dim,
            nhead=self.hparams.pred_nhead,
            num_layers=self.hparams.pred_layers,
        )

    def module_step(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x"]
        y = batch["y"]
        gap = batch["gap"]

        context_tokens = self.tokenizer(x)
        keep_idx, mask_idx = random_mask_indices(
            batch_size=context_tokens.shape[0],
            n_patches=context_tokens.shape[1],
            mask_ratio=self.future_context_mask_ratio,
            device=context_tokens.device,
        )
        visible_tokens = gather_tokens(context_tokens, keep_idx)
        ctx_emb = self.encoder(visible_tokens, positions=keep_idx)

        # Deliberately train through the target branch: no EMA, no no_grad, no detach.
        tgt_emb = self.encoder(self.tokenizer(y))
        pred_emb = self.predictor(ctx_emb, keep_idx, mask_idx, gap)

        z = torch.cat([ctx_emb, tgt_emb], dim=1)
        z_proj = self.projector(z)

        pred_loss = F.smooth_l1_loss(pred_emb, tgt_emb)
        sigreg_loss = self.sigreg(z_proj.transpose(0, 1))
        loss = pred_loss + self.sigreg_lambda * sigreg_loss
        return loss, pred_loss, sigreg_loss, z, z_proj
