import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa import JEPA
from ts_jepa import random_mask_indices


class ContextMaskJEPA(JEPA):
    """
    JEPA variant where future prediction sees a randomly masked context.

    This keeps the original JEPA objective:
        predictor(encoder(masked_tokenizer(x)), gap) -> encoder(tokenizer(y))

    There is no in-window masked prediction objective for the current context.
    """

    def __init__(
        self,
        *args,
        future_context_mask_ratio: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.future_context_mask_ratio = float(future_context_mask_ratio)
        self.hparams.future_context_mask_ratio = self.future_context_mask_ratio
        self.future_mask_token = nn.Parameter(torch.zeros(1, 1, self.hparams.embed_dim))
        nn.init.trunc_normal_(self.future_mask_token, std=0.02)

    def _mask_context_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        _, mask_idx = random_mask_indices(
            batch_size=tokens.shape[0],
            n_patches=tokens.shape[1],
            mask_ratio=self.future_context_mask_ratio,
            device=tokens.device,
        )
        masked_tokens = tokens.clone()
        mask_token = self.future_mask_token.to(device=tokens.device, dtype=tokens.dtype)
        masked_tokens.scatter_(
            dim=1,
            index=mask_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
            src=mask_token.expand(tokens.shape[0], mask_idx.shape[1], tokens.shape[-1]),
        )
        return masked_tokens

    def module_step(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            x = batch["x"]                           # (B, C, T)
            y = batch["y"]                           # (B, C, T)
            gap = batch["gap"]                       # (B,)
        else:
            raise Exception

        ctx_tokens = self._mask_context_tokens(self.tokenizer(x))
        tgt_tokens = self.tokenizer(y)

        ctx_emb = self.encoder(ctx_tokens)
        tgt_emb = self.encoder(tgt_tokens)
        pred_emb = self.predictor(ctx_emb, gap)

        pred_loss = F.smooth_l1_loss(pred_emb, tgt_emb)
        proj = torch.cat([ctx_emb, tgt_emb], dim=1).transpose(0, 1)
        sigreg_loss = self.sigreg(proj)

        loss = pred_loss + self.sigreg_lambda * sigreg_loss
        return loss, pred_loss, sigreg_loss, torch.cat([ctx_emb, tgt_emb], dim=1)
