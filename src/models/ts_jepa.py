from __future__ import annotations

import copy

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from sub_models import Block, Encoder, SIGReg, SinusoidalPE, Tokenizer, Transformer


def random_mask_indices(batch_size: int, n_patches: int, mask_ratio: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if n_patches < 2:
        raise ValueError(f"masking requires at least 2 patches, got {n_patches}")
    n_mask = max(1, min(n_patches - 1, int(round(n_patches * mask_ratio))))
    noise = torch.rand(batch_size, n_patches, device=device)
    ids = torch.argsort(noise, dim=1)
    mask_idx = ids[:, :n_mask]
    keep_idx = ids[:, n_mask:]
    return keep_idx.sort(dim=1).values, mask_idx.sort(dim=1).values


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class MaskedEmbeddingPredictor(nn.Module):
    """Predict masked target embeddings from visible context embeddings."""

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

    def forward(self, ctx: torch.Tensor, keep_idx: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        b, _, d = ctx.shape
        mask_tokens = self.mask_token.expand(b, mask_idx.shape[1], d)
        x = torch.cat([ctx, mask_tokens], dim=1)
        positions = torch.cat([keep_idx, mask_idx], dim=1)
        x = self.pos(x, positions=positions)
        x = self.transformer(x)
        return x[:, -mask_idx.shape[1] :]


class TSJEPA(L.LightningModule):
    """Masked TS-JEPA baseline for one time-series window."""

    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        strides: int,
        num_channels: int,
        embed_dim: int = 128,
        enc_nhead: int = 4,
        enc_layers: int = 6,
        pred_nhead: int = 4,
        pred_layers: int = 2,
        mask_ratio: float = 0.5,
        ema_momentum: float = 0.996,
        sigreg_lambda: float = 0.0,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.mask_ratio = mask_ratio
        self.ema_momentum = ema_momentum
        self.sigreg_lambda = sigreg_lambda
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.tokenizer = Tokenizer(seq_len, patch_size, strides, num_channels, embed_dim)
        self.encoder = Encoder(embed_dim, enc_nhead, enc_layers)
        self.target_tokenizer = copy.deepcopy(self.tokenizer)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_tokenizer.parameters():
            p.requires_grad = False
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = MaskedEmbeddingPredictor(embed_dim, pred_nhead, pred_layers)
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        m = self.ema_momentum
        for online, target in zip(self.tokenizer.parameters(), self.target_tokenizer.parameters(), strict=True):
            target.data.mul_(m).add_(online.data, alpha=1.0 - m)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.data.mul_(m).add_(online.data, alpha=1.0 - m)

    def module_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x"]
        tokens = self.tokenizer(x)
        keep_idx, mask_idx = random_mask_indices(tokens.shape[0], tokens.shape[1], self.mask_ratio, tokens.device)

        visible_tokens = gather_tokens(tokens, keep_idx)
        ctx_z = self.encoder(visible_tokens, positions=keep_idx)
        pred_z = self.predictor(ctx_z, keep_idx, mask_idx)

        with torch.no_grad():
            target_tokens = self.target_tokenizer(x)
            target_z = self.target_encoder(target_tokens)
            target_z = gather_tokens(target_z, mask_idx)

        pred_loss = F.smooth_l1_loss(pred_z, target_z)
        if self.sigreg_lambda:
            proj = rearrange(ctx_z, "b n d -> () (b n) d")
            sigreg_loss = self.sigreg(proj)
        else:
            sigreg_loss = pred_loss.new_zeros(())
        loss = pred_loss + self.sigreg_lambda * sigreg_loss
        return loss, pred_loss, sigreg_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, pred_loss, sigreg_loss = self.module_step(batch)
        self.log_dict({"train/loss": loss, "train/pred_loss": pred_loss, "train/sigreg_loss": sigreg_loss})
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss = self.module_step(batch)
        self.log_dict({"val/loss": loss, "val/pred_loss": pred_loss, "val/sigreg_loss": sigreg_loss})

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss = self.module_step(batch)
        self.log_dict({"test/loss": loss, "test/pred_loss": pred_loss, "test/sigreg_loss": sigreg_loss})

    def on_train_batch_end(self, outputs, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._update_target_encoder()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        max_epochs = self.trainer.max_epochs or 100
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - self.warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
