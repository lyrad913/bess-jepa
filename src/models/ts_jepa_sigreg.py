from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange

from sub_models import Encoder, SIGReg, Tokenizer
from ts_jepa import MaskedEmbeddingPredictor, gather_tokens, random_mask_indices


class TSJEPASIGReg(L.LightningModule):
    """Masked TS-JEPA without EMA target encoder, using SIGReg anti-collapse."""

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
        sigreg_lambda: float = 0.1,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.mask_ratio = mask_ratio
        self.sigreg_lambda = sigreg_lambda
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # ema_momentum is accepted only for config compatibility with TSJEPA.
        _ = ema_momentum

        self.tokenizer = Tokenizer(seq_len, patch_size, strides, num_channels, embed_dim)
        self.encoder = Encoder(embed_dim, enc_nhead, enc_layers)
        self.predictor = MaskedEmbeddingPredictor(embed_dim, pred_nhead, pred_layers)
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def module_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x"]
        tokens = self.tokenizer(x)
        keep_idx, mask_idx = random_mask_indices(tokens.shape[0], tokens.shape[1], self.mask_ratio, tokens.device)

        visible_tokens = gather_tokens(tokens, keep_idx)
        ctx_z = self.encoder(visible_tokens, positions=keep_idx)
        pred_z = self.predictor(ctx_z, keep_idx, mask_idx)

        full_z = self.encoder(tokens)
        target_z = gather_tokens(full_z, mask_idx)

        pred_loss = F.smooth_l1_loss(pred_z, target_z)
        proj = rearrange(full_z, "b n d -> n b d")
        sigreg_loss = self.sigreg(proj)
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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
