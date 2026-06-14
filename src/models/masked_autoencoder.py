from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from sub_models import Block, Encoder, SinusoidalPE, Tokenizer, Transformer


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


def patchify(x: torch.Tensor, patch_size: int, strides: int) -> torch.Tensor:
    patches = x.unfold(-1, patch_size, strides)
    return patches.permute(0, 2, 3, 1).reshape(x.shape[0], patches.shape[2], patch_size * x.shape[1])


class MaskedAutoencoder(L.LightningModule):
    """Time-series masked autoencoder baseline."""

    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        strides: int,
        num_channels: int,
        embed_dim: int = 128,
        enc_nhead: int = 4,
        enc_layers: int = 6,
        decoder_nhead: int | None = None,
        decoder_layers: int = 2,
        decoder_embed_dim: int | None = None,
        mask_ratio: float = 0.75,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        pred_nhead: int | None = None,
        pred_layers: int | None = None,
        sigreg_lambda: float | None = None,
        sigreg_knots: int | None = None,
        sigreg_num_proj: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.patch_size = patch_size
        self.strides = strides
        self.num_channels = num_channels
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        decoder_embed_dim = decoder_embed_dim or embed_dim
        decoder_nhead = decoder_nhead or enc_nhead

        self.tokenizer = Tokenizer(seq_len, patch_size, strides, num_channels, embed_dim)
        self.encoder = Encoder(embed_dim, enc_nhead, enc_layers)
        self.enc_to_dec = nn.Linear(embed_dim, decoder_embed_dim) if embed_dim != decoder_embed_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos = SinusoidalPE(decoder_embed_dim)
        self.decoder = Transformer(
            input_dim=decoder_embed_dim,
            hidden_dim=decoder_embed_dim,
            output_dim=decoder_embed_dim,
            depth=decoder_layers,
            heads=decoder_nhead,
            dim_head=decoder_embed_dim // decoder_nhead,
            mlp_dim=4 * decoder_embed_dim,
            dropout=0.0,
            block_class=Block,
            is_causal=False,
        )
        self.reconstruction_head = nn.Linear(decoder_embed_dim, patch_size * num_channels)

    def module_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["x"]
        raw_patches = patchify(x, self.patch_size, self.strides)
        tokens = self.tokenizer(x)
        keep_idx, mask_idx = random_mask_indices(tokens.shape[0], tokens.shape[1], self.mask_ratio, tokens.device)

        visible_tokens = gather_tokens(tokens, keep_idx)
        visible_z = self.encoder(visible_tokens, positions=keep_idx)
        visible_z = self.enc_to_dec(visible_z)

        b, _, d = visible_z.shape
        mask_tokens = self.mask_token.expand(b, mask_idx.shape[1], d)
        decoder_input = torch.cat([visible_z, mask_tokens], dim=1)
        positions = torch.cat([keep_idx, mask_idx], dim=1)
        decoder_input = self.decoder_pos(decoder_input, positions=positions)
        decoded = self.decoder(decoder_input)
        pred_patches = self.reconstruction_head(decoded[:, -mask_idx.shape[1] :])
        target_patches = gather_tokens(raw_patches, mask_idx)

        loss = F.mse_loss(pred_patches, target_patches)
        return loss, loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, recon_loss = self.module_step(batch)
        self.log_dict({"train/loss": loss, "train/recon_loss": recon_loss})
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, recon_loss = self.module_step(batch)
        self.log_dict({"val/loss": loss, "val/recon_loss": recon_loss})

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, recon_loss = self.module_step(batch)
        self.log_dict({"test/loss": loss, "test/recon_loss": recon_loss})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
