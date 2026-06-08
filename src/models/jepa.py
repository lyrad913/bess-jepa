from einops import rearrange
import torch
import torch.nn.functional as F
import lightning as L

from sub_models import Tokenizer, Encoder, Predictor, SIGReg

# ---------------------------------------------------------------------------
# Optional visualisation deps – imported lazily inside _log_epoch_diagnostics
# so jepa.py can be imported even without matplotlib / clearml installed.
# ---------------------------------------------------------------------------


class JEPA(L.LightningModule):
    """
    TS-JEPA + LeJEPA for multivariate time series.

    Flow:
        ctx_tokens = Tokenizer(x)                     # (B, N, D)
        tgt_tokens = Tokenizer(y)                     # (B, N, D)
        ctx_emb    = Encoder(ctx_tokens)              # (B, N, D)
        tgt_emb    = Encoder(tgt_tokens)              # (B, N, D)
        pred       = Predictor(ctx_emb, gap)          # (B, N, D)
        loss       = MSE(pred, tgt_emb) + SIGReg(ctx_emb) + SIGReg(tgt_emb)
    """

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
        sigreg_lambda: float = 0.05,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sigreg_lambda = sigreg_lambda
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.tokenizer = Tokenizer(seq_len, patch_size, strides, num_channels, embed_dim)
        self.encoder = Encoder(embed_dim, enc_nhead, enc_layers)
        self.predictor = Predictor(embed_dim, pred_nhead, pred_layers)
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

        self._val_z: list[torch.Tensor] = []

    def module_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            x = batch["x"]                           # (B, C, T)
            y = batch["y"]                           # (B, C, T)
            gap = batch["gap"]                       # (B,)
        else:
            raise Exception

        emb_tokens = self.tokenizer(x)                   # (B, N, D)
        tgt_tokens = self.tokenizer(y)                   # (B, N, D)

        ctx_emb = self.encoder(emb_tokens)
        tgt_emb = self.encoder(tgt_tokens)
        pred_emb = self.predictor(ctx_emb, gap)

        # Loss
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        ce = rearrange(ctx_emb, 'b n d -> n b d')
        te = rearrange(tgt_emb, 'b n d -> n b d')
        sigreg_loss = 0.5 * self.sigreg(ce) + 0.5 * self.sigreg(te)
        loss = pred_loss + self.sigreg_lambda * sigreg_loss

        return loss, pred_loss, sigreg_loss, torch.cat([ctx_emb, tgt_emb])

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, pred_loss, sigreg_loss, _ = self.module_step(batch)
        self.log_dict({"train/loss": loss, "train/pred_loss": pred_loss, "train/sigreg_loss": sigreg_loss})
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, z = self.module_step(batch)
        self.log_dict({"val/loss": loss, "val/pred_loss": pred_loss, "val/sigreg_loss": sigreg_loss})
        self._log_collapse_metrics(z.mean(dim=1))
        self._val_z.append(z.mean(dim=1).detach().cpu())

    def on_validation_epoch_end(self) -> None:
        z = torch.cat(self._val_z, dim=0)
        self._val_z.clear()
        self._log_epoch_diagnostics(z)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, _ = self.module_step(batch)
        self.log_dict({"test/loss": loss, "test/pred_loss": pred_loss, "test/sigreg_loss": sigreg_loss})

    def _log_collapse_metrics(self, z: torch.Tensor) -> None:
        """z: (B, D) – view embeddings averaged over patches."""
        # std per dim across the batch → mean; approaches 0 on collapse
        emb_std = z.std(dim=0).mean()

        # mean pairwise cosine similarity; approaches 1 on collapse
        z_norm = F.normalize(z, dim=-1)
        B = z.shape[0]
        if B > 1:
            cos_mat = z_norm @ z_norm.T                      # (B, B)
            n_pairs = B * (B - 1) / 2
            cos_sim = cos_mat.triu(diagonal=1).sum() / n_pairs
        else:
            cos_sim = z.new_zeros(())

        self.log_dict({"emb/std": emb_std, "emb/cos_sim": cos_sim})

    def _log_epoch_diagnostics(self, z: torch.Tensor) -> None:
        """Epoch-level plots: PCA scatter + per-dim std. Logged via ClearML."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z = z[:2000]                                         # cap for speed
        z_c = z - z.mean(0)

        # Effective rank via SVD (entropy of normalised singular values)
        _, s, Vt = torch.linalg.svd(z_c, full_matrices=False)
        p = s / (s.sum() + 1e-8)
        eff_rank = torch.exp(-(p * (p + 1e-8).log()).sum()).item()
        self.log("emb/effective_rank", eff_rank)

        try:
            from clearml import Logger as ClearMLLogger
            cl_logger = ClearMLLogger.current_logger()
        except Exception:
            return

        step = self.current_epoch

        # 1. PCA scatter – shows structure of learned representations
        var_ratio = ((s[:2] ** 2) / ((s ** 2).sum() + 1e-8)).tolist()
        coords = (z_c @ Vt[:2].T).cpu().numpy()             # (N, 2)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.3)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})")
        ax.set_title(f"View embedding PCA  epoch={step}  eff_rank={eff_rank:.1f}")
        cl_logger.report_matplotlib_figure("Embeddings", "PCA", fig, step)
        plt.close(fig)

        # 2. Per-dim std sorted descending – dead dims show up as near-zero tail
        dim_std = z.std(dim=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(dim_std)), sorted(dim_std, reverse=True), width=1.0)
        ax.axhline(0.1, color="r", linestyle="--", label="collapse threshold (0.1)")
        dead = int((dim_std < 0.1).sum())
        ax.set_xlabel("Embedding dimension (sorted by std)")
        ax.set_ylabel("Std across samples")
        ax.set_title(f"Per-dim std  epoch={step}  dead_dims={dead}/{len(dim_std)}")
        ax.legend()
        cl_logger.report_matplotlib_figure("Embeddings", "Dim Std", fig, step)
        plt.close(fig)

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
            optimizer, T_max=max_epochs - self.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
