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
        loss       = SmoothL1(pred, tgt_emb) + SIGReg(ctx_emb, tgt_emb)
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

        emb_tokens = self.tokenizer(x)                   # (B, N, D)
        tgt_tokens = self.tokenizer(y)                   # (B, N, D)

        ctx_emb = self.encoder(emb_tokens)
        tgt_emb = self.encoder(tgt_tokens)
        pred_emb = self.predictor(ctx_emb, gap)

        # Loss
        pred_loss = F.smooth_l1_loss(pred_emb, tgt_emb)
        proj = torch.cat([ctx_emb, tgt_emb], dim=1).transpose(0, 1)
        sigreg_loss = self.sigreg(proj)
        loss = pred_loss + self.sigreg_lambda * sigreg_loss

        return loss, pred_loss, sigreg_loss, torch.cat([ctx_emb, tgt_emb], dim=1)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, pred_loss, sigreg_loss, _ = self.module_step(batch)
        self.log_dict({"train/loss": loss, "train/pred_loss": pred_loss, "train/sigreg_loss": sigreg_loss})
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, z = self.module_step(batch)
        self.log_dict({"val/loss": loss, "val/pred_loss": pred_loss, "val/sigreg_loss": sigreg_loss})
        self._log_collapse_metrics(z.mean(dim=1), prefix="emb")
        self._log_collapse_metrics(z.flatten(0, 1), prefix="emb_patch")
        self._val_z.append(z.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        z = torch.cat(self._val_z, dim=0)
        self._val_z.clear()
        self._log_epoch_diagnostics(
            z.mean(dim=1), title="Sequence", series_suffix="", section="Embeddings", metric_prefix="emb"
        )
        self._log_epoch_diagnostics(
            z.flatten(0, 1), title="Patch", series_suffix=" Patch", section="Embeddings", metric_prefix="emb_patch"
        )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, _ = self.module_step(batch)
        self.log_dict({"test/loss": loss, "test/pred_loss": pred_loss, "test/sigreg_loss": sigreg_loss})

    def _log_collapse_metrics(self, z: torch.Tensor, prefix: str) -> None:
        """z: (samples, D)."""
        # std per dim across the batch → mean; approaches 0 on collapse
        emb_std = z.std(dim=0).mean()

        # mean pairwise cosine similarity; approaches 1 on collapse
        z_cos = z[:512]
        z_norm = F.normalize(z_cos, dim=-1)
        B = z_cos.shape[0]
        if B > 1:
            cos_mat = z_norm @ z_norm.T                      # (B, B)
            n_pairs = B * (B - 1) / 2
            cos_sim = cos_mat.triu(diagonal=1).sum() / n_pairs
        else:
            cos_sim = z.new_zeros(())

        self.log_dict({f"{prefix}/std": emb_std, f"{prefix}/cos_sim": cos_sim})

    def _log_epoch_diagnostics(
        self,
        z: torch.Tensor,
        title: str,
        series_suffix: str,
        section: str,
        metric_prefix: str,
    ) -> None:
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
        self.log(f"{metric_prefix}/effective_rank", eff_rank)

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
        ax.set_title(f"{title} embedding PCA  epoch={step}  eff_rank={eff_rank:.1f}")
        cl_logger.report_matplotlib_figure(
            section,
            f"PCA{series_suffix}",
            fig,
            step,
            report_interactive=False,
        )
        plt.close(fig)

        # 2. Per-dim std sorted descending – dead dims show up as near-zero tail
        dim_std = z.std(dim=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(dim_std)), sorted(dim_std, reverse=True), width=1.0)
        ax.axhline(0.1, color="r", linestyle="--", label="collapse threshold (0.1)")
        dead = int((dim_std < 0.1).sum())
        ax.set_xlabel("Embedding dimension (sorted by std)")
        ax.set_ylabel("Std across samples")
        ax.set_title(f"{title} per-dim std  epoch={step}  dead_dims={dead}/{len(dim_std)}")
        ax.legend()
        cl_logger.report_matplotlib_figure(section, f"Dim Std{series_suffix}", fig, step)
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
