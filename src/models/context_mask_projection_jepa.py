import torch
import torch.nn.functional as F

from context_mask_jepa import ContextMaskJEPA
from projection_jepa import ProjectionHead


class ContextMaskProjectionJEPA(ContextMaskJEPA):
    """
    Context-masked JEPA with SIGReg applied in projection space.

    Prediction stays in encoder space:
        SmoothL1(Predictor(ctx_emb, gap), tgt_emb)

    SIGReg is applied only after the projection head:
        SIGReg(projector([ctx_emb, tgt_emb]))
    """

    def __init__(
        self,
        *args,
        projection_hidden_scale: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.projection_hidden_scale = int(projection_hidden_scale)
        self.hparams.projection_hidden_scale = self.projection_hidden_scale
        self.projector = ProjectionHead(self.hparams.embed_dim, hidden_scale=self.projection_hidden_scale)
        self._val_proj: list[torch.Tensor] = []

    def module_step(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        z = torch.cat([ctx_emb, tgt_emb], dim=1)
        z_proj = self.projector(z)

        pred_loss = F.smooth_l1_loss(pred_emb, tgt_emb)
        sigreg_loss = self.sigreg(z_proj.transpose(0, 1))
        loss = pred_loss + self.sigreg_lambda * sigreg_loss

        return loss, pred_loss, sigreg_loss, z, z_proj

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, pred_loss, sigreg_loss, _, _ = self.module_step(batch)
        self.log_dict({"train/loss": loss, "train/pred_loss": pred_loss, "train/sigreg_loss": sigreg_loss})
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, z, z_proj = self.module_step(batch)
        self.log_dict({"val/loss": loss, "val/pred_loss": pred_loss, "val/sigreg_loss": sigreg_loss})
        self._log_collapse_metrics(z.mean(dim=1), prefix="emb")
        self._log_collapse_metrics(z.flatten(0, 1), prefix="emb_patch")
        self._log_collapse_metrics(z_proj.mean(dim=1), prefix="proj")
        self._log_collapse_metrics(z_proj.flatten(0, 1), prefix="proj_patch")
        self._val_z.append(z.detach().cpu())
        self._val_proj.append(z_proj.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        z = torch.cat(self._val_z, dim=0)
        z_proj = torch.cat(self._val_proj, dim=0)
        self._val_z.clear()
        self._val_proj.clear()

        self._log_epoch_diagnostics(
            z.mean(dim=1), title="Sequence", series_suffix="", section="Embeddings", metric_prefix="emb"
        )
        self._log_epoch_diagnostics(
            z.flatten(0, 1), title="Patch", series_suffix=" Patch", section="Embeddings", metric_prefix="emb_patch"
        )
        self._log_epoch_diagnostics(
            z_proj.mean(dim=1), title="Sequence", series_suffix="", section="Projections", metric_prefix="proj"
        )
        self._log_epoch_diagnostics(
            z_proj.flatten(0, 1), title="Patch", series_suffix=" Patch", section="Projections", metric_prefix="proj_patch"
        )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, _, _ = self.module_step(batch)
        self.log_dict({"test/loss": loss, "test/pred_loss": pred_loss, "test/sigreg_loss": sigreg_loss})
