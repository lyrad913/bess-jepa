import torch
import torch.nn.functional as F

from hybrid_jepa import HybridJEPA
from projection_jepa import ProjectionHead
from ts_jepa import gather_tokens, random_mask_indices


class HybridProjectionJEPA(HybridJEPA):
    """
    HybridJEPA with SIGReg applied in a projection space.

    Prediction losses stay in encoder space:
      - gap == 0: masked in-window prediction
      - gap >= 1: future-window prediction

    SIGReg is applied to projected encoder embeddings only.
    """

    def __init__(
        self,
        *args,
        projection_hidden_scale: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.projection_hidden_scale = projection_hidden_scale
        self.projector = ProjectionHead(self.hparams.embed_dim, hidden_scale=projection_hidden_scale)
        self._val_proj: list[torch.Tensor] = []

    def module_step(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            x = batch["x"]                           # (B, C, T)
            y = batch["y"]                           # (B, C, T)
            gap = batch["gap"]                       # (B,)
        else:
            raise Exception

        tokens = self.tokenizer(x)
        x_emb = self.encoder(tokens)

        gap_long = gap.to(device=x.device, dtype=torch.long).view(-1)
        zero_mask = gap_long == 0
        future_mask = gap_long >= 1

        total_weight = x.new_tensor(0.0)
        pred_loss_sum = x.new_tensor(0.0)
        masked_pred_loss = x.new_zeros(())
        future_pred_loss = x.new_zeros(())

        if zero_mask.any():
            zero_idx = zero_mask.nonzero(as_tuple=True)[0]
            zero_tokens = tokens[zero_idx]
            keep_idx, mask_idx = random_mask_indices(
                zero_tokens.shape[0],
                zero_tokens.shape[1],
                self.mask_ratio,
                zero_tokens.device,
            )
            visible_tokens = gather_tokens(zero_tokens, keep_idx)
            ctx_z = self.encoder(visible_tokens, positions=keep_idx)
            pred_z = self.masked_predictor(ctx_z, keep_idx, mask_idx)
            target_z = gather_tokens(x_emb[zero_idx], mask_idx)
            masked_pred_loss = F.smooth_l1_loss(pred_z, target_z)
            weight = x.new_tensor(float(zero_idx.numel()))
            pred_loss_sum = pred_loss_sum + masked_pred_loss * weight
            total_weight = total_weight + weight

        future_tgt_emb = None
        if future_mask.any():
            future_idx = future_mask.nonzero(as_tuple=True)[0]
            ctx_emb = x_emb[future_idx]
            tgt_tokens = self.tokenizer(y[future_idx])
            future_tgt_emb = self.encoder(tgt_tokens)
            pred_emb = self.predictor(ctx_emb, gap[future_idx])
            future_pred_loss = F.smooth_l1_loss(pred_emb, future_tgt_emb)
            weight = x.new_tensor(float(future_idx.numel()))
            pred_loss_sum = pred_loss_sum + future_pred_loss * weight
            total_weight = total_weight + weight

        pred_loss = pred_loss_sum / total_weight.clamp_min(1.0)

        z_proj = self.projector(x_emb)
        sigreg_loss = self.sigreg(z_proj.transpose(0, 1))
        if future_tgt_emb is not None:
            future_tgt_proj = self.projector(future_tgt_emb)
            sigreg_loss = 0.5 * (sigreg_loss + self.sigreg(future_tgt_proj.transpose(0, 1)))

        loss = pred_loss + self.sigreg_lambda * sigreg_loss
        return loss, pred_loss, sigreg_loss, x_emb, z_proj, masked_pred_loss, future_pred_loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, pred_loss, sigreg_loss, _, _, masked_pred_loss, future_pred_loss = self.module_step(batch)
        self.log_dict({
            "train/loss": loss,
            "train/pred_loss": pred_loss,
            "train/masked_pred_loss": masked_pred_loss,
            "train/future_pred_loss": future_pred_loss,
            "train/sigreg_loss": sigreg_loss,
        })
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, z, z_proj, masked_pred_loss, future_pred_loss = self.module_step(batch)
        self.log_dict({
            "val/loss": loss,
            "val/pred_loss": pred_loss,
            "val/masked_pred_loss": masked_pred_loss,
            "val/future_pred_loss": future_pred_loss,
            "val/sigreg_loss": sigreg_loss,
        })
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
        loss, pred_loss, sigreg_loss, _, _, masked_pred_loss, future_pred_loss = self.module_step(batch)
        self.log_dict({
            "test/loss": loss,
            "test/pred_loss": pred_loss,
            "test/masked_pred_loss": masked_pred_loss,
            "test/future_pred_loss": future_pred_loss,
            "test/sigreg_loss": sigreg_loss,
        })
