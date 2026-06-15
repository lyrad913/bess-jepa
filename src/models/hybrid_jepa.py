import torch
import torch.nn.functional as F

from jepa import JEPA
from ts_jepa import MaskedEmbeddingPredictor, gather_tokens, random_mask_indices


class HybridJEPA(JEPA):
    """
    JEPA variant with two objectives.

    gap == 0:
        masked in-window representation prediction, TS-JEPA style.

    gap >= 1:
        future-window representation prediction, standard JEPA style.
    """

    def __init__(
        self,
        *args,
        mask_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.masked_predictor = MaskedEmbeddingPredictor(
            embed_dim=self.hparams.embed_dim,
            nhead=self.hparams.pred_nhead,
            num_layers=self.hparams.pred_layers,
        )

    def module_step(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        sigreg_loss = self.sigreg(x_emb.transpose(0, 1))
        if future_tgt_emb is not None:
            sigreg_loss = 0.5 * (sigreg_loss + self.sigreg(future_tgt_emb.transpose(0, 1)))

        loss = pred_loss + self.sigreg_lambda * sigreg_loss
        return loss, pred_loss, sigreg_loss, x_emb, masked_pred_loss, future_pred_loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, pred_loss, sigreg_loss, _, masked_pred_loss, future_pred_loss = self.module_step(batch)
        self.log_dict({
            "train/loss": loss,
            "train/pred_loss": pred_loss,
            "train/masked_pred_loss": masked_pred_loss,
            "train/future_pred_loss": future_pred_loss,
            "train/sigreg_loss": sigreg_loss,
        })
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, z, masked_pred_loss, future_pred_loss = self.module_step(batch)
        self.log_dict({
            "val/loss": loss,
            "val/pred_loss": pred_loss,
            "val/masked_pred_loss": masked_pred_loss,
            "val/future_pred_loss": future_pred_loss,
            "val/sigreg_loss": sigreg_loss,
        })
        self._log_collapse_metrics(z.mean(dim=1), prefix="emb")
        self._log_collapse_metrics(z.flatten(0, 1), prefix="emb_patch")
        self._val_z.append(z.detach().cpu())

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, pred_loss, sigreg_loss, _, masked_pred_loss, future_pred_loss = self.module_step(batch)
        self.log_dict({
            "test/loss": loss,
            "test/pred_loss": pred_loss,
            "test/masked_pred_loss": masked_pred_loss,
            "test/future_pred_loss": future_pred_loss,
            "test/sigreg_loss": sigreg_loss,
        })
