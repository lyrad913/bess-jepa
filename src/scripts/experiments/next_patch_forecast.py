"""
Frozen JEPA encoder + linear readout for next-patch forecasting.

This follows the TS-JEPA downstream evaluation style:
  context window -> tokenizer/encoder -> sum pooled tokens -> Linear -> next raw patch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR / "models"))
sys.path.insert(0, str(SRC_DIR / "data"))

from forecast_loader import FEATURES, ForecastLoader
from jepa import JEPA
from sub_models import Encoder, Tokenizer


class NextPatchForecastProbe(L.LightningModule):
    """Frozen JEPA representation with a single linear next-patch readout."""

    def __init__(
        self,
        tokenizer: torch.nn.Module,
        encoder: torch.nn.Module,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        train_encoder: bool,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "encoder"])

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.train_encoder = train_encoder
        for p in self.tokenizer.parameters():
            p.requires_grad = train_encoder
        for p in self.encoder.parameters():
            p.requires_grad = train_encoder

        self.patch_size = int(patch_size)
        self.n_features = int(num_channels)
        self.head = nn.Linear(int(embed_dim), self.n_features * self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.train_encoder:
            tokens = self.tokenizer(x)
            z = self.encoder(tokens)
        else:
            self.tokenizer.eval()
            self.encoder.eval()
            with torch.no_grad():
                tokens = self.tokenizer(x)
                z = self.encoder(tokens)
        pooled = z.sum(dim=1)
        y_hat = self.head(pooled)
        return y_hat.view(x.shape[0], self.n_features, self.patch_size)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        y_hat = self(batch["x"])
        y = batch["y"][:, :, : self.patch_size]
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)

        logs = {
            f"{stage}/mse": loss,
            f"{stage}/mae": mae,
        }
        for i, name in enumerate(FEATURES):
            logs[f"{stage}/{name}_mse"] = F.mse_loss(y_hat[:, i], y[:, i])
            logs[f"{stage}/{name}_mae"] = F.l1_loss(y_hat[:, i], y[:, i])

        self.log_dict(logs, prog_bar=stage in {"train", "val"}, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self):
        params = list(self.head.parameters())
        if self.train_encoder:
            params = list(self.tokenizer.parameters()) + list(self.encoder.parameters()) + params
        return torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def train(
    cfg: DictConfig,
    variant: str,
    warm_start_ckpt: str | None = None,
) -> tuple[NextPatchForecastProbe, L.Trainer, ForecastLoader, str, int | None, list[dict[str, float]]]:
    # Variant별 tokenizer/encoder와 next-patch forecasting datamodule 준비
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa = JEPA.load_from_checkpoint(cfg.checkpoint, map_location=device)
    jepa.eval()
    hp = jepa.hparams

    seq_len = cfg.data.seq_len or int(hp.seq_len)
    if seq_len != int(hp.seq_len):
        raise ValueError(
            f"Next-patch forecast seq_len={seq_len} must match checkpoint seq_len={int(hp.seq_len)}"
        )

    patch_size = int(hp.patch_size)
    configured_rollout_steps = cfg.data.rollout_steps
    configured_rollout_steps = None if configured_rollout_steps is None else int(configured_rollout_steps)
    dm = ForecastLoader(
        data_dir=cfg.data.data_dir,
        datasets=cfg.data.datasets,
        seq_len=seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        stride=cfg.data.stride,
        rollout_steps=configured_rollout_steps,
        rollout_step_size=patch_size,
        processed_dir_name=cfg.data.processed_dir_name,
    )
    dm.setup()

    pretrained = variant.startswith("pretrained")
    train_encoder = variant in {"random_supervised", "pretrained_finetune"}
    if pretrained:
        tokenizer = jepa.tokenizer
        encoder = jepa.encoder
    else:
        tokenizer = Tokenizer(
            seq_len=int(hp.seq_len),
            patch_len=int(hp.patch_size),
            strides=int(hp.strides),
            n_features=int(hp.num_channels),
            embed_dim=int(hp.embed_dim),
        )
        encoder = Encoder(
            embed_dim=int(hp.embed_dim),
            nhead=int(hp.enc_nhead),
            num_layers=int(hp.enc_layers),
        )

    # Variant 조건에 따라 encoder를 freeze하거나 supervised로 같이 학습
    lr = 1e-4 if variant == "pretrained_finetune" else cfg.probe.lr
    model = NextPatchForecastProbe(
        tokenizer=tokenizer,
        encoder=encoder,
        patch_size=int(hp.patch_size),
        num_channels=int(hp.num_channels),
        embed_dim=int(hp.embed_dim),
        train_encoder=train_encoder,
        lr=lr,
        weight_decay=cfg.probe.weight_decay,
    )
    if variant == "pretrained_finetune":
        if warm_start_ckpt is None:
            raise RuntimeError("pretrained_finetune requires pretrained_frozen best checkpoint")
        checkpoint = torch.load(warm_start_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"[{variant}] warm-start from pretrained_frozen checkpoint: {warm_start_ckpt}")
    encoder_params = list(model.tokenizer.parameters()) + list(model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in encoder_params if p.requires_grad)
    probe_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(
        f"[{variant}] lr={lr} train_encoder={train_encoder} "
        f"encoder_trainable={encoder_trainable} probe_trainable={probe_trainable}"
    )
    if train_encoder and encoder_trainable == 0:
        raise RuntimeError(f"{variant}: encoder should be trainable but no encoder parameters require grad")
    if not train_encoder and encoder_trainable != 0:
        raise RuntimeError(f"{variant}: encoder should be frozen but some encoder parameters require grad")
    frozen_before = None
    if not train_encoder:
        frozen_before = [
            p.detach().cpu().clone()
            for p in encoder_params
        ]
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[
            EarlyStopping(monitor="val/mse", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(monitor="val/mse", mode="min", save_top_k=1, filename=f"next-patch-forecast-{variant}-best"),
        ],
    )
    trainer.fit(model, dm)
    if frozen_before is not None:
        for before, param in zip(frozen_before, encoder_params):
            if not torch.equal(before, param.detach().cpu()):
                raise RuntimeError(f"{variant}: frozen encoder/tokenizer parameters changed during training")
    test_results = trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path="best")

    # 이후 bench는 best checkpoint 기준으로 수행
    if trainer.checkpoint_callback.best_model_path:
        model = NextPatchForecastProbe.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            tokenizer=tokenizer,
            encoder=encoder,
            map_location=device,
        )
    model.to(device)
    return model, trainer, dm, device, configured_rollout_steps, test_results


def do_bench(
    model: NextPatchForecastProbe,
    dm: ForecastLoader,
    device: str,
    configured_rollout_steps: int | None,
    test_results: list[dict[str, float]],
    cfg: DictConfig,
    variant: str,
    task,
    report_dir: Path,
) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One-step next-patch 예시 수집
    one_step_preds, one_step_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dm.test_dataloader()):
            if batch_idx >= cfg.report.plot_batches:
                break
            x = batch["x"].to(device)
            y = batch["y"][:, :, : model.patch_size]
            y_hat = model(x)
            try:
                one_step_preds.append(dm.inverse_transform(y_hat.cpu()).numpy())
            except Exception:
                one_step_preds.append(y_hat.cpu().numpy())
            try:
                one_step_targets.append(dm.inverse_transform(y.cpu()).numpy())
            except Exception:
                one_step_targets.append(y.cpu().numpy())

    pred = np.concatenate(one_step_preds, axis=0)
    target = np.concatenate(one_step_targets, axis=0)

    # One-step next-patch 예시 그림 저장
    n_rows = min(3, pred.shape[0])
    fig, axes = plt.subplots(n_rows, len(FEATURES), figsize=(12, 2.8 * n_rows), squeeze=False)
    t = np.arange(pred.shape[-1])
    for row in range(n_rows):
        for col, name in enumerate(FEATURES):
            ax = axes[row, col]
            ax.plot(t, target[row, col], label="target", linewidth=1.5)
            ax.plot(t, pred[row, col], label="pred", linewidth=1.2, alpha=0.8)
            ax.set_title(f"sample {row} - {name}")
            if row == 0 and col == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "next_patch_examples.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Next Patch Forecast/{variant}/examples",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # Autoregressive next-patch rollout 평가
    mse_sum: torch.Tensor | None = None
    mae_sum: torch.Tensor | None = None
    sample_mse_sum: torch.Tensor | None = None
    sample_mse_sq_sum: torch.Tensor | None = None
    sample_channel_mse_sum: torch.Tensor | None = None
    sample_channel_mse_sq_sum: torch.Tensor | None = None
    count: torch.Tensor | None = None
    plot_preds, plot_targets = [], []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dm.rollout_dataloader()):
            x = batch["x"].to(device)
            target = batch["rollout_y"].to(device)
            b, available_steps, c, patch_size = target.shape
            if patch_size != model.patch_size:
                raise ValueError(f"Expected rollout patch size {model.patch_size}, got {patch_size}")
            eval_steps = configured_rollout_steps or available_steps
            target = target[:, :eval_steps]

            preds = []
            current = x
            for _ in range(eval_steps):
                y_hat = model(current)
                preds.append(y_hat)
                current = torch.cat([current[:, :, model.patch_size :], y_hat], dim=-1)
            pred = torch.stack(preds, dim=1)
            err = pred - target
            batch_mse = err.square().mean(dim=(0, 3))
            batch_mae = err.abs().mean(dim=(0, 3))
            sample_mse = err.square().mean(dim=(2, 3))
            sample_channel_mse = err.square().mean(dim=3)
            sample_count = sample_mse.shape[0]
            if mse_sum is None:
                mse_sum = torch.zeros(eval_steps, len(FEATURES), device=device)
                mae_sum = torch.zeros(eval_steps, len(FEATURES), device=device)
                sample_mse_sum = torch.zeros(eval_steps, device=device)
                sample_mse_sq_sum = torch.zeros(eval_steps, device=device)
                sample_channel_mse_sum = torch.zeros(eval_steps, len(FEATURES), device=device)
                sample_channel_mse_sq_sum = torch.zeros(eval_steps, len(FEATURES), device=device)
                count = torch.zeros(eval_steps, 1, device=device)
            elif eval_steps > mse_sum.shape[0]:
                extra = eval_steps - mse_sum.shape[0]
                mse_sum = torch.cat([mse_sum, torch.zeros(extra, len(FEATURES), device=device)], dim=0)
                mae_sum = torch.cat([mae_sum, torch.zeros(extra, len(FEATURES), device=device)], dim=0)
                sample_mse_sum = torch.cat([sample_mse_sum, torch.zeros(extra, device=device)], dim=0)
                sample_mse_sq_sum = torch.cat([sample_mse_sq_sum, torch.zeros(extra, device=device)], dim=0)
                sample_channel_mse_sum = torch.cat(
                    [sample_channel_mse_sum, torch.zeros(extra, len(FEATURES), device=device)], dim=0
                )
                sample_channel_mse_sq_sum = torch.cat(
                    [sample_channel_mse_sq_sum, torch.zeros(extra, len(FEATURES), device=device)], dim=0
                )
                count = torch.cat([count, torch.zeros(extra, 1, device=device)], dim=0)
            mse_sum[:eval_steps] += batch_mse * sample_count
            mae_sum[:eval_steps] += batch_mae * sample_count
            sample_mse_sum[:eval_steps] += sample_mse.sum(dim=0)
            sample_mse_sq_sum[:eval_steps] += sample_mse.square().sum(dim=0)
            sample_channel_mse_sum[:eval_steps] += sample_channel_mse.sum(dim=0)
            sample_channel_mse_sq_sum[:eval_steps] += sample_channel_mse.square().sum(dim=0)
            count[:eval_steps] += sample_count

            if batch_idx < cfg.report.plot_batches and not plot_preds:
                flat_pred = pred.reshape(b * eval_steps, len(FEATURES), -1)
                flat_target = target.reshape(b * eval_steps, len(FEATURES), -1)
                try:
                    plot_preds.append(
                        dm.inverse_transform(flat_pred.cpu()).numpy().reshape(b, eval_steps, len(FEATURES), -1)
                    )
                except Exception:
                    plot_preds.append(flat_pred.cpu().numpy().reshape(b, eval_steps, len(FEATURES), -1))
                try:
                    plot_targets.append(
                        dm.inverse_transform(flat_target.cpu()).numpy().reshape(b, eval_steps, len(FEATURES), -1)
                    )
                except Exception:
                    plot_targets.append(flat_target.cpu().numpy().reshape(b, eval_steps, len(FEATURES), -1))

    if (
        mse_sum is None
        or mae_sum is None
        or sample_mse_sum is None
        or sample_mse_sq_sum is None
        or sample_channel_mse_sum is None
        or sample_channel_mse_sq_sum is None
        or count is None
    ):
        raise RuntimeError("No rollout batches were available for evaluation")

    valid = count.squeeze(-1) > 0
    mse = (mse_sum[valid] / count[valid]).cpu().numpy()
    mae = (mae_sum[valid] / count[valid]).cpu().numpy()
    sample_count = count.squeeze(-1)[valid]
    step_mean = sample_mse_sum[valid] / sample_count
    step_var = sample_mse_sq_sum[valid] / sample_count - step_mean.square()
    step_std = torch.sqrt(torch.clamp(step_var, min=0.0))
    channel_mean = sample_channel_mse_sum[valid] / sample_count[:, None]
    channel_var = sample_channel_mse_sq_sum[valid] / sample_count[:, None] - channel_mean.square()
    channel_std = torch.sqrt(torch.clamp(channel_var, min=0.0))
    rollout_stats = {
        "mean": step_mean.cpu().numpy().tolist(),
        "std": step_std.cpu().numpy().tolist(),
        "channel_mean": {
            name: channel_mean[:, i].cpu().numpy().tolist()
            for i, name in enumerate(FEATURES)
        },
        "channel_std": {
            name: channel_std[:, i].cpu().numpy().tolist()
            for i, name in enumerate(FEATURES)
        },
        "count": sample_count.cpu().numpy().astype(int).tolist(),
    }
    actual_steps = int(mse.shape[0])
    cumulative_mse = np.cumsum(mse, axis=0)
    metrics = {}
    for step in range(actual_steps):
        metrics[f"rollout/step_{step + 1}_mse"] = float(mse[step].mean())
        metrics[f"rollout/step_{step + 1}_mae"] = float(mae[step].mean())
        metrics[f"rollout/cumulative_step_{step + 1}_mse"] = float(cumulative_mse[step].mean())
        for i, name in enumerate(FEATURES):
            metrics[f"rollout/step_{step + 1}_{name}_mse"] = float(mse[step, i])
            metrics[f"rollout/step_{step + 1}_{name}_mae"] = float(mae[step, i])
            metrics[f"rollout/cumulative_step_{step + 1}_{name}_mse"] = float(cumulative_mse[step, i])

    if task is not None:
        logger = task.get_logger()
        for key, value in metrics.items():
            title, series = key.split("/", 1)
            logger.report_scalar(f"{variant}/{title}", series, value=value, iteration=0)
    (report_dir / "next_patch_rollout_metrics.json").write_text(json.dumps(metrics, indent=2))
    (report_dir / "next_patch_rollout_step_mse_stats.json").write_text(json.dumps(rollout_stats, indent=2))

    # Rollout sample 그림 저장
    rollout_pred = np.concatenate(plot_preds, axis=0)
    rollout_target = np.concatenate(plot_targets, axis=0)
    steps = rollout_pred.shape[1]
    sample = 0
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(12, 7), squeeze=False)
    t = np.arange(steps * rollout_pred.shape[-1])
    for col, name in enumerate(FEATURES):
        ax = axes[col, 0]
        y_true = rollout_target[sample, :, col].reshape(-1)
        y_pred = rollout_pred[sample, :, col].reshape(-1)
        ax.plot(t, y_true, label="target", linewidth=1.5)
        ax.plot(t, y_pred, label="rollout", linewidth=1.2, alpha=0.8)
        for step in range(1, steps):
            ax.axvline(step * rollout_pred.shape[-1], color="0.85", linewidth=0.8)
        ax.set_title(f"rollout - {name}")
        if col == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "next_patch_rollout.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Next Patch Forecast/{variant}/rollout_example",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # Cumulative MSE 그림 저장
    x = np.arange(1, actual_steps + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    overall = [metrics[f"rollout/cumulative_step_{i}_mse"] for i in x]
    ax.plot(x, overall, label="overall", color="black", linewidth=2.0)

    colors = {
        "voltage": "tab:blue",
        "current": "tab:orange",
        "temperature": "tab:green",
    }
    for name in FEATURES:
        values = [metrics[f"rollout/cumulative_step_{i}_{name}_mse"] for i in x]
        ax.plot(x, values, label=name, color=colors.get(name), linewidth=1.5, alpha=0.9)

    ax.set_title("Next-Patch Rollout Cumulative MSE")
    ax.set_xlabel("# Patches")
    ax.set_ylabel("Cumulative MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "next_patch_cumulative_mse.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Next Patch Forecast/{variant}/cumulative_mse",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # Step MSE 평균과 표준편차 band 그림 저장
    mean = np.asarray(rollout_stats["mean"], dtype=np.float32)
    std = np.asarray(rollout_stats["std"], dtype=np.float32)
    x = np.arange(1, len(mean) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, mean, label="mean", color="black", linewidth=2.0)
    ax.fill_between(
        x,
        np.maximum(mean - std, 0.0),
        mean + std,
        color="tab:blue",
        alpha=0.18,
        label="+/- 1 std",
    )
    ax.set_title("Next-Patch Rollout Step MSE")
    ax.set_xlabel("# Patches")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    count_ax = ax.twinx()
    count_ax.plot(x, rollout_stats["count"], color="0.45", linestyle="--", linewidth=1.2, label="sample count")
    count_ax.set_ylabel("# Samples")
    count_ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(report_dir / "next_patch_step_mse_band.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Next Patch Forecast/{variant}/step_mse_band",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # 채널별 Step MSE band 그림 저장
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(8, 7), sharex=True, squeeze=False)
    for row, name in enumerate(FEATURES):
        mean = np.asarray(rollout_stats["channel_mean"][name], dtype=np.float32)
        std = np.asarray(rollout_stats["channel_std"][name], dtype=np.float32)
        x = np.arange(1, len(mean) + 1)
        ax = axes[row, 0]
        color = colors.get(name, "tab:blue")
        ax.plot(x, mean, label=f"{name} mean", color=color, linewidth=1.8)
        ax.fill_between(
            x,
            np.maximum(mean - std, 0.0),
            mean + std,
            color=color,
            alpha=0.18,
            label="+/- 1 std",
        )
        ax.set_ylabel("MSE")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

    axes[-1, 0].set_xlabel("# Patches")
    fig.suptitle("Next-Patch Rollout Step MSE by Channel", y=0.995)
    fig.tight_layout()
    fig.savefig(report_dir / "next_patch_channel_step_mse_bands.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Next Patch Forecast/{variant}/channel_step_mse_bands",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # Rollout step별 sample count 그림 저장
    count = np.asarray(rollout_stats["count"], dtype=np.int32)
    x = np.arange(1, len(count) + 1)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.step(x, count, where="mid", color="black", linewidth=1.8)
    ax.set_title("Next-Patch Rollout Sample Count by Step")
    ax.set_xlabel("# Patches")
    ax.set_ylabel("# Samples")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "next_patch_sample_count.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Next Patch Forecast/{variant}/sample_count",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # 최종 scalar summary 정리 및 저장
    summary = {}
    if test_results:
        for key, value in test_results[0].items():
            summary[f"next_patch/{key}"] = float(value)

    step_mse = [metrics[f"rollout/step_{i}_mse"] for i in range(1, actual_steps + 1)]
    step_mae = [metrics[f"rollout/step_{i}_mae"] for i in range(1, actual_steps + 1)]
    summary["rollout/mean_mse"] = float(np.mean(step_mse))
    summary["rollout/mean_mae"] = float(np.mean(step_mae))
    summary["rollout/final_step_mse"] = float(step_mse[-1])
    summary["rollout/final_step_mae"] = float(step_mae[-1])
    summary["rollout/final_cumulative_mse"] = float(
        metrics[f"rollout/cumulative_step_{actual_steps}_mse"]
    )

    for name in FEATURES:
        ch_mse = [metrics[f"rollout/step_{i}_{name}_mse"] for i in range(1, actual_steps + 1)]
        ch_mae = [metrics[f"rollout/step_{i}_{name}_mae"] for i in range(1, actual_steps + 1)]
        summary[f"rollout/{name}_mean_mse"] = float(np.mean(ch_mse))
        summary[f"rollout/{name}_mean_mae"] = float(np.mean(ch_mae))
        summary[f"rollout/{name}_final_step_mse"] = float(ch_mse[-1])
        summary[f"rollout/{name}_final_step_mae"] = float(ch_mae[-1])
        summary[f"rollout/{name}_final_cumulative_mse"] = float(
            metrics[f"rollout/cumulative_step_{actual_steps}_{name}_mse"]
        )

    if task is not None:
        logger = task.get_logger()
        for key, value in summary.items():
            logger.report_single_value(f"{variant}/{key}", value)
    (report_dir / "next_patch_final_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


@hydra.main(config_path="../../../config", config_name="forecast", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)
    report_dir = Path(cfg.report.dir)
    if report_dir == Path("forecast_reports"):
        report_dir = Path("next_patch_forecast_reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    # ClearML Task init
    task = None
    if cfg.report.use_clearml:
        try:
            from clearml import Task

            task = Task.init(
                project_name="BESS-JEPA",
                task_name="next_patch_forecast",
                reuse_last_task_id=False,
            )
            task.connect(OmegaConf.to_container(cfg, resolve=True))
        except Exception as e:
            print(f"ClearML 사용 불가: {e}")

    # 네 가지 encoder 조건에 대해 같은 downstream 실험 실행
    variants = [
        "pretrained_frozen",
        "random_frozen",
        "random_supervised",
        "pretrained_finetune",
    ]
    comparison: dict[str, dict[str, float]] = {}
    best_paths: dict[str, str] = {}
    for variant in variants:
        print(f"\n=== next-patch forecast variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        warm_start_ckpt = best_paths.get("pretrained_frozen") if variant == "pretrained_finetune" else None
        model, trainer, dm, device, configured_rollout_steps, test_results = train(
            cfg, variant, warm_start_ckpt=warm_start_ckpt
        )
        comparison[variant] = do_bench(
            model,
            dm,
            device,
            configured_rollout_steps,
            test_results,
            cfg,
            variant,
            task,
            variant_dir,
        )
        if trainer.checkpoint_callback.best_model_path:
            best_paths[variant] = trainer.checkpoint_callback.best_model_path

    # Variant별 scalar 비교 표와 그림 저장
    key_order = sorted({key for metrics in comparison.values() for key in metrics})
    comparison_rows = [
        {"variant": variant, **{key: comparison[variant].get(key, float("nan")) for key in key_order}}
        for variant in variants
    ]
    (report_dir / "comparison_summary.json").write_text(json.dumps(comparison_rows, indent=2))
    with (report_dir / "comparison_summary.csv").open("w") as f:
        f.write("variant," + ",".join(key_order) + "\n")
        for row in comparison_rows:
            f.write(str(row["variant"]) + "," + ",".join(str(row[key]) for key in key_order) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_keys = [
        key
        for key in ["next_patch/test/mse", "next_patch/test/mae", "rollout/mean_mse", "rollout/final_cumulative_mse"]
        if key in key_order
    ]
    if plot_keys:
        fig, axes = plt.subplots(len(plot_keys), 1, figsize=(9, 3.2 * len(plot_keys)), squeeze=False)
        x = np.arange(len(variants))
        for row, key in enumerate(plot_keys):
            ax = axes[row, 0]
            values = [comparison[variant].get(key, np.nan) for variant in variants]
            ax.bar(x, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
            ax.set_xticks(x, variants, rotation=20, ha="right")
            ax.set_title(key)
            ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(report_dir / "variant_comparison.png", dpi=150, bbox_inches="tight")
        if task is not None:
            task.get_logger().report_matplotlib_figure("Next Patch Forecast", "variant_comparison", fig, 0)
        plt.close(fig)

    if task is not None:
        logger = task.get_logger()
        for variant, metrics in comparison.items():
            for key, value in metrics.items():
                logger.report_single_value(f"{variant}/{key}", value)

    print(f"next-patch forecast reports saved under {report_dir}")
    print("comparison summary:")
    for variant in variants:
        compact = {
            key: comparison[variant][key]
            for key in ["next_patch/test/mse", "next_patch/test/mae", "rollout/mean_mse", "rollout/final_cumulative_mse"]
            if key in comparison[variant]
        }
        print(f"  {variant}: {compact}")
    for variant, path in best_paths.items():
        print(f"best probe checkpoint [{variant}]: {path}")


if __name__ == "__main__":
    main()
