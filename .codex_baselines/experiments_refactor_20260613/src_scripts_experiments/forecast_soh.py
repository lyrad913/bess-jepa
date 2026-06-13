"""
Frozen JEPA encoder + horizon-conditioned linear head for future SoH prediction.

Input: discharge V/I/T segments from a current cycle plus delta_cycle
Output: target future cycle SoH scalar
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

from forecast_soh_loader import ForecastSohLoader
from jepa import JEPA

SOH_PERCENT_SCALE = 100.0


class ForecastSohProbe(L.LightningModule):
    """Frozen JEPA tokenizer/encoder with a delta-cycle-conditioned SoH head."""

    def __init__(
        self,
        jepa: JEPA,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dt_scale: float = 300.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["jepa"])

        self.tokenizer = jepa.tokenizer
        self.encoder = jepa.encoder
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

        hp = jepa.hparams
        embed_dim = int(hp.embed_dim)
        self.dt_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.head = nn.Linear(embed_dim * 2, 1)

    def encode_segment(self, segment: torch.Tensor) -> torch.Tensor:
        if segment.ndim != 2:
            raise ValueError(f"expected segment shape (C, T), got {tuple(segment.shape)}")
        self.tokenizer.eval()
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.tokenizer(segment.to(self.device).unsqueeze(0))
            z = self.encoder(tokens)
        return z.mean(dim=1).squeeze(0)

    def _dt_features(self, delta_cycle: torch.Tensor) -> torch.Tensor:
        scale = float(self.hparams.dt_scale)
        delta_cycle = delta_cycle.to(self.device).float()
        dt_norm = delta_cycle / scale
        dt_log = torch.log1p(delta_cycle) / np.log1p(scale)
        return torch.stack([dt_norm, dt_log], dim=-1)

    def forward(self, segments: list[list[torch.Tensor]], delta_cycle: torch.Tensor) -> torch.Tensor:
        cycle_embeddings = []
        for cycle_segments in segments:
            segment_embeddings = [self.encode_segment(segment) for segment in cycle_segments]
            cycle_embeddings.append(torch.stack(segment_embeddings, dim=0).mean(dim=0))

        cycle_emb = torch.stack(cycle_embeddings, dim=0)
        dt_emb = self.dt_embed(self._dt_features(delta_cycle))
        return self.head(torch.cat([cycle_emb, dt_emb], dim=-1)).squeeze(-1)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pred = self(batch["segments"], batch["delta_cycle"])
        target = batch["target_soh"].to(pred.device) / SOH_PERCENT_SCALE
        loss = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target) * SOH_PERCENT_SCALE
        rmse = torch.sqrt(loss + 1e-8) * SOH_PERCENT_SCALE

        self.log_dict(
            {
                f"{stage}/mae": mae,
                f"{stage}/rmse": rmse,
            },
            batch_size=target.shape[0],
            prog_bar=stage in {"train", "val"},
            on_step=False,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.dt_embed.parameters()) + list(self.head.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def make_task(cfg: DictConfig):
    if not cfg.report.use_clearml:
        return None
    try:
        from clearml import Task

        task = Task.init(
            project_name="BESS-JEPA",
            task_name="experiment_forecast_soh",
            reuse_last_task_id=False,
        )
        task.connect(OmegaConf.to_container(cfg, resolve=True))
        return task
    except Exception as e:
        print(f"ClearML 사용 불가: {e}")
        return None


def collect_predictions(
    model: ForecastSohProbe,
    loader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds, targets, current_cycles, target_cycles, delta_cycles = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            delta = batch["delta_cycle"].to(device)
            pred = model(batch["segments"], delta)
            preds.append((pred * SOH_PERCENT_SCALE).cpu().numpy())
            targets.append(batch["target_soh"].cpu().numpy())
            current_cycles.append(batch["current_cycle"].cpu().numpy())
            target_cycles.append(batch["target_cycle"].cpu().numpy())
            delta_cycles.append(batch["delta_cycle"].cpu().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(current_cycles),
        np.concatenate(target_cycles),
        np.concatenate(delta_cycles),
    )


def make_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "test/rmse": float(np.sqrt(np.mean(err ** 2))),
        "test/mae": float(np.mean(np.abs(err))),
    }


def plot_predictions(pred: np.ndarray, target: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(target, pred, s=10, alpha=0.45)
    lo = float(min(target.min(), pred.min()))
    hi = float(max(target.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, label="ideal")
    ax.set_xlabel("Target future SoH (%)")
    ax.set_ylabel("Predicted future SoH (%)")
    ax.set_title("Future SoH Prediction")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_by_target_cycle(pred: np.ndarray, target: np.ndarray, target_cycles: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(target_cycles)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(target_cycles[order], target[order], ".", label="target", markersize=4, alpha=0.55)
    ax.plot(target_cycles[order], pred[order], ".", label="pred", markersize=4, alpha=0.55)
    ax.set_xlabel("Target cycle index")
    ax.set_ylabel("SoH (%)")
    ax.set_title("Future SoH by Target Cycle")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_by_horizon(pred: np.ndarray, target: np.ndarray, delta_cycles: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = np.abs(pred - target)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(delta_cycles, err, s=10, alpha=0.4)
    ax.set_xlabel("Delta cycle")
    ax.set_ylabel("Absolute error (% SoH)")
    ax.set_title("Future SoH Error by Horizon")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def report_figure(task, title: str, series: str, fig, out_path: Path) -> None:
    if task is not None:
        task.get_logger().report_matplotlib_figure(title, series, fig, 0)
    else:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def report_summary(task, metrics: dict[str, float]) -> None:
    if task is None:
        return
    logger = task.get_logger()
    for key, value in metrics.items():
        logger.report_single_value(key, value)


@hydra.main(config_path="../../../config", config_name="forecast_soh", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)
    report_dir = Path(cfg.report.dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    task = make_task(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa = JEPA.load_from_checkpoint(cfg.checkpoint, map_location=device)
    jepa.eval()

    seq_len = cfg.data.seq_len or int(jepa.hparams.seq_len)
    if seq_len != int(jepa.hparams.seq_len):
        raise ValueError(
            f"Forecast SoH seq_len={seq_len} must match checkpoint seq_len={int(jepa.hparams.seq_len)}"
        )

    dm = ForecastSohLoader(
        data_dir=cfg.data.data_dir,
        seq_len=seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        stride=cfg.data.stride,
        processed_dir_name=cfg.data.processed_dir_name,
        operations=cfg.data.operations,
        max_horizon=cfg.data.max_horizon,
        horizon_stride=cfg.data.horizon_stride,
        min_segment_len=int(jepa.hparams.patch_size),
    )
    dm.setup()

    model = ForecastSohProbe(
        jepa,
        lr=cfg.probe.lr,
        weight_decay=cfg.probe.weight_decay,
        dt_scale=cfg.data.dt_scale,
    )
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[
            EarlyStopping(monitor="val/rmse", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(monitor="val/rmse", mode="min", save_top_k=1, filename="forecast-soh-probe-best"),
        ],
    )
    trainer.fit(model, dm)
    trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path="best")

    if trainer.checkpoint_callback.best_model_path:
        model = ForecastSohProbe.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            jepa=jepa,
            map_location=device,
        )
    model.to(device)

    pred, target, current_cycles, target_cycles, delta_cycles = collect_predictions(
        model,
        dm.test_dataloader(),
        device,
    )
    metrics = make_metrics(pred, target)
    report_summary(task, metrics)
    (report_dir / "final_summary.json").write_text(json.dumps(metrics, indent=2))

    fig = plot_predictions(pred, target)
    report_figure(task, "Forecast SoH", "pred_vs_target", fig, report_dir / "pred_vs_target.png")

    fig = plot_by_target_cycle(pred, target, target_cycles)
    report_figure(task, "Forecast SoH", "target_cycle_plot", fig, report_dir / "target_cycle_plot.png")

    fig = plot_error_by_horizon(pred, target, delta_cycles)
    report_figure(task, "Forecast SoH", "error_by_horizon", fig, report_dir / "error_by_horizon.png")

    print(f"forecast SoH reports saved under {report_dir}")
    print("final summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    if trainer.checkpoint_callback.best_model_path:
        print(f"best probe checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
