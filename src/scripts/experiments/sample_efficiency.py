"""
Sample-efficiency check for frozen representations.

This compares:
  1. pretrained_frozen: pretrained JEPA tokenizer/encoder frozen + decoder probe
  2. random_frozen: random tokenizer/encoder frozen + decoder probe

The train split is subsampled at several fractions while val/test remain unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR / "models"))
sys.path.insert(0, str(SRC_DIR / "data"))

from forecast_loader import FEATURES, ForecastLoader
from jepa import JEPA
from sub_models import Decoder, Encoder, Tokenizer


class SampleEfficiencyProbe(L.LightningModule):
    """Frozen tokenizer/encoder with a forecast decoder probe."""

    def __init__(
        self,
        tokenizer: torch.nn.Module,
        encoder: torch.nn.Module,
        seq_len: int,
        patch_size: int,
        strides: int,
        num_channels: int,
        embed_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "encoder"])
        self.tokenizer = tokenizer
        self.encoder = encoder

        for p in self.tokenizer.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

        n_patches = (seq_len - patch_size) // strides + 1
        self.decoder = Decoder(
            n_patches=n_patches,
            embed_dim=embed_dim,
            n_features=num_channels,
            seq_len=seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.tokenizer.eval()
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.tokenizer(x)
            z = self.encoder(tokens)
        return self.decoder(z)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        y_hat = self(batch["x"])
        y = batch["y"]
        loss = F.smooth_l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)

        logs = {
            f"{stage}/mse": mse,
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
        return torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def train(
    cfg: DictConfig,
    variant: str,
    train_fraction: float,
) -> tuple[SampleEfficiencyProbe, L.Trainer, ForecastLoader, list[dict[str, float]], int, int]:
    # Forecast datamodule 준비 후 train index만 fraction별로 줄인다.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa = JEPA.load_from_checkpoint(cfg.checkpoint, map_location=device)
    jepa.eval()
    hp = jepa.hparams

    seq_len = cfg.data.seq_len or int(hp.seq_len)
    if seq_len != int(hp.seq_len):
        raise ValueError(
            f"Sample efficiency seq_len={seq_len} must match checkpoint seq_len={int(hp.seq_len)}"
        )

    dm = ForecastLoader(
        data_dir=cfg.data.data_dir,
        datasets=cfg.data.datasets,
        seq_len=seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        stride=cfg.data.stride,
        rollout_steps=cfg.data.rollout_steps,
        processed_dir_name=cfg.data.processed_dir_name,
    )
    dm.setup()
    if dm.train_ds is None:
        raise RuntimeError("ForecastLoader.setup() did not create train dataset")

    full_train_count = len(dm.train_ds)
    train_count = max(1, int(round(full_train_count * train_fraction)))
    if train_count < full_train_count:
        rng = np.random.default_rng(int(cfg.trainer.seed))
        selected = np.sort(rng.choice(full_train_count, size=train_count, replace=False))
        dm.train_ds.index = [dm.train_ds.index[int(i)] for i in selected]

    if variant == "pretrained_frozen":
        tokenizer = jepa.tokenizer
        encoder = jepa.encoder
    elif variant == "random_frozen":
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
    else:
        raise ValueError(f"unknown sample-efficiency variant: {variant}")

    model = SampleEfficiencyProbe(
        tokenizer=tokenizer,
        encoder=encoder,
        seq_len=int(hp.seq_len),
        patch_size=int(hp.patch_size),
        strides=int(hp.strides),
        num_channels=int(hp.num_channels),
        embed_dim=int(hp.embed_dim),
        lr=cfg.probe.lr,
        weight_decay=cfg.probe.weight_decay,
    )

    frozen_params = list(model.tokenizer.parameters()) + list(model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in frozen_params if p.requires_grad)
    probe_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(
        f"[{variant} fraction={train_fraction:g}] "
        f"train_samples={train_count}/{full_train_count} "
        f"encoder_trainable={encoder_trainable} probe_trainable={probe_trainable}"
    )
    if encoder_trainable != 0:
        raise RuntimeError(f"{variant}: encoder/tokenizer should be frozen")
    frozen_before = [p.detach().cpu().clone() for p in frozen_params]

    fraction_tag = str(train_fraction).replace(".", "p")
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[
            EarlyStopping(monitor="val/mse", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(
                monitor="val/mse",
                mode="min",
                save_top_k=1,
                filename=f"sample-efficiency-{variant}-{fraction_tag}-best",
            ),
        ],
    )
    trainer.fit(model, dm)
    for before, param in zip(frozen_before, frozen_params):
        if not torch.equal(before, param.detach().cpu()):
            raise RuntimeError(f"{variant}: frozen encoder/tokenizer parameters changed during training")

    test_results = trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path="best")
    if trainer.checkpoint_callback.best_model_path:
        model = SampleEfficiencyProbe.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            tokenizer=tokenizer,
            encoder=encoder,
            map_location=device,
        )
    model.to(device)
    return model, trainer, dm, test_results, train_count, full_train_count


def do_bench(
    test_results: list[dict[str, float]],
    variant: str,
    train_fraction: float,
    train_count: int,
    full_train_count: int,
    task,
    report_dir: Path,
) -> dict[str, float]:
    summary = {
        "variant": variant,
        "train_fraction": float(train_fraction),
        "train_count": int(train_count),
        "full_train_count": int(full_train_count),
    }
    if test_results:
        for key, value in test_results[0].items():
            summary[f"one_step/{key}"] = float(value)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if task is not None:
        logger = task.get_logger()
        label = f"{variant}/fraction_{train_fraction:g}"
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                logger.report_single_value(f"{label}/{key}", value)

    return summary


@hydra.main(config_path="../../../config", config_name="forecast", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)
    fractions = [0.1, 0.25, 0.5, 1.0]
    variants = ["pretrained_frozen", "random_frozen"]
    report_dir = Path("sample_efficiency_reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    task = None
    if cfg.report.use_clearml:
        try:
            from clearml import Task

            task = Task.init(
                project_name="BESS-JEPA",
                task_name="sample_efficiency",
                reuse_last_task_id=False,
            )
            task.connect(OmegaConf.to_container(cfg, resolve=True))
            task.connect({"sample_efficiency_fractions": fractions})
        except Exception as e:
            print(f"ClearML 사용 불가: {e}")

    rows: list[dict[str, float]] = []
    best_paths: dict[str, str] = {}
    for fraction in fractions:
        for variant in variants:
            print(f"\n=== sample efficiency: {variant}, train_fraction={fraction:g} ===")
            model, trainer, dm, test_results, train_count, full_train_count = train(
                cfg,
                variant,
                fraction,
            )
            del model, dm
            row = do_bench(
                test_results,
                variant,
                fraction,
                train_count,
                full_train_count,
                task,
                report_dir / f"fraction_{str(fraction).replace('.', 'p')}" / variant,
            )
            rows.append(row)
            if trainer.checkpoint_callback.best_model_path:
                best_paths[f"{variant}@{fraction:g}"] = trainer.checkpoint_callback.best_model_path

    key_order = sorted({key for row in rows for key in row})
    (report_dir / "comparison_summary.json").write_text(json.dumps(rows, indent=2))
    with (report_dir / "comparison_summary.csv").open("w") as f:
        f.write(",".join(key_order) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(key, "")) for key in key_order) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_keys = [
        key
        for key in ["one_step/test/mse", "one_step/test/mae"]
        if any(key in row for row in rows)
    ]
    if plot_keys:
        fig, axes = plt.subplots(len(plot_keys), 1, figsize=(8, 3.2 * len(plot_keys)), squeeze=False)
        for row_idx, key in enumerate(plot_keys):
            ax = axes[row_idx, 0]
            for variant in variants:
                values = [
                    next(
                        row[key]
                        for row in rows
                        if row["variant"] == variant and row["train_fraction"] == fraction
                    )
                    for fraction in fractions
                ]
                ax.plot(fractions, values, marker="o", label=variant)
            ax.set_title(key)
            ax.set_xlabel("Train fraction")
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(report_dir / "sample_efficiency.png", dpi=150, bbox_inches="tight")
        if task is not None:
            task.get_logger().report_matplotlib_figure("Sample Efficiency", "one_step_metrics", fig, 0)
        plt.close(fig)

    print(f"sample efficiency reports saved under {report_dir}")
    print("comparison summary:")
    for row in rows:
        compact = {
            key: row[key]
            for key in ["one_step/test/mse", "one_step/test/mae"]
            if key in row
        }
        print(f"  {row['variant']} @ {row['train_fraction']}: {compact}")
    for key, path in best_paths.items():
        print(f"best probe checkpoint [{key}]: {path}")


if __name__ == "__main__":
    main()
