"""
Frozen JEPA encoder + linear head로 현재 cycle의 SoH를 예측한다.

입력: 한 cycle 안의 V/I/T segments
출력: 해당 cycle의 discharge capacity 기반 SoH scalar
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

from current_soh_loader import CurrentSohLoader
from jepa import JEPA
from sub_models import Encoder, Tokenizer

SOH_PERCENT_SCALE = 100.0


class CurrentSohProbe(L.LightningModule):
    """JEPA tokenizer/encoder는 얼리고 linear regression head만 학습한다."""

    def __init__(
        self,
        tokenizer: torch.nn.Module,
        encoder: torch.nn.Module,
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

        self.head = nn.Linear(int(embed_dim), 1)

    def encode_segment(self, segment: torch.Tensor) -> torch.Tensor:
        if segment.ndim != 2:
            raise ValueError(f"expected segment shape (C, T), got {tuple(segment.shape)}")
        if self.train_encoder:
            tokens = self.tokenizer(segment.to(self.device).unsqueeze(0))
            z = self.encoder(tokens)
        else:
            self.tokenizer.eval()
            self.encoder.eval()
            with torch.no_grad():
                tokens = self.tokenizer(segment.to(self.device).unsqueeze(0))
                z = self.encoder(tokens)
        return z.mean(dim=1).squeeze(0)

    def forward(self, segments: list[list[torch.Tensor]]) -> torch.Tensor:
        cycle_embeddings = []
        for cycle_segments in segments:
            segment_embeddings = [
                self.encode_segment(segment)
                for segment in cycle_segments
            ]
            cycle_embeddings.append(torch.stack(segment_embeddings, dim=0).mean(dim=0))

        cycle_emb = torch.stack(cycle_embeddings, dim=0)
        return self.head(cycle_emb).squeeze(-1)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pred = self(batch["segments"])
        target = batch["soh"].to(pred.device) / SOH_PERCENT_SCALE
        loss = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target) * SOH_PERCENT_SCALE
        rmse = torch.sqrt(loss + 1e-8) * SOH_PERCENT_SCALE

        self.log_dict(
            {
                f"{stage}/mae": mae,
                f"{stage}/rmse": rmse,
            },
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
) -> tuple[CurrentSohProbe, L.Trainer, CurrentSohLoader, str]:
    # Variant별 tokenizer/encoder와 Current-SoH datamodule 준비
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa = JEPA.load_from_checkpoint(cfg.checkpoint, map_location=device)
    jepa.eval()
    hp = jepa.hparams

    seq_len = cfg.data.seq_len or int(hp.seq_len)
    if seq_len != int(hp.seq_len):
        raise ValueError(
            f"Current SoH seq_len={seq_len} must match checkpoint seq_len={int(hp.seq_len)}"
        )

    dm = CurrentSohLoader(
        data_dir=cfg.data.data_dir,
        seq_len=seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        stride=cfg.data.stride,
        processed_dir_name=cfg.data.processed_dir_name,
        operations=cfg.data.operations,
        min_segment_len=int(hp.patch_size),
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
    model = CurrentSohProbe(
        tokenizer=tokenizer,
        encoder=encoder,
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
            EarlyStopping(monitor="val/rmse", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(monitor="val/rmse", mode="min", save_top_k=1, filename=f"current-soh-{variant}-best"),
        ],
    )
    trainer.fit(model, dm)
    if frozen_before is not None:
        for before, param in zip(frozen_before, encoder_params):
            if not torch.equal(before, param.detach().cpu()):
                raise RuntimeError(f"{variant}: frozen encoder/tokenizer parameters changed during training")
    trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path="best")

    # 이후 bench는 best checkpoint 기준으로 수행
    if trainer.checkpoint_callback.best_model_path:
        model = CurrentSohProbe.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            tokenizer=tokenizer,
            encoder=encoder,
            map_location=device,
        )
    model.to(device)
    return model, trainer, dm, device


def do_bench(
    model: CurrentSohProbe,
    dm: CurrentSohLoader,
    device: str,
    variant: str,
    task,
    report_dir: Path,
) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Test set 전체에 대해 prediction, target, cycle index 수집
    preds, targets, cycles = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in dm.test_dataloader():
            pred = model(batch["segments"])
            preds.append((pred * SOH_PERCENT_SCALE).cpu().numpy())
            targets.append(batch["soh"].cpu().numpy())
            cycles.append(batch["cycle_index"].cpu().numpy())

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    cycles = np.concatenate(cycles)

    # 스칼라 metric 정리 및 저장
    err = pred - target
    metrics = {
        "test/rmse": float(np.sqrt(np.mean(err ** 2))),
        "test/mae": float(np.mean(np.abs(err))),
    }
    if task is not None:
        logger = task.get_logger()
        for key, value in metrics.items():
            logger.report_single_value(f"{variant}/{key}", value)
    (report_dir / "final_summary.json").write_text(json.dumps(metrics, indent=2))

    # Predicted SoH vs target SoH 그림
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(target, pred, s=10, alpha=0.45)
    lo = float(min(target.min(), pred.min()))
    hi = float(max(target.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, label="ideal")
    ax.set_xlabel("Target SoH (%)")
    ax.set_ylabel("Predicted SoH (%)")
    ax.set_title("Current SoH Prediction")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "pred_vs_target.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Current SoH/{variant}/pred_vs_target",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # Cycle index 기준 target/prediction 변화 그림
    order = np.argsort(cycles)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cycles[order], target[order], ".", label="target", markersize=4, alpha=0.65)
    ax.plot(cycles[order], pred[order], ".", label="pred", markersize=4, alpha=0.65)
    ax.set_xlabel("Cycle index")
    ax.set_ylabel("SoH (%)")
    ax.set_title("SoH by Cycle")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "cycle_plot.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Current SoH/{variant}/cycle_plot",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    return metrics


@hydra.main(config_path="../../../config", config_name="current_soh", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)
    report_dir = Path(cfg.report.dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ClearML Task init
    task = None
    if cfg.report.use_clearml:
        try:
            from clearml import Task

            task = Task.init(
                project_name="BESS-JEPA",
                task_name="experiment_current_soh",
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
        print(f"\n=== current SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        warm_start_ckpt = best_paths.get("pretrained_frozen") if variant == "pretrained_finetune" else None
        model, trainer, dm, device = train(cfg, variant, warm_start_ckpt=warm_start_ckpt)
        comparison[variant] = do_bench(model, dm, device, variant, task, variant_dir)
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

    plot_keys = [key for key in ["test/rmse", "test/mae"] if key in key_order]
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
            task.get_logger().report_matplotlib_figure("Current SoH", "variant_comparison", fig, 0)
        plt.close(fig)

    if task is not None:
        logger = task.get_logger()
        for variant, metrics in comparison.items():
            for key, value in metrics.items():
                logger.report_single_value(f"{variant}/{key}", value)

    print(f"current SoH reports saved under {report_dir}")
    print("comparison summary:")
    for variant in variants:
        compact = {key: comparison[variant][key] for key in ["test/rmse", "test/mae"] if key in comparison[variant]}
        print(f"  {variant}: {compact}")
    for variant, path in best_paths.items():
        print(f"best probe checkpoint [{variant}]: {path}")


if __name__ == "__main__":
    main()
