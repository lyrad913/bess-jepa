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
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR / "models"))
sys.path.insert(0, str(SRC_DIR / "data"))

from forecast_soh_loader import ForecastSohLoader
from context_query_projection_jepa import ContextQueryProjectionJEPA
from context_mask_projection_jepa import ContextMaskProjectionJEPA
from context_mask_jepa import ContextMaskJEPA
from jepa import JEPA
from masked_autoencoder import MaskedAutoencoder
from projection_jepa import ProjectionJEPA
from sub_models import Encoder, Tokenizer
from ts_jepa import TSJEPA
from ts_jepa_sigreg import TSJEPASIGReg

SOH_PERCENT_SCALE = 100.0


def assert_encoder_compatible(reference_hp, candidate_hp, variant: str) -> None:
    for key in ["seq_len", "patch_size", "strides", "num_channels", "embed_dim"]:
        ref = int(getattr(reference_hp, key))
        cand = int(getattr(candidate_hp, key))
        if ref != cand:
            raise ValueError(f"{variant}: checkpoint hparam {key}={cand} does not match JEPA {key}={ref}")


def load_reference_pretrain_model(cfg: DictConfig, device: str) -> L.LightningModule:
    candidates = [
        ("experiments.pretrained_frozen", "checkpoint", JEPA),
        ("experiments.pretrained_finetune", "checkpoint", JEPA),
        ("experiments.pretrained_projection_jepa_frozen", "projection_jepa_checkpoint", ProjectionJEPA),
        ("experiments.pretrained_context_mask_jepa_frozen", "context_mask_jepa_checkpoint", ContextMaskJEPA),
        (
            "experiments.pretrained_context_mask_projection_jepa_frozen",
            "context_mask_projection_jepa_checkpoint",
            ContextMaskProjectionJEPA,
        ),
        (
            "experiments.pretrained_context_mask_projection_jepa_finetune",
            "context_mask_projection_jepa_checkpoint",
            ContextMaskProjectionJEPA,
        ),
        (
            "experiments.pretrained_context_query_projection_jepa_frozen",
            "context_query_projection_jepa_checkpoint",
            ContextQueryProjectionJEPA,
        ),
        (
            "experiments.pretrained_context_query_projection_jepa_finetune",
            "context_query_projection_jepa_checkpoint",
            ContextQueryProjectionJEPA,
        ),
        ("experiments.pretrained_ts_jepa_frozen", "ts_jepa_checkpoint", TSJEPA),
        ("experiments.pretrained_ts_jepa_sigreg_frozen", "ts_jepa_sigreg_checkpoint", TSJEPASIGReg),
        ("experiments.pretrained_mae_frozen", "mae_checkpoint", MaskedAutoencoder),
    ]
    for flag, checkpoint_key, cls in candidates:
        if OmegaConf.select(cfg, flag, default=False):
            return cls.load_from_checkpoint(OmegaConf.select(cfg, checkpoint_key), map_location=device)
    return JEPA.load_from_checkpoint(cfg.checkpoint, map_location=device)


class ForecastSohProbe(L.LightningModule):
    """Frozen JEPA tokenizer/encoder with a delta-cycle-conditioned SoH head."""

    def __init__(
        self,
        tokenizer: torch.nn.Module,
        encoder: torch.nn.Module,
        embed_dim: int,
        train_encoder: bool,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dt_scale: float = 300.0,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "encoder"])

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.train_encoder = train_encoder
        self.pooling = str(pooling)
        if self.pooling not in {"mean", "attention"}:
            raise ValueError(f"unknown pooling={self.pooling}; expected 'mean' or 'attention'")
        for p in self.tokenizer.parameters():
            p.requires_grad = train_encoder
        for p in self.encoder.parameters():
            p.requires_grad = train_encoder

        embed_dim = int(embed_dim)
        if self.pooling == "attention":
            self.patch_pool = nn.Linear(embed_dim, 1)
            self.segment_pool = nn.Linear(embed_dim, 1)
        self.dt_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.head = nn.Linear(embed_dim * 2, 1)

    def pool_embeddings(self, x: torch.Tensor, pooler: nn.Linear) -> torch.Tensor:
        weights = torch.softmax(pooler(x).squeeze(-1), dim=0)
        return torch.sum(x * weights.unsqueeze(-1), dim=0)

    def pool_patch_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N, D) -> (B, D)."""
        if self.pooling == "attention":
            weights = torch.softmax(self.patch_pool(z).squeeze(-1), dim=1)
            return torch.sum(z * weights.unsqueeze(-1), dim=1)
        return z.mean(dim=1)

    def encode_segments(self, segments: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        """Encode all segments in a batch, grouped by raw sequence length."""
        flat_segments = [segment for cycle_segments in segments for segment in cycle_segments]
        if not flat_segments:
            raise ValueError("expected at least one segment in the batch")
        for segment in flat_segments:
            if segment.ndim != 2:
                raise ValueError(f"expected segment shape (C, T), got {tuple(segment.shape)}")

        groups: dict[int, list[int]] = {}
        for idx, segment in enumerate(flat_segments):
            groups.setdefault(int(segment.shape[-1]), []).append(idx)

        flat_embeddings: list[torch.Tensor | None] = [None] * len(flat_segments)
        if not self.train_encoder:
            self.tokenizer.eval()
            self.encoder.eval()

        grad_context = torch.enable_grad() if self.train_encoder else torch.no_grad()
        with grad_context:
            for _, indices in groups.items():
                x = torch.stack([flat_segments[i].to(self.device) for i in indices], dim=0)
                tokens = self.tokenizer(x)
                z = self.encoder(tokens)
                pooled = self.pool_patch_embeddings(z)
                for local_idx, flat_idx in enumerate(indices):
                    flat_embeddings[flat_idx] = pooled[local_idx]

        cycle_embeddings = []
        cursor = 0
        for cycle_segments in segments:
            n_segments = len(cycle_segments)
            segment_embeddings = torch.stack(
                [emb for emb in flat_embeddings[cursor : cursor + n_segments] if emb is not None],
                dim=0,
            )
            cursor += n_segments
            if self.pooling == "attention":
                cycle_embeddings.append(self.pool_embeddings(segment_embeddings, self.segment_pool))
            else:
                cycle_embeddings.append(segment_embeddings.mean(dim=0))
        return cycle_embeddings

    def _dt_features(self, delta_cycle: torch.Tensor) -> torch.Tensor:
        scale = float(self.hparams.dt_scale)
        delta_cycle = delta_cycle.to(self.device).float()
        dt_norm = delta_cycle / scale
        dt_log = torch.log1p(delta_cycle) / np.log1p(scale)
        return torch.stack([dt_norm, dt_log], dim=-1)

    def forward(self, segments: list[list[torch.Tensor]], delta_cycle: torch.Tensor) -> torch.Tensor:
        cycle_emb = torch.stack(self.encode_segments(segments), dim=0)
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
        params = list(self.dt_embed.parameters()) + list(self.head.parameters())
        if self.pooling == "attention":
            params = list(self.patch_pool.parameters()) + list(self.segment_pool.parameters()) + params
        if self.train_encoder:
            params = list(self.tokenizer.parameters()) + list(self.encoder.parameters()) + params
        return torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def train(
    model: ForecastSohProbe,
    dm: ForecastSohLoader,
    cfg: DictConfig,
    checkpoint_name: str,
    frozen_modules: list[torch.nn.Module] | None = None,
) -> tuple[L.Trainer, str]:
    # 이미 main에서 명시적으로 만든 model과 datamodule을 받아 학습만 수행
    frozen_params = []
    for module in frozen_modules or []:
        frozen_params.extend(list(module.parameters()))
    encoder_params = list(model.tokenizer.parameters()) + list(model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in encoder_params if p.requires_grad)
    probe_params = list(model.dt_embed.parameters()) + list(model.head.parameters())
    if model.pooling == "attention":
        probe_params = list(model.patch_pool.parameters()) + list(model.segment_pool.parameters()) + probe_params
    probe_trainable = sum(p.numel() for p in probe_params if p.requires_grad)
    print(
        f"[{checkpoint_name}] lr={model.hparams.lr} train_encoder={model.train_encoder} "
        f"encoder_trainable={encoder_trainable} probe_trainable={probe_trainable}"
    )
    if frozen_params and any(p.requires_grad for p in frozen_params):
        raise RuntimeError(f"{checkpoint_name}: frozen parameters still require grad")
    frozen_before = [p.detach().cpu().clone() for p in frozen_params]
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=TensorBoardLogger(save_dir="logs/experiments/forecast_soh", name=checkpoint_name),
        callbacks=[
            EarlyStopping(monitor="val/rmse", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(monitor="val/rmse", mode="min", save_top_k=1, filename=checkpoint_name),
        ],
    )
    trainer.fit(model, dm)
    for before, param in zip(frozen_before, frozen_params):
        if not torch.equal(before, param.detach().cpu()):
            raise RuntimeError(f"{checkpoint_name}: frozen parameters changed during training")
    best_path = trainer.checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError(f"{checkpoint_name}: best checkpoint was not saved")
    best_score = trainer.checkpoint_callback.best_model_score
    best_score_value = float(best_score.detach().cpu()) if best_score is not None else float("nan")
    print(
        f"[{checkpoint_name}] best_val/rmse={best_score_value:.6g} "
        f"current_epoch={trainer.current_epoch} max_epochs={cfg.trainer.max_epochs} "
        f"best_checkpoint={best_path}"
    )
    return trainer, best_path


def do_bench(
    model: ForecastSohProbe,
    dm: ForecastSohLoader,
    device: str,
    variant: str,
    task,
    report_dir: Path,
) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Test set 전체에 대해 prediction, target, cycle/horizon metadata 수집
    preds, targets, target_cycles, delta_cycles = [], [], [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dm.test_dataloader():
            delta = batch["delta_cycle"].to(device)
            pred = model(batch["segments"], delta)
            preds.append((pred * SOH_PERCENT_SCALE).cpu().numpy())
            targets.append(batch["target_soh"].cpu().numpy())
            target_cycles.append(batch["target_cycle"].cpu().numpy())
            delta_cycles.append(batch["delta_cycle"].cpu().numpy())

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    target_cycles = np.concatenate(target_cycles)
    delta_cycles = np.concatenate(delta_cycles)

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

    # Predicted future SoH vs target future SoH 그림
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
    fig.savefig(report_dir / "pred_vs_target.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Forecast SoH/{variant}/pred_vs_target",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # Target cycle 기준 target/prediction 변화 그림
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
    fig.savefig(report_dir / "target_cycle_plot.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Forecast SoH/{variant}/target_cycle_plot",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    # Horizon이 길어질 때 absolute error가 어떻게 변하는지 확인
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(delta_cycles, np.abs(err), s=10, alpha=0.4)
    ax.set_xlabel("Delta cycle")
    ax.set_ylabel("Absolute error (% SoH)")
    ax.set_title("Future SoH Error by Horizon")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "error_by_horizon.png", dpi=150, bbox_inches="tight")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            f"Forecast SoH/{variant}/error_by_horizon",
            "figure",
            fig,
            0,
        )
    plt.close(fig)

    return metrics


@hydra.main(config_path="../../../config", config_name="forecast_soh", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report_dir = Path(cfg.report.dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ClearML Task init
    task = None
    if cfg.report.use_clearml:
        try:
            from clearml import Task

            task = Task.init(
                project_name="BESS-JEPA",
                task_name="experiment_forecast_soh",
                reuse_last_task_id=False,
                auto_connect_frameworks={"matplotlib": False},
            )
            task.connect(OmegaConf.to_container(cfg, resolve=True))
        except Exception as e:
            print(f"ClearML 사용 불가: {e}")

    # enabled pretrain checkpoint의 hparams와 future-SoH datamodule 준비
    reference_model = load_reference_pretrain_model(cfg, device)
    reference_model.eval()
    hp = reference_model.hparams
    seq_len = cfg.data.seq_len or int(hp.seq_len)
    if seq_len != int(hp.seq_len):
        raise ValueError(f"Forecast SoH seq_len={seq_len} must match checkpoint seq_len={int(hp.seq_len)}")
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
        min_segment_len=int(hp.patch_size),
    )
    dm.setup()

    comparison: dict[str, dict[str, float]] = {}
    best_paths: dict[str, str] = {}

    # pretrained_frozen: pretrained tokenizer/encoder를 얼리고 horizon-conditioned SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_frozen_jepa = JEPA.load_from_checkpoint(cfg.checkpoint, map_location=device)
        pretrained_frozen_model = ForecastSohProbe(
            tokenizer=pretrained_frozen_jepa.tokenizer,
            encoder=pretrained_frozen_jepa.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, pretrained_frozen_ckpt = train(
            pretrained_frozen_model,
            dm,
            cfg,
            "forecast-soh-pretrained_frozen-best",
            frozen_modules=[pretrained_frozen_model.tokenizer, pretrained_frozen_model.encoder],
        )
        pretrained_frozen_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_frozen_ckpt,
            tokenizer=pretrained_frozen_model.tokenizer,
            encoder=pretrained_frozen_model.encoder,
            map_location=device,
        )
        pretrained_frozen_best.to(device)
        test_results = trainer.test(pretrained_frozen_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_frozen_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_frozen_ckpt

    # pretrained_frozen_attention: pretrained tokenizer/encoder는 얼리고 attention pooling future-SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_frozen_attention", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_frozen_attention"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_frozen_attention_jepa = JEPA.load_from_checkpoint(cfg.checkpoint, map_location=device)
        pretrained_frozen_attention_model = ForecastSohProbe(
            tokenizer=pretrained_frozen_attention_jepa.tokenizer,
            encoder=pretrained_frozen_attention_jepa.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
            pooling="attention",
        )
        trainer, pretrained_frozen_attention_ckpt = train(
            pretrained_frozen_attention_model,
            dm,
            cfg,
            "forecast-soh-pretrained_frozen_attention-best",
            frozen_modules=[pretrained_frozen_attention_model.tokenizer, pretrained_frozen_attention_model.encoder],
        )
        pretrained_frozen_attention_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_frozen_attention_ckpt,
            tokenizer=pretrained_frozen_attention_model.tokenizer,
            encoder=pretrained_frozen_attention_model.encoder,
            map_location=device,
        )
        pretrained_frozen_attention_best.to(device)
        test_results = trainer.test(pretrained_frozen_attention_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_frozen_attention_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_frozen_attention_ckpt

    # random_frozen: random tokenizer/encoder를 얼리고 horizon-conditioned SoH head만 학습
    if OmegaConf.select(cfg, "experiments.random_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "random_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        random_frozen_model = ForecastSohProbe(
            tokenizer=Tokenizer(
                seq_len=int(hp.seq_len),
                patch_len=int(hp.patch_size),
                strides=int(hp.strides),
                n_features=int(hp.num_channels),
                embed_dim=int(hp.embed_dim),
            ),
            encoder=Encoder(
                embed_dim=int(hp.embed_dim),
                nhead=int(hp.enc_nhead),
                num_layers=int(hp.enc_layers),
            ),
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, random_frozen_ckpt = train(
            random_frozen_model,
            dm,
            cfg,
            "forecast-soh-random_frozen-best",
            frozen_modules=[random_frozen_model.tokenizer, random_frozen_model.encoder],
        )
        random_frozen_best = ForecastSohProbe.load_from_checkpoint(
            random_frozen_ckpt,
            tokenizer=random_frozen_model.tokenizer,
            encoder=random_frozen_model.encoder,
            map_location=device,
        )
        random_frozen_best.to(device)
        test_results = trainer.test(random_frozen_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(random_frozen_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = random_frozen_ckpt

    # random_frozen_attention: random tokenizer/encoder는 얼리고 attention pooling future-SoH head만 학습
    if OmegaConf.select(cfg, "experiments.random_frozen_attention", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "random_frozen_attention"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        random_frozen_attention_model = ForecastSohProbe(
            tokenizer=Tokenizer(
                seq_len=int(hp.seq_len),
                patch_len=int(hp.patch_size),
                strides=int(hp.strides),
                n_features=int(hp.num_channels),
                embed_dim=int(hp.embed_dim),
            ),
            encoder=Encoder(
                embed_dim=int(hp.embed_dim),
                nhead=int(hp.enc_nhead),
                num_layers=int(hp.enc_layers),
            ),
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
            pooling="attention",
        )
        trainer, random_frozen_attention_ckpt = train(
            random_frozen_attention_model,
            dm,
            cfg,
            "forecast-soh-random_frozen_attention-best",
            frozen_modules=[random_frozen_attention_model.tokenizer, random_frozen_attention_model.encoder],
        )
        random_frozen_attention_best = ForecastSohProbe.load_from_checkpoint(
            random_frozen_attention_ckpt,
            tokenizer=random_frozen_attention_model.tokenizer,
            encoder=random_frozen_attention_model.encoder,
            map_location=device,
        )
        random_frozen_attention_best.to(device)
        test_results = trainer.test(random_frozen_attention_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(random_frozen_attention_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = random_frozen_attention_ckpt

    # random_supervised: random tokenizer/encoder와 SoH head를 모두 supervised로 학습
    if OmegaConf.select(cfg, "experiments.random_supervised", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "random_supervised"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        random_supervised_model = ForecastSohProbe(
            tokenizer=Tokenizer(
                seq_len=int(hp.seq_len),
                patch_len=int(hp.patch_size),
                strides=int(hp.strides),
                n_features=int(hp.num_channels),
                embed_dim=int(hp.embed_dim),
            ),
            encoder=Encoder(
                embed_dim=int(hp.embed_dim),
                nhead=int(hp.enc_nhead),
                num_layers=int(hp.enc_layers),
            ),
            embed_dim=int(hp.embed_dim),
            train_encoder=True,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, random_supervised_ckpt = train(
            random_supervised_model,
            dm,
            cfg,
            "forecast-soh-random_supervised-best",
        )
        random_supervised_best = ForecastSohProbe.load_from_checkpoint(
            random_supervised_ckpt,
            tokenizer=random_supervised_model.tokenizer,
            encoder=random_supervised_model.encoder,
            map_location=device,
        )
        random_supervised_best.to(device)
        test_results = trainer.test(random_supervised_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(random_supervised_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = random_supervised_ckpt

    # pretrained_finetune: pretrained_frozen의 학습된 SoH head까지 불러온 뒤 1e-4로 전체 finetuning
    if OmegaConf.select(cfg, "experiments.pretrained_finetune", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_finetune"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_finetune_jepa = JEPA.load_from_checkpoint(cfg.checkpoint, map_location=device)
        pretrained_finetune_model = ForecastSohProbe(
            tokenizer=pretrained_finetune_jepa.tokenizer,
            encoder=pretrained_finetune_jepa.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=True,
            lr=1e-4,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        checkpoint = torch.load(pretrained_frozen_ckpt, map_location=device, weights_only=False)
        pretrained_finetune_model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"[{variant}] warm-start from pretrained_frozen checkpoint: {pretrained_frozen_ckpt}")
        trainer, pretrained_finetune_ckpt = train(
            pretrained_finetune_model,
            dm,
            cfg,
            "forecast-soh-pretrained_finetune-best",
        )
        pretrained_finetune_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_finetune_ckpt,
            tokenizer=pretrained_finetune_model.tokenizer,
            encoder=pretrained_finetune_model.encoder,
            map_location=device,
        )
        pretrained_finetune_best.to(device)
        test_results = trainer.test(pretrained_finetune_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_finetune_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_finetune_ckpt

    # pretrained_projection_jepa_frozen: projection-SIGReg JEPA tokenizer/encoder를 얼리고 SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_projection_jepa_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_projection_jepa_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_projection_jepa = ProjectionJEPA.load_from_checkpoint(cfg.projection_jepa_checkpoint, map_location=device)
        assert_encoder_compatible(hp, pretrained_projection_jepa.hparams, variant)
        pretrained_projection_jepa_model = ForecastSohProbe(
            tokenizer=pretrained_projection_jepa.tokenizer,
            encoder=pretrained_projection_jepa.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, pretrained_projection_jepa_ckpt = train(
            pretrained_projection_jepa_model,
            dm,
            cfg,
            "forecast-soh-pretrained_projection_jepa_frozen-best",
            frozen_modules=[pretrained_projection_jepa_model.tokenizer, pretrained_projection_jepa_model.encoder],
        )
        pretrained_projection_jepa_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_projection_jepa_ckpt,
            tokenizer=pretrained_projection_jepa_model.tokenizer,
            encoder=pretrained_projection_jepa_model.encoder,
            map_location=device,
        )
        pretrained_projection_jepa_best.to(device)
        test_results = trainer.test(pretrained_projection_jepa_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_projection_jepa_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_projection_jepa_ckpt

    # pretrained_context_mask_jepa_frozen: 미래 예측 context-masked JEPA tokenizer/encoder를 얼리고 SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_context_mask_jepa_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_context_mask_jepa_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_context_mask_jepa = ContextMaskJEPA.load_from_checkpoint(
            cfg.context_mask_jepa_checkpoint,
            map_location=device,
        )
        assert_encoder_compatible(hp, pretrained_context_mask_jepa.hparams, variant)
        pretrained_context_mask_jepa_model = ForecastSohProbe(
            tokenizer=pretrained_context_mask_jepa.tokenizer,
            encoder=pretrained_context_mask_jepa.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, pretrained_context_mask_jepa_ckpt = train(
            pretrained_context_mask_jepa_model,
            dm,
            cfg,
            "forecast-soh-pretrained_context_mask_jepa_frozen-best",
            frozen_modules=[
                pretrained_context_mask_jepa_model.tokenizer,
                pretrained_context_mask_jepa_model.encoder,
            ],
        )
        pretrained_context_mask_jepa_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_context_mask_jepa_ckpt,
            tokenizer=pretrained_context_mask_jepa_model.tokenizer,
            encoder=pretrained_context_mask_jepa_model.encoder,
            map_location=device,
        )
        pretrained_context_mask_jepa_best.to(device)
        test_results = trainer.test(pretrained_context_mask_jepa_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_context_mask_jepa_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_context_mask_jepa_ckpt

    # pretrained_context_mask_projection_jepa_frozen: context-masked projection-SIGReg JEPA tokenizer/encoder를 얼리고 SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_context_mask_projection_jepa_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_context_mask_projection_jepa_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_context_mask_projection_jepa = ContextMaskProjectionJEPA.load_from_checkpoint(
            cfg.context_mask_projection_jepa_checkpoint,
            map_location=device,
        )
        assert_encoder_compatible(hp, pretrained_context_mask_projection_jepa.hparams, variant)
        pretrained_context_mask_projection_jepa_model = ForecastSohProbe(
            tokenizer=pretrained_context_mask_projection_jepa.tokenizer,
            encoder=pretrained_context_mask_projection_jepa.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, pretrained_context_mask_projection_jepa_ckpt = train(
            pretrained_context_mask_projection_jepa_model,
            dm,
            cfg,
            "forecast-soh-pretrained_context_mask_projection_jepa_frozen-best",
            frozen_modules=[
                pretrained_context_mask_projection_jepa_model.tokenizer,
                pretrained_context_mask_projection_jepa_model.encoder,
            ],
        )
        pretrained_context_mask_projection_jepa_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_context_mask_projection_jepa_ckpt,
            tokenizer=pretrained_context_mask_projection_jepa_model.tokenizer,
            encoder=pretrained_context_mask_projection_jepa_model.encoder,
            map_location=device,
        )
        pretrained_context_mask_projection_jepa_best.to(device)
        test_results = trainer.test(pretrained_context_mask_projection_jepa_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_context_mask_projection_jepa_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_context_mask_projection_jepa_ckpt

    # pretrained_context_mask_projection_jepa_finetune: frozen encoder + SoH head 학습 checkpoint에서 전체 finetuning
    if OmegaConf.select(cfg, "experiments.pretrained_context_mask_projection_jepa_finetune", default=True):
        warm_start_ckpt = OmegaConf.select(cfg, "context_mask_projection_jepa_probe_checkpoint")
        if warm_start_ckpt is None:
            warm_start_ckpt = best_paths.get("pretrained_context_mask_projection_jepa_frozen")
        if warm_start_ckpt is None:
            raise RuntimeError(
                "pretrained_context_mask_projection_jepa_finetune requires "
                "context_mask_projection_jepa_probe_checkpoint or "
                "experiments.pretrained_context_mask_projection_jepa_frozen=true"
            )
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_context_mask_projection_jepa_finetune"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_context_mask_projection_jepa = ContextMaskProjectionJEPA.load_from_checkpoint(
            cfg.context_mask_projection_jepa_checkpoint,
            map_location=device,
        )
        assert_encoder_compatible(hp, pretrained_context_mask_projection_jepa.hparams, variant)
        pretrained_context_mask_projection_jepa_model = ForecastSohProbe(
            tokenizer=pretrained_context_mask_projection_jepa.tokenizer,
            encoder=pretrained_context_mask_projection_jepa.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=True,
            lr=1e-4,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        checkpoint = torch.load(warm_start_ckpt, map_location=device, weights_only=False)
        pretrained_context_mask_projection_jepa_model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"[{variant}] warm-start from frozen-encoder probe checkpoint: {warm_start_ckpt}")
        trainer, pretrained_context_mask_projection_jepa_finetune_ckpt = train(
            pretrained_context_mask_projection_jepa_model,
            dm,
            cfg,
            "forecast-soh-pretrained_context_mask_projection_jepa_finetune-best",
        )
        pretrained_context_mask_projection_jepa_finetune_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_context_mask_projection_jepa_finetune_ckpt,
            tokenizer=pretrained_context_mask_projection_jepa_model.tokenizer,
            encoder=pretrained_context_mask_projection_jepa_model.encoder,
            map_location=device,
        )
        pretrained_context_mask_projection_jepa_finetune_best.to(device)
        test_results = trainer.test(pretrained_context_mask_projection_jepa_finetune_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(
            pretrained_context_mask_projection_jepa_finetune_best,
            dm,
            device,
            variant,
            task,
            variant_dir,
        )
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_context_mask_projection_jepa_finetune_ckpt

    # pretrained_context_query_projection_jepa_frozen: query-projection JEPA encoder를 얼리고 SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_context_query_projection_jepa_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_context_query_projection_jepa_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained = ContextQueryProjectionJEPA.load_from_checkpoint(
            cfg.context_query_projection_jepa_checkpoint, map_location=device
        )
        assert_encoder_compatible(hp, pretrained.hparams, variant)
        model = ForecastSohProbe(
            tokenizer=pretrained.tokenizer,
            encoder=pretrained.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, probe_ckpt = train(
            model,
            dm,
            cfg,
            "forecast-soh-pretrained_context_query_projection_jepa_frozen-best",
            frozen_modules=[model.tokenizer, model.encoder],
        )
        best = ForecastSohProbe.load_from_checkpoint(
            probe_ckpt,
            tokenizer=model.tokenizer,
            encoder=model.encoder,
            map_location=device,
        )
        best.to(device)
        test_results = trainer.test(best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = probe_ckpt

    # pretrained_context_query_projection_jepa_finetune: frozen probe checkpoint에서 전체 finetuning
    if OmegaConf.select(cfg, "experiments.pretrained_context_query_projection_jepa_finetune", default=True):
        warm_start_ckpt = OmegaConf.select(cfg, "context_query_projection_jepa_probe_checkpoint")
        if warm_start_ckpt is None:
            warm_start_ckpt = best_paths.get("pretrained_context_query_projection_jepa_frozen")
        if warm_start_ckpt is None:
            raise RuntimeError(
                "pretrained_context_query_projection_jepa_finetune requires "
                "context_query_projection_jepa_probe_checkpoint or "
                "experiments.pretrained_context_query_projection_jepa_frozen=true"
            )
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_context_query_projection_jepa_finetune"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained = ContextQueryProjectionJEPA.load_from_checkpoint(
            cfg.context_query_projection_jepa_checkpoint, map_location=device
        )
        assert_encoder_compatible(hp, pretrained.hparams, variant)
        model = ForecastSohProbe(
            tokenizer=pretrained.tokenizer,
            encoder=pretrained.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=True,
            lr=1e-4,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        checkpoint = torch.load(warm_start_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"[{variant}] warm-start from frozen-encoder probe checkpoint: {warm_start_ckpt}")
        trainer, finetune_ckpt = train(
            model,
            dm,
            cfg,
            "forecast-soh-pretrained_context_query_projection_jepa_finetune-best",
        )
        best = ForecastSohProbe.load_from_checkpoint(
            finetune_ckpt,
            tokenizer=model.tokenizer,
            encoder=model.encoder,
            map_location=device,
        )
        best.to(device)
        test_results = trainer.test(best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = finetune_ckpt

    # pretrained_ts_jepa_frozen: pretrained TS-JEPA tokenizer/encoder를 얼리고 SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_ts_jepa_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_ts_jepa_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_ts_jepa = TSJEPA.load_from_checkpoint(cfg.ts_jepa_checkpoint, map_location=device)
        assert_encoder_compatible(hp, pretrained_ts_jepa.hparams, variant)
        pretrained_ts_jepa_model = ForecastSohProbe(
            tokenizer=pretrained_ts_jepa.tokenizer,
            encoder=pretrained_ts_jepa.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, pretrained_ts_jepa_ckpt = train(
            pretrained_ts_jepa_model,
            dm,
            cfg,
            "forecast-soh-pretrained_ts_jepa_frozen-best",
            frozen_modules=[pretrained_ts_jepa_model.tokenizer, pretrained_ts_jepa_model.encoder],
        )
        pretrained_ts_jepa_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_ts_jepa_ckpt,
            tokenizer=pretrained_ts_jepa_model.tokenizer,
            encoder=pretrained_ts_jepa_model.encoder,
            map_location=device,
        )
        pretrained_ts_jepa_best.to(device)
        test_results = trainer.test(pretrained_ts_jepa_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_ts_jepa_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_ts_jepa_ckpt

    # pretrained_ts_jepa_sigreg_frozen: EMA 없이 SIGReg로 학습한 TS-JEPA encoder를 얼리고 SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_ts_jepa_sigreg_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_ts_jepa_sigreg_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_ts_jepa_sigreg = TSJEPASIGReg.load_from_checkpoint(cfg.ts_jepa_sigreg_checkpoint, map_location=device)
        assert_encoder_compatible(hp, pretrained_ts_jepa_sigreg.hparams, variant)
        pretrained_ts_jepa_sigreg_model = ForecastSohProbe(
            tokenizer=pretrained_ts_jepa_sigreg.tokenizer,
            encoder=pretrained_ts_jepa_sigreg.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, pretrained_ts_jepa_sigreg_ckpt = train(
            pretrained_ts_jepa_sigreg_model,
            dm,
            cfg,
            "forecast-soh-pretrained_ts_jepa_sigreg_frozen-best",
            frozen_modules=[pretrained_ts_jepa_sigreg_model.tokenizer, pretrained_ts_jepa_sigreg_model.encoder],
        )
        pretrained_ts_jepa_sigreg_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_ts_jepa_sigreg_ckpt,
            tokenizer=pretrained_ts_jepa_sigreg_model.tokenizer,
            encoder=pretrained_ts_jepa_sigreg_model.encoder,
            map_location=device,
        )
        pretrained_ts_jepa_sigreg_best.to(device)
        test_results = trainer.test(pretrained_ts_jepa_sigreg_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_ts_jepa_sigreg_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_ts_jepa_sigreg_ckpt

    # pretrained_mae_frozen: pretrained MAE tokenizer/encoder를 얼리고 SoH head만 학습
    if OmegaConf.select(cfg, "experiments.pretrained_mae_frozen", default=True):
        L.seed_everything(cfg.trainer.seed)
        variant = "pretrained_mae_frozen"
        print(f"\n=== forecast SoH variant: {variant} ===")
        variant_dir = report_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pretrained_mae = MaskedAutoencoder.load_from_checkpoint(cfg.mae_checkpoint, map_location=device)
        assert_encoder_compatible(hp, pretrained_mae.hparams, variant)
        pretrained_mae_model = ForecastSohProbe(
            tokenizer=pretrained_mae.tokenizer,
            encoder=pretrained_mae.encoder,
            embed_dim=int(hp.embed_dim),
            train_encoder=False,
            lr=cfg.probe.lr,
            weight_decay=cfg.probe.weight_decay,
            dt_scale=cfg.data.dt_scale,
        )
        trainer, pretrained_mae_ckpt = train(
            pretrained_mae_model,
            dm,
            cfg,
            "forecast-soh-pretrained_mae_frozen-best",
            frozen_modules=[pretrained_mae_model.tokenizer, pretrained_mae_model.encoder],
        )
        pretrained_mae_best = ForecastSohProbe.load_from_checkpoint(
            pretrained_mae_ckpt,
            tokenizer=pretrained_mae_model.tokenizer,
            encoder=pretrained_mae_model.encoder,
            map_location=device,
        )
        pretrained_mae_best.to(device)
        test_results = trainer.test(pretrained_mae_best, dataloaders=dm.test_dataloader())
        comparison[variant] = do_bench(pretrained_mae_best, dm, device, variant, task, variant_dir)
        comparison[variant]["best/val_rmse"] = float(trainer.checkpoint_callback.best_model_score.detach().cpu())
        comparison[variant]["fit/current_epoch"] = float(trainer.current_epoch)
        if test_results:
            comparison[variant]["lightning/test_rmse"] = float(test_results[0].get("test/rmse", float("nan")))
            comparison[variant]["lightning/test_mae"] = float(test_results[0].get("test/mae", float("nan")))
        best_paths[variant] = pretrained_mae_ckpt

    # Variant별 scalar 비교 표와 그림 저장
    variants = list(comparison)
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
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(variants)))
        for row, key in enumerate(plot_keys):
            ax = axes[row, 0]
            values = [comparison[variant].get(key, np.nan) for variant in variants]
            ax.bar(x, values, color=colors)
            ax.set_xticks(x, variants, rotation=20, ha="right")
            ax.set_title(key)
            ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(report_dir / "variant_comparison.png", dpi=150, bbox_inches="tight")
        if task is not None:
            task.get_logger().report_matplotlib_figure("Forecast SoH", "variant_comparison", fig, 0)
        plt.close(fig)

    if task is not None:
        logger = task.get_logger()
        for variant, metrics in comparison.items():
            for key, value in metrics.items():
                logger.report_single_value(f"{variant}/{key}", value)

    print(f"forecast SoH reports saved under {report_dir}")
    print("comparison summary:")
    for variant in variants:
        compact = {key: comparison[variant][key] for key in ["test/rmse", "test/mae"] if key in comparison[variant]}
        print(f"  {variant}: {compact}")
    for variant, path in best_paths.items():
        print(f"best probe checkpoint [{variant}]: {path}")


if __name__ == "__main__":
    main()
