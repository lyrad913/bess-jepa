"""
Frozen JEPA encoder + linear decoder로 다음 V/I/T 윈도우를 예측한다.

실험 1: x[0:320] -> x_hat[320:640] one-step forecast
실험 2: 예측값을 다시 입력으로 넣는 autoregressive rollout
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
from sub_models import Decoder


class ForecastProbe(L.LightningModule):
    """JEPA tokenizer/encoder는 얼리고 Decoder만 학습하는 linear probe."""

    def __init__(
        self,
        jepa: JEPA,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
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
        n_patches = (hp.seq_len - hp.patch_size) // hp.strides + 1
        self.decoder = Decoder(
            n_patches=n_patches,
            embed_dim=hp.embed_dim,
            n_features=hp.num_channels,
            seq_len=hp.seq_len,
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


def make_task(cfg: DictConfig):
    if not cfg.report.use_clearml:
        return None
    try:
        from clearml import Task

        task = Task.init(
            project_name="BESS-JEPA",
            task_name="experiment_forecast",
            reuse_last_task_id=False,
        )
        task.connect(OmegaConf.to_container(cfg, resolve=True))
        return task
    except Exception as e:
        print(f"ClearML 사용 불가: {e}")
        return None


def to_original_scale(dm: ForecastLoader, x: torch.Tensor) -> np.ndarray:
    try:
        return dm.inverse_transform(x.cpu()).numpy()
    except Exception:
        return x.cpu().numpy()


def collect_one_step_examples(
    model: ForecastProbe,
    loader,
    dm: ForecastLoader,
    device: str,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = batch["x"].to(device)
            y = batch["y"]
            y_hat = model(x)
            preds.append(to_original_scale(dm, y_hat))
            targets.append(to_original_scale(dm, y))
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def rollout(
    model: ForecastProbe,
    x: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    preds = []
    current = x
    for _ in range(steps):
        y_hat = model(current)
        preds.append(y_hat)
        current = y_hat
    return torch.stack(preds, dim=1)


def evaluate_rollout(
    model: ForecastProbe,
    loader,
    dm: ForecastLoader,
    device: str,
    steps: int | None,
    max_plot_batches: int,
) -> tuple[dict[str, float], dict[str, list[float]], np.ndarray, np.ndarray, int]:
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
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device)
            target = batch["rollout_y"].to(device)
            eval_steps = steps or target.shape[1]
            target = target[:, :eval_steps]
            pred = rollout(model, x, eval_steps)

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

            if batch_idx < max_plot_batches and not plot_preds:
                b = pred.shape[0]
                flat_pred = pred.reshape(b * eval_steps, len(FEATURES), -1)
                flat_target = target.reshape(b * eval_steps, len(FEATURES), -1)
                plot_preds.append(to_original_scale(dm, flat_pred).reshape(b, eval_steps, len(FEATURES), -1))
                plot_targets.append(to_original_scale(dm, flat_target).reshape(b, eval_steps, len(FEATURES), -1))

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

    return metrics, rollout_stats, np.concatenate(plot_preds, axis=0), np.concatenate(plot_targets, axis=0), actual_steps


def plot_one_step(pred: np.ndarray, target: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    return fig


def plot_rollout(pred: np.ndarray, target: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = pred.shape[1]
    sample = 0
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(12, 7), squeeze=False)
    t = np.arange(steps * pred.shape[-1])
    for col, name in enumerate(FEATURES):
        ax = axes[col, 0]
        y_true = target[sample, :, col].reshape(-1)
        y_pred = pred[sample, :, col].reshape(-1)
        ax.plot(t, y_true, label="target", linewidth=1.5)
        ax.plot(t, y_pred, label="rollout", linewidth=1.2, alpha=0.8)
        for step in range(1, steps):
            ax.axvline(step * pred.shape[-1], color="0.85", linewidth=0.8)
        ax.set_title(f"rollout - {name}")
        if col == 0:
            ax.legend()
    fig.tight_layout()
    return fig


def plot_cumulative_mse(rollout_metrics: dict[str, float], steps: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, steps + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    overall = [rollout_metrics[f"rollout/cumulative_step_{i}_mse"] for i in x]
    ax.plot(x, overall, label="overall", color="black", linewidth=2.0)

    colors = {
        "voltage": "tab:blue",
        "current": "tab:orange",
        "temperature": "tab:green",
    }
    for name in FEATURES:
        values = [rollout_metrics[f"rollout/cumulative_step_{i}_{name}_mse"] for i in x]
        ax.plot(x, values, label=name, color=colors.get(name), linewidth=1.5, alpha=0.9)

    ax.set_title("Rollout Cumulative MSE")
    ax.set_xlabel("# Steps")
    ax.set_ylabel("Cumulative MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_step_mse_band(rollout_stats: dict[str, list[float]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    ax.set_title("Rollout Step MSE")
    ax.set_xlabel("# Steps")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    count_ax = ax.twinx()
    count_ax.plot(x, rollout_stats["count"], color="0.45", linestyle="--", linewidth=1.2, label="sample count")
    count_ax.set_ylabel("# Samples")
    count_ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_channel_step_mse_bands(rollout_stats: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(8, 7), sharex=True, squeeze=False)
    colors = {
        "voltage": "tab:blue",
        "current": "tab:orange",
        "temperature": "tab:green",
    }
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

    axes[-1, 0].set_xlabel("# Steps")
    fig.suptitle("Rollout Step MSE by Channel", y=0.995)
    fig.tight_layout()
    return fig


def plot_rollout_sample_count(rollout_stats: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = np.asarray(rollout_stats["count"], dtype=np.int32)
    x = np.arange(1, len(count) + 1)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.step(x, count, where="mid", color="black", linewidth=1.8)
    ax.set_title("Rollout Sample Count by Step")
    ax.set_xlabel("# Steps")
    ax.set_ylabel("# Samples")
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


def report_scalars(task, metrics: dict[str, float]) -> None:
    if task is None:
        return
    logger = task.get_logger()
    for key, value in metrics.items():
        title, series = key.split("/", 1)
        logger.report_scalar(title, series, value=value, iteration=0)


def report_summary(task, metrics: dict[str, float]) -> None:
    if task is None:
        return
    logger = task.get_logger()
    for key, value in metrics.items():
        logger.report_single_value(key, value)


def make_final_summary(
    test_results: list[dict[str, float]],
    rollout_metrics: dict[str, float],
    rollout_steps: int,
) -> dict[str, float]:
    summary = {}
    if test_results:
        for key, value in test_results[0].items():
            summary[f"one_step/{key}"] = float(value)

    step_mse = [rollout_metrics[f"rollout/step_{i}_mse"] for i in range(1, rollout_steps + 1)]
    step_mae = [rollout_metrics[f"rollout/step_{i}_mae"] for i in range(1, rollout_steps + 1)]
    summary["rollout/mean_mse"] = float(np.mean(step_mse))
    summary["rollout/mean_mae"] = float(np.mean(step_mae))
    summary["rollout/final_step_mse"] = float(step_mse[-1])
    summary["rollout/final_step_mae"] = float(step_mae[-1])
    summary["rollout/final_cumulative_mse"] = float(
        rollout_metrics[f"rollout/cumulative_step_{rollout_steps}_mse"]
    )

    for name in FEATURES:
        ch_mse = [rollout_metrics[f"rollout/step_{i}_{name}_mse"] for i in range(1, rollout_steps + 1)]
        ch_mae = [rollout_metrics[f"rollout/step_{i}_{name}_mae"] for i in range(1, rollout_steps + 1)]
        summary[f"rollout/{name}_mean_mse"] = float(np.mean(ch_mse))
        summary[f"rollout/{name}_mean_mae"] = float(np.mean(ch_mae))
        summary[f"rollout/{name}_final_step_mse"] = float(ch_mse[-1])
        summary[f"rollout/{name}_final_step_mae"] = float(ch_mae[-1])
        summary[f"rollout/{name}_final_cumulative_mse"] = float(
            rollout_metrics[f"rollout/cumulative_step_{rollout_steps}_{name}_mse"]
        )

    return summary


@hydra.main(config_path="../../../config", config_name="forecast", version_base=None)
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
            f"Forecast seq_len={seq_len} must match checkpoint seq_len={int(jepa.hparams.seq_len)}"
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

    model = ForecastProbe(jepa, lr=cfg.probe.lr, weight_decay=cfg.probe.weight_decay)
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[
            EarlyStopping(monitor="val/mse", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(monitor="val/mse", mode="min", save_top_k=1, filename="forecast-probe-best"),
        ],
    )
    trainer.fit(model, dm)
    test_results = trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path="best")

    if trainer.checkpoint_callback.best_model_path:
        model = ForecastProbe.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            jepa=jepa,
            map_location=device,
        )
    model.to(device)

    pred, target = collect_one_step_examples(
        model=model,
        loader=dm.test_dataloader(),
        dm=dm,
        device=device,
        max_batches=cfg.report.plot_batches,
    )
    fig = plot_one_step(pred, target)
    report_figure(task, "Forecast", "one_step_examples", fig, report_dir / "one_step_examples.png")

    configured_rollout_steps = cfg.data.rollout_steps
    configured_rollout_steps = None if configured_rollout_steps is None else int(configured_rollout_steps)
    rollout_metrics, rollout_stats, rollout_pred, rollout_target, rollout_steps = evaluate_rollout(
        model=model,
        loader=dm.rollout_dataloader(),
        dm=dm,
        device=device,
        steps=configured_rollout_steps,
        max_plot_batches=cfg.report.plot_batches,
    )
    report_scalars(task, rollout_metrics)
    (report_dir / "rollout_metrics.json").write_text(json.dumps(rollout_metrics, indent=2))
    (report_dir / "rollout_step_mse_stats.json").write_text(json.dumps(rollout_stats, indent=2))

    fig = plot_rollout(rollout_pred, rollout_target)
    report_figure(task, "Forecast", "rollout_example", fig, report_dir / "rollout_example.png")

    fig = plot_cumulative_mse(rollout_metrics, rollout_steps)
    report_figure(task, "Forecast", "cumulative_mse", fig, report_dir / "cumulative_mse.png")

    fig = plot_step_mse_band(rollout_stats)
    report_figure(task, "Forecast", "step_mse_band", fig, report_dir / "step_mse_band.png")

    fig = plot_channel_step_mse_bands(rollout_stats)
    report_figure(task, "Forecast", "channel_step_mse_bands", fig, report_dir / "channel_step_mse_bands.png")

    fig = plot_rollout_sample_count(rollout_stats)
    report_figure(task, "Forecast", "sample_count", fig, report_dir / "sample_count.png")

    final_summary = make_final_summary(test_results, rollout_metrics, rollout_steps)
    report_summary(task, final_summary)
    (report_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2))

    print(f"forecast reports saved under {report_dir}")
    print("final summary:")
    for key, value in final_summary.items():
        print(f"  {key}: {value:.6f}")
    if trainer.checkpoint_callback.best_model_path:
        print(f"best probe checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
