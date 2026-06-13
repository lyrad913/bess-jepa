from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pretrain_loader import FEATURES, PretrainLoader, SeriesRecord


class ForecastWindowDataset(Dataset):
    """Single-window forecasting dataset.

    Each sample is:
      x: context window                      (C, T)
      y: immediately following target window (C, T)
      rollout_y: next N target windows       (N, C, T)
    """

    def __init__(
        self,
        records: list[SeriesRecord],
        seq_len: int,
        stride: int | None,
        rollout_steps: int | None,
        rollout_step_size: int | None = None,
        require_full_rollout: bool = False,
    ) -> None:
        self.records = records
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.rollout_steps = max(1, rollout_steps) if rollout_steps is not None else None
        self.rollout_step_size = rollout_step_size or seq_len
        self.require_full_rollout = require_full_rollout
        self._cache: dict[int, np.ndarray] = {}
        self.index = self._build_index()

    def _build_index(self) -> list[tuple[int, int, int]]:
        index: list[tuple[int, int, int]] = []
        if self.require_full_rollout and self.rollout_steps is None:
            needed = self.seq_len + self.rollout_step_size
        else:
            steps = self.rollout_steps or 1
            needed = self.seq_len + steps * self.rollout_step_size if self.require_full_rollout else 2 * self.seq_len

        for rec_idx, record in enumerate(self.records):
            segments = record.metadata.get("segments") or [{"start": 0, "length": record.n_points}]
            for segment in segments:
                segment_start = int(segment["start"])
                segment_end = segment_start + int(segment["length"])
                last_start = segment_end - needed
                if last_start < segment_start:
                    continue
                if self.require_full_rollout and self.rollout_steps is None:
                    index.append((rec_idx, segment_start, segment_end))
                    continue
                for start in range(segment_start, last_start + 1, self.stride):
                    index.append((rec_idx, start, segment_end))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def _load_array(self, rec_idx: int) -> np.ndarray:
        if rec_idx not in self._cache:
            path = self.records[rec_idx].processed_path
            self._cache[rec_idx] = pd.read_parquet(path, columns=list(FEATURES)).to_numpy(dtype=np.float32)
        return self._cache[rec_idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec_idx, start, segment_end = self.index[idx]
        values = self._load_array(rec_idx)

        x = values[start : start + self.seq_len].T
        y_start = start + self.seq_len
        y = values[y_start : y_start + self.seq_len].T

        sample = {
            "x": torch.from_numpy(x.copy()),
            "y": torch.from_numpy(y.copy()),
        }
        if self.require_full_rollout:
            rollout_steps = self.rollout_steps
            if rollout_steps is None:
                rollout_steps = (segment_end - y_start) // self.rollout_step_size
            rollout = []
            for step in range(rollout_steps):
                target_start = y_start + step * self.rollout_step_size
                rollout.append(values[target_start : target_start + self.rollout_step_size].T)
            sample["rollout_y"] = torch.from_numpy(np.stack(rollout, axis=0).copy())
        return sample


class ForecastLoader(L.LightningDataModule):
    """Forecast datamodule built on the same processed files as JEPA pretraining."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        datasets: str = "mixed",
        seq_len: int = 320,
        batch_size: int = 64,
        num_workers: int = 4,
        stride: int | None = None,
        rollout_steps: int | None = 8,
        rollout_step_size: int | None = None,
        processed_dir_name: str = "processed",
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.datasets = datasets
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.rollout_steps = rollout_steps
        self.rollout_step_size = rollout_step_size
        self.processed_dir_name = processed_dir_name

        self.pretrain_loader: PretrainLoader | None = None
        self.records: list[SeriesRecord] = []
        self.train_ds: ForecastWindowDataset | None = None
        self.val_ds: ForecastWindowDataset | None = None
        self.test_ds: ForecastWindowDataset | None = None
        self.rollout_test_ds: ForecastWindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self.pretrain_loader = PretrainLoader(
            data_dir=self.data_dir,
            datasets=self.datasets,
            seq_len=self.seq_len,
            max_gap=0,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            stride=self.stride,
            processed_dir_name=self.processed_dir_name,
        )
        self.pretrain_loader.setup(stage)
        self.records = self.pretrain_loader.records

        self.train_ds = self._dataset_for_split("train")
        self.val_ds = self._dataset_for_split("val")
        self.test_ds = self._dataset_for_split("test")
        self.rollout_test_ds = self._dataset_for_split("test", require_full_rollout=True)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.train_ds), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.val_ds), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.test_ds), shuffle=False)

    def rollout_dataloader(self) -> DataLoader:
        batch_size = 1 if self.rollout_steps is None else self.batch_size
        return self._loader(self._require_dataset(self.rollout_test_ds), shuffle=False, batch_size=batch_size)

    def inverse_transform(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.pretrain_loader is None:
            raise RuntimeError("ForecastLoader.setup() must be called before inverse_transform")
        return self.pretrain_loader.inverse_transform(values)

    def _dataset_for_split(self, split: str, require_full_rollout: bool = False) -> ForecastWindowDataset:
        records = [record for record in self.records if record.split == split]
        return ForecastWindowDataset(
            records=records,
            seq_len=self.seq_len,
            stride=self.stride,
            rollout_steps=self.rollout_steps,
            rollout_step_size=self.rollout_step_size,
            require_full_rollout=require_full_rollout,
        )

    def _loader(self, dataset: ForecastWindowDataset, shuffle: bool, batch_size: int | None = None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    @staticmethod
    def _require_dataset(dataset: ForecastWindowDataset | None) -> ForecastWindowDataset:
        if dataset is None:
            raise RuntimeError("ForecastLoader.setup() must be called before requesting dataloaders")
        return dataset
