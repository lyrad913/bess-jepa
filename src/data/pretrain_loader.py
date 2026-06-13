from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


FEATURES = ("voltage", "current", "temperature")


@dataclass(frozen=True)
class SeriesRecord:
    dataset: str
    series_id: str
    group_id: str
    split: str
    raw_path: str
    processed_path: str
    n_points: int
    dt_seconds: float
    metadata: dict[str, Any]


class WindowPairDataset(Dataset):
    """Context/target window pairs loaded from preprocessed parquet files."""

    def __init__(
        self,
        records: list[SeriesRecord],
        seq_len: int,
        max_gap: int,
        stride: int | None,
        seed: int,
        random_gap: bool,
    ) -> None:
        self.records = records
        self.seq_len = seq_len
        self.max_gap = max_gap
        self.stride = stride or seq_len
        self.seed = seed
        self.random_gap = random_gap
        self._cache: dict[int, np.ndarray] = {}
        self.index: list[tuple[int, int, int]] = self._build_index()

    def _build_index(self) -> list[tuple[int, int, int]]:
        index: list[tuple[int, int, int]] = []
        for rec_idx, record in enumerate(self.records):
            segments = record.metadata.get("segments") or [{"start": 0, "length": record.n_points}]
            for segment in segments:
                segment_start = int(segment["start"])
                segment_end = segment_start + int(segment["length"])
                last_start = segment_end - 2 * self.seq_len
                if last_start < segment_start:
                    continue
                for start in range(segment_start, last_start + 1, self.stride):
                    max_feasible_gap = min(
                        self.max_gap,
                        (segment_end - start) // self.seq_len - 2,
                    )
                    index.append((rec_idx, start, max_feasible_gap))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def _load_array(self, rec_idx: int) -> np.ndarray:
        if rec_idx not in self._cache:
            path = self.records[rec_idx].processed_path
            self._cache[rec_idx] = pd.read_parquet(path, columns=list(FEATURES)).to_numpy(dtype=np.float32)
        return self._cache[rec_idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec_idx, start, max_feasible_gap = self.index[idx]
        gap = self._sample_gap(idx, max_feasible_gap)
        values = self._load_array(rec_idx)
        target_start = start + (gap + 1) * self.seq_len

        x = values[start : start + self.seq_len].T
        y = values[target_start : target_start + self.seq_len].T
        return {
            "x": torch.from_numpy(x.copy()),
            "y": torch.from_numpy(y.copy()),
            "gap": torch.tensor(gap, dtype=torch.float32),
        }

    def _sample_gap(self, idx: int, max_feasible_gap: int) -> int:
        if max_feasible_gap <= 0:
            return 0
        if self.random_gap:
            return int(torch.randint(max_feasible_gap + 1, size=()).item())
        rng = np.random.default_rng(self.seed + idx)
        return int(rng.integers(0, max_feasible_gap + 1))

class PretrainLoader(L.LightningDataModule):
    """
    Load processed MATR/NASA parquet files for JEPA pretraining.

    Run src/scripts/process_data.py before training. Gap semantics:
    gap=0 predicts the immediately following window, while gap=k skips k
    full windows between context and target.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        datasets: str = "mixed",
        seq_len: int = 96,
        max_gap: int = 10,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 64,
        num_workers: int = 4,
        stride: int | None = None,
        seed: int = 42,
        force_reprocess: bool = False,
        processed_dir_name: str = "processed",
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.datasets = datasets
        self.seq_len = seq_len
        self.max_gap = max_gap
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.seed = seed
        self.force_reprocess = force_reprocess
        self.processed_dir_name = processed_dir_name

        self.processed_dir = self._resolve_processed_dir()
        self.manifest_path = self.processed_dir / "manifest.json"
        self.scaler_path = self.processed_dir / "scaler.json"

        self.records: list[SeriesRecord] = []
        self.scaler: dict[str, list[float]] | None = None
        self.train_ds: WindowPairDataset | None = None
        self.val_ds: WindowPairDataset | None = None
        self.test_ds: WindowPairDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.force_reprocess:
            raise RuntimeError("Preprocessing is handled by src/scripts/process_data.py")
        if not self.manifest_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Missing processed data under {self.processed_dir}. "
                "Run `uv run python src/scripts/process_data.py` first."
            )

        manifest = json.loads(self.manifest_path.read_text())
        processed_datasets = manifest.get("config", {}).get("datasets")
        selected = self._selected_datasets()
        if processed_datasets is not None:
            processed_selected = self._parse_datasets(processed_datasets)
        else:
            processed_selected = selected
        if not selected.issubset(processed_selected):
            raise RuntimeError(
                f"Processed data was built with datasets={processed_datasets!r}, "
                f"but loader requested datasets={self.datasets!r}. "
                "Rerun src/scripts/process_data.py with a dataset set that includes the requested data."
            )
        self.records = [
            SeriesRecord(**item)
            for item in manifest["records"]
            if item["dataset"] in selected
        ]
        self.scaler = self._applied_scaler(json.loads(self.scaler_path.read_text()))

        self.train_ds = self._dataset_for_split("train")
        self.val_ds = self._dataset_for_split("val")
        self.test_ds = self._dataset_for_split("test")

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.train_ds), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.val_ds), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.test_ds), shuffle=False)

    def inverse_transform(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Return normalized values to original voltage/current/temperature scale."""
        if self.scaler is None:
            self.scaler = self._applied_scaler(json.loads(self.scaler_path.read_text()))
        scaler = self._inverse_scaler()
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)

        if isinstance(values, torch.Tensor):
            mean_t = torch.as_tensor(mean, device=values.device, dtype=values.dtype)
            std_t = torch.as_tensor(std, device=values.device, dtype=values.dtype)
            if values.shape[-2] == len(FEATURES):
                return values * std_t[:, None] + mean_t[:, None]
            return values * std_t + mean_t

        if values.shape[-2] == len(FEATURES):
            return values * std[:, None] + mean[:, None]
        return values * std + mean

    def _dataset_for_split(self, split: str) -> WindowPairDataset:
        records = [record for record in self.records if record.split == split]
        return WindowPairDataset(
            records=records,
            seq_len=self.seq_len,
            max_gap=self.max_gap,
            stride=self.stride,
            seed=self.seed,
            random_gap=split == "train",
        )

    def _loader(self, dataset: WindowPairDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def _resolve_processed_dir(self) -> Path:
        if self.processed_dir_name == "half_processed":
            return self.data_dir / "half_processed"
        if self.processed_dir_name not in {"processed", "processed "}:
            return self.data_dir / self.processed_dir_name
        for path in (self.data_dir / "processed ", self.data_dir / "processed"):
            if path.exists():
                return path
        return self.data_dir / "processed"

    def _selected_datasets(self) -> set[str]:
        return self._parse_datasets(self.datasets)

    @staticmethod
    def _parse_datasets(datasets: str) -> set[str]:
        if datasets == "mixed":
            return {"matr", "nasa"}
        selected = {part.strip().lower() for part in str(datasets).split(",")}
        valid = {"matr", "nasa"}
        unknown = selected - valid
        if unknown:
            raise ValueError(f"Unknown pretrain datasets: {sorted(unknown)}")
        return selected

    @staticmethod
    def _applied_scaler(scaler: dict[str, Any]) -> dict[str, list[float]]:
        if "applied" in scaler:
            return scaler["applied"]
        return scaler

    def _inverse_scaler(self) -> dict[str, Any]:
        if self.scaler is None:
            raise RuntimeError("PretrainLoader.setup() must be called before inverse_transform")
        if "mean" in self.scaler and "std" in self.scaler:
            return self.scaler
        by_dataset = self.scaler.get("by_dataset", {})
        selected = self._selected_datasets()
        if len(selected) == 1:
            return by_dataset[next(iter(selected))]
        raise ValueError(
            "inverse_transform for mixed per-dataset scaling needs a dataset-specific scaler. "
            "Call it on a single-dataset loader or use scaler['by_dataset'][dataset] explicitly."
        )

    def _require_scaler(self) -> dict[str, list[float]]:
        if self.scaler is None:
            raise RuntimeError("PretrainLoader.setup() must be called before creating datasets")
        return self.scaler

    @staticmethod
    def _require_dataset(dataset: WindowPairDataset | None) -> WindowPairDataset:
        if dataset is None:
            raise RuntimeError("PretrainLoader.setup() must be called before requesting dataloaders")
        return dataset
