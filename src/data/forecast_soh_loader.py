from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from current_soh_loader import load_nasa_soh
from pretrain_loader import FEATURES, PretrainLoader, SeriesRecord


class ForecastSohDataset(Dataset):
    """Pairs a current NASA cycle segment with a future-cycle SoH label."""

    def __init__(
        self,
        records: list[SeriesRecord],
        soh_by_path: dict[str, dict[int, float]],
        operations: tuple[str, ...],
        max_horizon: int | None,
        horizon_stride: int,
        min_segment_len: int | None = None,
    ) -> None:
        self.records = records
        self.soh_by_path = soh_by_path
        self.operations = set(operations)
        self.max_horizon = max_horizon
        self.horizon_stride = max(1, horizon_stride)
        self.min_segment_len = min_segment_len or 1
        self._cache: dict[int, np.ndarray] = {}
        self.index = self._build_index()

    def _build_index(self) -> list[dict]:
        current_cycles: list[dict] = []

        for rec_idx, record in enumerate(self.records):
            if record.dataset != "nasa":
                continue

            cycle_soh = self.soh_by_path.get(record.raw_path, {})
            if not cycle_soh:
                continue

            cycles: dict[int, dict] = {}
            for segment in record.metadata.get("segments", []):
                operation = str(segment.get("operation", ""))
                if self.operations and operation not in self.operations:
                    continue

                start = int(segment["start"])
                length = int(segment["length"])
                if length < self.min_segment_len:
                    continue

                cycle_idx = int(segment["cycle_index"])
                if cycle_idx not in cycle_soh:
                    continue

                item = cycles.setdefault(
                    cycle_idx,
                    {
                        "rec_idx": rec_idx,
                        "raw_path": record.raw_path,
                        "battery_id": record.metadata.get("battery_id", record.group_id),
                        "current_cycle": cycle_idx,
                        "segments": [],
                        "operations": set(),
                    },
                )
                item["segments"].append({"start": start, "length": length, "operation": operation})
                item["operations"].add(operation)

            current_cycles.extend(cycles.values())

        pairs: list[dict] = []
        for item in current_cycles:
            cycle_soh = self.soh_by_path.get(str(item["raw_path"]), {})
            current_cycle = int(item["current_cycle"])
            future_cycles = [
                cycle
                for cycle in sorted(cycle_soh)
                if cycle > current_cycle
                and (self.max_horizon is None or cycle - current_cycle <= self.max_horizon)
                and (cycle - current_cycle) % self.horizon_stride == 0
            ]
            for target_cycle in future_cycles:
                pairs.append(
                    {
                        **item,
                        "target_cycle": target_cycle,
                        "delta_cycle": target_cycle - current_cycle,
                        "target_soh": float(cycle_soh[target_cycle]),
                        "operations": ",".join(sorted(item["operations"])),
                    }
                )

        pairs.sort(
            key=lambda row: (
                str(row["battery_id"]),
                int(row["current_cycle"]),
                int(row["target_cycle"]),
            )
        )
        return pairs

    def __len__(self) -> int:
        return len(self.index)

    def _load_array(self, rec_idx: int) -> np.ndarray:
        if rec_idx not in self._cache:
            path = self.records[rec_idx].processed_path
            self._cache[rec_idx] = pd.read_parquet(path, columns=list(FEATURES)).to_numpy(dtype=np.float32)
        return self._cache[rec_idx]

    def __getitem__(self, idx: int) -> dict:
        item = self.index[idx]
        rec_idx = int(item["rec_idx"])
        values = self._load_array(rec_idx)
        segments = [
            torch.from_numpy(values[seg["start"] : seg["start"] + seg["length"]].T.copy())
            for seg in item["segments"]
        ]

        return {
            "segments": segments,
            "target_soh": torch.tensor(float(item["target_soh"]), dtype=torch.float32),
            "current_cycle": torch.tensor(int(item["current_cycle"]), dtype=torch.float32),
            "target_cycle": torch.tensor(int(item["target_cycle"]), dtype=torch.float32),
            "delta_cycle": torch.tensor(int(item["delta_cycle"]), dtype=torch.float32),
            "num_segments": torch.tensor(len(segments), dtype=torch.long),
            "battery_id": str(item["battery_id"]),
            "operations": str(item["operations"]),
        }


def collate_forecast_soh(batch: list[dict]) -> dict:
    return {
        "segments": [item["segments"] for item in batch],
        "target_soh": torch.stack([item["target_soh"] for item in batch]),
        "current_cycle": torch.stack([item["current_cycle"] for item in batch]),
        "target_cycle": torch.stack([item["target_cycle"] for item in batch]),
        "delta_cycle": torch.stack([item["delta_cycle"] for item in batch]),
        "num_segments": torch.stack([item["num_segments"] for item in batch]),
        "battery_id": [item["battery_id"] for item in batch],
        "operations": [item["operations"] for item in batch],
    }


class ForecastSohLoader(L.LightningDataModule):
    """DataModule for x[current cycle], delta_cycle -> future cycle SoH."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        seq_len: int = 320,
        batch_size: int = 64,
        num_workers: int = 4,
        stride: int | None = None,
        processed_dir_name: str = "processed",
        operations: str = "discharge",
        max_horizon: int | None = None,
        horizon_stride: int = 1,
        min_segment_len: int | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.processed_dir_name = processed_dir_name
        self.operations = tuple(part.strip() for part in str(operations).split(",") if part.strip())
        self.max_horizon = max_horizon
        self.horizon_stride = horizon_stride
        self.min_segment_len = min_segment_len

        self.pretrain_loader: PretrainLoader | None = None
        self.records: list[SeriesRecord] = []
        self.train_ds: ForecastSohDataset | None = None
        self.val_ds: ForecastSohDataset | None = None
        self.test_ds: ForecastSohDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self.pretrain_loader = PretrainLoader(
            data_dir=self.data_dir,
            datasets="nasa",
            seq_len=self.seq_len,
            max_gap=0,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            stride=self.stride,
            processed_dir_name=self.processed_dir_name,
        )
        self.pretrain_loader.setup(stage)
        self.records = self.pretrain_loader.records

        soh_by_path = load_nasa_soh(self.records)
        self.train_ds = self._dataset_for_split("train", soh_by_path)
        self.val_ds = self._dataset_for_split("val", soh_by_path)
        self.test_ds = self._dataset_for_split("test", soh_by_path)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.train_ds), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.val_ds), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.test_ds), shuffle=False)

    def _dataset_for_split(
        self,
        split: str,
        soh_by_path: dict[str, dict[int, float]],
    ) -> ForecastSohDataset:
        records = [record for record in self.records if record.split == split]
        dataset = ForecastSohDataset(
            records=records,
            soh_by_path=soh_by_path,
            operations=self.operations,
            max_horizon=self.max_horizon,
            horizon_stride=self.horizon_stride,
            min_segment_len=self.min_segment_len,
        )
        if len(dataset) == 0:
            raise RuntimeError(f"No ForecastSohDataset samples for split={split!r}")
        return dataset

    def _loader(self, dataset: ForecastSohDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_forecast_soh,
        )

    @staticmethod
    def _require_dataset(dataset: ForecastSohDataset | None) -> ForecastSohDataset:
        if dataset is None:
            raise RuntimeError("ForecastSohLoader.setup() must be called before requesting dataloaders")
        return dataset
