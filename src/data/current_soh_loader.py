from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pretrain_loader import FEATURES, PretrainLoader, SeriesRecord

NASA_NOMINAL_CAPACITY_AH = 2.0
SOH_PERCENT_SCALE = 100.0


def load_nasa_soh(records: list[SeriesRecord]) -> dict[str, dict[int, float]]:
    """Return {raw_path: {cycle_index: soh_percent}} from NASA discharge capacity."""
    import scipy.io as sio

    paths = sorted({record.raw_path for record in records if record.dataset == "nasa"})
    soh_by_path: dict[str, dict[int, float]] = {}

    for raw_path in paths:
        stem = Path(raw_path).stem
        try:
            mat = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
            cycles = np.atleast_1d(mat[stem].cycle)
        except Exception:
            soh_by_path[raw_path] = {}
            continue

        capacities: dict[int, float] = {}
        for cycle_idx, cycle in enumerate(cycles):
            if str(cycle.type) != "discharge":
                continue
            try:
                capacities[cycle_idx] = float(np.atleast_1d(cycle.data.Capacity)[-1])
            except Exception:
                pass

        if not capacities:
            soh_by_path[raw_path] = {}
            continue

        soh_by_path[raw_path] = {
            cycle_idx: SOH_PERCENT_SCALE * capacity / NASA_NOMINAL_CAPACITY_AH
            for cycle_idx, capacity in capacities.items()
            if capacity > 0
        }

    return soh_by_path


class CurrentSohDataset(Dataset):
    """One sample per NASA cycle, represented by all selected operation segments."""

    def __init__(
        self,
        records: list[SeriesRecord],
        seq_len: int,
        stride: int | None,
        soh_by_path: dict[str, dict[int, float]],
        operations: tuple[str, ...],
        min_segment_len: int | None = None,
    ) -> None:
        self.records = records
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.soh_by_path = soh_by_path
        self.operations = set(operations)
        self.min_segment_len = min_segment_len or seq_len
        self._cache: dict[int, np.ndarray] = {}
        self.index = self._build_index()

    def _build_index(self) -> list[dict]:
        cycles: dict[tuple[int, int], dict] = {}

        for rec_idx, record in enumerate(self.records):
            if record.dataset != "nasa":
                continue

            cycle_soh = self.soh_by_path.get(record.raw_path, {})
            if not cycle_soh:
                continue

            for segment in record.metadata.get("segments", []):
                operation = str(segment.get("operation", ""))
                if self.operations and operation not in self.operations:
                    continue

                cycle_idx = int(segment["cycle_index"])
                label_cycle = cycle_idx if cycle_idx in cycle_soh else min(cycle_soh, key=lambda c: abs(c - cycle_idx))
                soh = float(cycle_soh[label_cycle])

                start = int(segment["start"])
                length = int(segment["length"])
                if length < self.min_segment_len:
                    continue

                key = (rec_idx, cycle_idx)
                item = cycles.setdefault(
                    key,
                    {
                        "rec_idx": rec_idx,
                        "cycle_index": cycle_idx,
                        "label_cycle_index": label_cycle,
                        "soh": soh,
                        "segments": [],
                        "operations": set(),
                        "battery_id": record.metadata.get("battery_id", record.group_id),
                    },
                )
                item["operations"].add(operation)
                item["segments"].append({"start": start, "length": length, "operation": operation})

        index = [item for item in cycles.values() if item["segments"]]
        index.sort(key=lambda item: (str(item["battery_id"]), int(item["cycle_index"])))
        for item in index:
            item["operations"] = ",".join(sorted(item["operations"]))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def _load_array(self, rec_idx: int) -> np.ndarray:
        if rec_idx not in self._cache:
            path = self.records[rec_idx].processed_path
            self._cache[rec_idx] = pd.read_parquet(path, columns=list(FEATURES)).to_numpy(dtype=np.float32)
        return self._cache[rec_idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.index[idx]
        rec_idx = int(item["rec_idx"])
        values = self._load_array(rec_idx)
        segments = [
            torch.from_numpy(values[seg["start"] : seg["start"] + seg["length"]].T.copy())
            for seg in item["segments"]
        ]

        return {
            "segments": segments,
            "soh": torch.tensor(float(item["soh"]), dtype=torch.float32),
            "cycle_index": torch.tensor(int(item["cycle_index"]), dtype=torch.float32),
            "label_cycle_index": torch.tensor(int(item["label_cycle_index"]), dtype=torch.float32),
            "num_segments": torch.tensor(len(segments), dtype=torch.long),
            "battery_id": str(item["battery_id"]),
            "operations": str(item["operations"]),
        }


def collate_cycle_soh(batch: list[dict]) -> dict:
    return {
        "segments": [item["segments"] for item in batch],
        "soh": torch.stack([item["soh"] for item in batch]),
        "cycle_index": torch.stack([item["cycle_index"] for item in batch]),
        "label_cycle_index": torch.stack([item["label_cycle_index"] for item in batch]),
        "num_segments": torch.stack([item["num_segments"] for item in batch]),
        "battery_id": [item["battery_id"] for item in batch],
        "operations": [item["operations"] for item in batch],
    }


class CurrentSohLoader(L.LightningDataModule):
    """DataModule for x[cycle window] -> current cycle SoH."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        seq_len: int = 320,
        batch_size: int = 64,
        num_workers: int = 4,
        stride: int | None = None,
        processed_dir_name: str = "processed",
        operations: str = "charge,discharge",
        min_segment_len: int | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.processed_dir_name = processed_dir_name
        self.min_segment_len = min_segment_len
        self.operations = tuple(
            part.strip()
            for part in str(operations).split(",")
            if part.strip()
        )

        self.pretrain_loader: PretrainLoader | None = None
        self.records: list[SeriesRecord] = []
        self.train_ds: CurrentSohDataset | None = None
        self.val_ds: CurrentSohDataset | None = None
        self.test_ds: CurrentSohDataset | None = None

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
    ) -> CurrentSohDataset:
        records = [record for record in self.records if record.split == split]
        dataset = CurrentSohDataset(
            records=records,
            seq_len=self.seq_len,
            stride=self.stride,
            soh_by_path=soh_by_path,
            operations=self.operations,
            min_segment_len=self.min_segment_len,
        )
        if len(dataset) == 0:
            raise RuntimeError(f"No CurrentSohDataset samples for split={split!r}")
        return dataset

    def _loader(self, dataset: CurrentSohDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_cycle_soh,
        )

    @staticmethod
    def _require_dataset(dataset: CurrentSohDataset | None) -> CurrentSohDataset:
        if dataset is None:
            raise RuntimeError("CurrentSohLoader.setup() must be called before requesting dataloaders")
        return dataset
