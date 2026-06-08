from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


FEATURES = ("voltage", "current", "temperature")
PREPROCESS_VERSION = 1


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


def main() -> None:
    args = parse_args()
    process_pretrain_data(
        data_dir=Path(args.data_dir),
        datasets=args.datasets,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        scale=args.scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MATR/NASA data for JEPA pretraining.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--datasets", default="mixed", choices=["matr", "nasa", "mixed"])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", dest="scale", action="store_true", default=True)
    parser.add_argument("--no-scale", dest="scale", action="store_false")
    return parser.parse_args()


def process_pretrain_data(
    data_dir: Path,
    datasets: str = "mixed",
    seq_len: int = 96,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    scale: bool = True,
) -> None:
    processed_dir = resolve_output_dir(data_dir, scale)

    selected = selected_datasets(datasets)
    records: list[SeriesRecord] = []
    if "matr" in selected:
        records.extend(preprocess_matr(data_dir, processed_dir, seq_len, val_fraction, seed))
    if "nasa" in selected:
        records.extend(preprocess_nasa(data_dir, processed_dir, seq_len, val_fraction, test_fraction, seed))

    if not records:
        raise RuntimeError(f"No pretraining series found under {data_dir}")

    scaler = build_scalers(records)
    if scale:
        applied_scaler = select_applied_scalers(scaler, selected)
        scale_processed_files(records, applied_scaler)
    else:
        applied_scaler = identity_scaler(selected)
    scaler["applied"] = applied_scaler
    config = {
        "datasets": datasets,
        "seq_len": seq_len,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "features": list(FEATURES),
        "storage": "parquet",
        "stored_scale": "standardized" if scale else "raw",
        "normalization": "preprocess applies train-only per-channel mean/std" if scale else "none",
        "scaler_selection": "each dataset is scaled by its own train scaler; global scaler is saved for reference",
        "output_dir": str(processed_dir),
        "matr_test_rule": "data/raw/MATR/oed_validation",
        "nasa_split_rule": "battery_id-level split",
        "nasa_excluded_folder": "5. BatteryAgingARC_49_50_51_52",
    }
    manifest = {
        "version": PREPROCESS_VERSION,
        "config_hash": config_hash(config),
        "config": config,
        "records": [asdict(record) for record in records],
        "summary": summarize(records),
    }

    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (processed_dir / "scaler.json").write_text(json.dumps(scaler, indent=2))
    splits_df = pd.DataFrame([asdict(record) for record in records])
    splits_df["metadata"] = splits_df["metadata"].map(json.dumps)
    splits_df.to_csv(processed_dir / "splits.csv", index=False)


def preprocess_matr(
    data_dir: Path,
    processed_dir: Path,
    seq_len: int,
    val_fraction: float,
    seed: int,
) -> list[SeriesRecord]:
    root = data_dir / "raw" / "MATR"
    test_files = sorted(path for path in (root / "oed_validation").glob("*.json"))
    splits = {path: "test" for path in test_files}
    for folder in sorted(root.glob("oed_[0-9]")):
        files = sorted(path for path in folder.glob("*.json") if path.is_file())
        splits.update(split_items(files, lambda path: path.name, val_fraction, 0.0, seed))

    records: list[SeriesRecord] = []
    for path, split in tqdm(splits.items(), desc="MATR files", unit="file"):
        values, dt_seconds = load_matr_json(path, seq_len)
        if values is None:
            continue
        series_id = matr_series_id(path)
        processed_path = processed_dir / "matr" / split / f"{series_id}.parquet"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        values.to_parquet(processed_path, index=False)
        records.append(
            SeriesRecord(
                dataset="matr",
                series_id=series_id,
                group_id=path.stem,
                split=split,
                raw_path=str(path),
                processed_path=str(processed_path),
                n_points=len(values),
                dt_seconds=dt_seconds,
                metadata={
                    "source_folder": path.parent.name,
                    "segments": [{"start": 0, "length": len(values)}],
                },
            )
        )
    return records


def preprocess_nasa(
    data_dir: Path,
    processed_dir: Path,
    seq_len: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[SeriesRecord]:
    root = data_dir / "raw" / "NASA" / "5. Battery Data Set"
    excluded = "5. BatteryAgingARC_49_50_51_52"
    files = sorted(path for path in root.rglob("B*.mat") if excluded not in path.parts)
    splits = split_items(files, lambda path: battery_id(path.stem), val_fraction, test_fraction, seed)

    records: list[SeriesRecord] = []
    for path, split in tqdm(splits.items(), desc="NASA batteries", unit="battery"):
        cycles = load_nasa_cycles(path, seq_len)
        if not cycles:
            continue

        frames: list[pd.DataFrame] = []
        segments: list[dict[str, Any]] = []
        start = 0
        dt_seconds_values: list[float] = []
        for segment_idx, (cycle_idx, values, dt_seconds, op_type, ambient) in enumerate(cycles):
            values = values.copy()
            values.insert(0, "segment_index", segment_idx)
            values.insert(1, "cycle_index", cycle_idx)
            values.insert(2, "operation", op_type)
            values.insert(3, "ambient_temperature", ambient)
            frames.append(values)
            segments.append(
                {
                    "segment_index": segment_idx,
                    "cycle_index": cycle_idx,
                    "operation": op_type,
                    "ambient_temperature": ambient,
                    "start": start,
                    "length": len(values),
                }
            )
            start += len(values)
            dt_seconds_values.append(dt_seconds)

        values = pd.concat(frames, ignore_index=True)
        series_id = nasa_series_id(path)
        processed_path = processed_dir / "nasa" / split / f"{series_id}.parquet"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        values.to_parquet(processed_path, index=False)
        records.append(
            SeriesRecord(
                dataset="nasa",
                series_id=series_id,
                group_id=battery_id(path.stem),
                split=split,
                raw_path=str(path),
                processed_path=str(processed_path),
                n_points=len(values),
                dt_seconds=float(np.median(dt_seconds_values)),
                metadata={
                    "battery_id": battery_id(path.stem),
                    "source_folder": path.parent.name,
                    "segments": segments,
                },
            )
        )
    return records


def load_matr_json(path: Path, seq_len: int) -> tuple[pd.DataFrame | None, float]:
    with path.open() as f:
        data = json.load(f)["raw_data"]
    df = pd.DataFrame(
        {
            "time": data["test_time"],
            "voltage": data["voltage"],
            "current": data["current"],
            "temperature": data["temperature"],
        }
    )
    return regularize_frame(df, seq_len)


def load_nasa_cycles(path: Path, seq_len: int) -> list[tuple[int, pd.DataFrame, float, str, float]]:
    import scipy.io as sio

    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    battery = mat[path.stem]
    cycles = np.atleast_1d(battery.cycle)
    out: list[tuple[int, pd.DataFrame, float, str, float]] = []

    for cycle_idx, cycle in enumerate(cycles):
        op_type = str(cycle.type)
        if op_type not in {"charge", "discharge"}:
            continue
        data = cycle.data
        df = pd.DataFrame(
            {
                "time": np.asarray(data.Time, dtype=np.float64),
                "voltage": np.asarray(data.Voltage_measured, dtype=np.float64),
                "current": np.asarray(data.Current_measured, dtype=np.float64),
                "temperature": np.asarray(data.Temperature_measured, dtype=np.float64),
            }
        )
        values, dt_seconds = regularize_frame(df, seq_len)
        if values is None:
            continue
        out.append((cycle_idx, values, dt_seconds, op_type, float(cycle.ambient_temperature)))
    return out


def regularize_frame(df: pd.DataFrame, seq_len: int) -> tuple[pd.DataFrame | None, float]:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df = df.groupby("time", as_index=False)[list(FEATURES)].mean()
    df[list(FEATURES)] = df[list(FEATURES)].interpolate(limit_direction="both")
    df = df.dropna(subset=list(FEATURES))

    if len(df) < seq_len * 2:
        return None, 0.0

    time = df["time"].to_numpy(dtype=np.float64)
    values = df[list(FEATURES)].to_numpy(dtype=np.float32)
    deltas = np.diff(time)
    positive_deltas = deltas[deltas > 0]
    if len(positive_deltas) == 0:
        return df[["time", *FEATURES]].reset_index(drop=True), 1.0

    dt_seconds = float(np.median(positive_deltas))
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        return df[["time", *FEATURES]].reset_index(drop=True), 1.0

    regular_time = np.arange(time[0], time[-1] + dt_seconds * 0.5, dt_seconds)
    if len(regular_time) < seq_len * 2:
        return None, dt_seconds

    regular_values = np.column_stack(
        [np.interp(regular_time, time, values[:, col]) for col in range(values.shape[1])]
    ).astype(np.float32)
    out = pd.DataFrame(regular_values, columns=FEATURES)
    out.insert(0, "time", regular_time.astype(np.float64))
    return out, dt_seconds


def split_items(
    items: list[Path],
    group_fn: Callable[[Path], str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    fixed_test_items: list[Path] | None = None,
) -> dict[Path, str]:
    rng = np.random.default_rng(seed)
    splits: dict[Path, str] = {}
    if fixed_test_items:
        splits.update({path: "test" for path in fixed_test_items})

    groups: dict[str, list[Path]] = {}
    for item in items:
        groups.setdefault(group_fn(item), []).append(item)

    group_ids = np.asarray(sorted(groups))
    rng.shuffle(group_ids)
    n_groups = len(group_ids)
    n_test = 0 if fixed_test_items else int(round(n_groups * test_fraction))
    n_val = int(round(n_groups * val_fraction))
    if n_groups > 2:
        n_val = max(1, n_val)
        if not fixed_test_items:
            n_test = max(1, n_test)

    test_groups = set(group_ids[:n_test])
    val_groups = set(group_ids[n_test : n_test + n_val])
    for group_id, paths in groups.items():
        split = "test" if group_id in test_groups else "val" if group_id in val_groups else "train"
        for path in paths:
            splits[path] = split
    return splits


def build_scalers(records: list[SeriesRecord]) -> dict[str, Any]:
    train_records = [record for record in records if record.split == "train"]
    by_dataset = {
        dataset: fit_scaler([record for record in train_records if record.dataset == dataset])
        for dataset in sorted({record.dataset for record in train_records})
    }
    global_scaler = fit_scaler(train_records)
    return {
        **global_scaler,
        "scope": "global_train",
        "selection_rule": "Each dataset is scaled with its matching by_dataset train scaler. The global scaler is saved for reference.",
        "by_dataset": by_dataset,
    }


def select_applied_scalers(scaler: dict[str, Any], selected: set[str]) -> dict[str, Any]:
    by_dataset = {
        dataset: {
            **scaler["by_dataset"][dataset],
            "scope": f"{dataset}_train",
        }
        for dataset in sorted(selected)
    }
    return {
        "features": list(FEATURES),
        "method": "per_dataset_standard",
        "fit_scope": "train_split_only",
        "by_dataset": by_dataset,
        "selected_datasets": sorted(selected),
    }


def identity_scaler(selected: set[str]) -> dict[str, Any]:
    return {
        "features": list(FEATURES),
        "method": "identity",
        "fit_scope": "not_applicable",
        "mean": [0.0] * len(FEATURES),
        "std": [1.0] * len(FEATURES),
        "fit_split": None,
        "n_points": 0,
        "scope": "raw_values",
        "selected_datasets": sorted(selected),
    }


def scale_processed_files(records: list[SeriesRecord], scaler: dict[str, Any]) -> None:
    for record in tqdm(records, desc="Scaling parquet files", unit="file"):
        record_scaler = scaler["by_dataset"][record.dataset]
        mean = np.asarray(record_scaler["mean"], dtype=np.float32)
        std = np.asarray(record_scaler["std"], dtype=np.float32)
        df = pd.read_parquet(record.processed_path)
        values = df.loc[:, list(FEATURES)].to_numpy(dtype=np.float32)
        df.loc[:, list(FEATURES)] = (values - mean) / std
        df.to_parquet(record.processed_path, index=False)


def resolve_output_dir(data_dir: Path, scale: bool) -> Path:
    names = ("processed ", "processed") if scale else ("half_processed",)
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    return data_dir / ("processed" if scale else "half_processed")


def fit_scaler(train_records: list[SeriesRecord]) -> dict[str, Any]:
    if not train_records:
        raise RuntimeError("Cannot fit scaler without train records")

    sums = np.zeros(len(FEATURES), dtype=np.float64)
    sumsq = np.zeros(len(FEATURES), dtype=np.float64)
    count = 0
    for record in tqdm(train_records, desc="Fitting scaler", unit="file", leave=False):
        values = pd.read_parquet(record.processed_path, columns=list(FEATURES)).to_numpy(dtype=np.float64)
        sums += values.sum(axis=0)
        sumsq += np.square(values).sum(axis=0)
        count += len(values)

    mean = sums / count
    var = np.maximum(sumsq / count - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return {
        "features": list(FEATURES),
        "method": "per_channel_standard",
        "fit_scope": "train_split_only",
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "fit_split": "train",
        "n_points": count,
    }


def summarize(records: list[SeriesRecord]) -> dict[str, Any]:
    df = pd.DataFrame([asdict(record) for record in records])
    by_dataset_split = (
        df.groupby(["dataset", "split"])
        .size()
        .rename("n_series")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "n_series": len(records),
        "by_dataset_split": by_dataset_split,
        "by_split": df.groupby("split").size().astype(int).to_dict(),
    }


def selected_datasets(datasets: str) -> set[str]:
    if datasets == "mixed":
        return {"matr", "nasa"}
    selected = {part.strip().lower() for part in str(datasets).split(",")}
    valid = {"matr", "nasa"}
    unknown = selected - valid
    if unknown:
        raise ValueError(f"Unknown pretrain datasets: {sorted(unknown)}")
    return selected


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def battery_id(stem: str) -> str:
    match = re.search(r"B(\d+)", stem)
    if not match:
        return stem
    return f"B{int(match.group(1)):04d}"


def matr_series_id(path: Path) -> str:
    name = path.stem.replace("_structure", "").replace("structure", "")
    return clean_name(name).strip("_")


def nasa_series_id(path: Path) -> str:
    folder = path.parent.name
    match = re.match(r"(\d+)\.\s*(.+)", folder)
    folder_id = int(match.group(1)) if match else 0
    return f"{folder_id}_{battery_id(path.stem)}"


def clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
