"""
PCA / t-SNE로 encoder 표현을 시각화하고 ClearML에 리포트한다.
cycle, SoH, operation, battery_id, dataset으로 색칠해 표현이 의미 있는지 확인한다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from jepa import JEPA
from pretrain_loader import PretrainLoader, WindowPairDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--datasets", default="mixed")
    p.add_argument("--processed-dir-name", default="processed")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def load_soh_map(records) -> dict[str, dict[int, float]]:
    """raw .mat에서 discharge capacity를 읽어 {raw_path: {cycle_idx: soh}} 반환."""
    import scipy.io as sio

    nasa_paths = {r.raw_path for r in records if r.dataset == "nasa"}
    soh_map: dict[str, dict[int, float]] = {}

    for raw_path in sorted(nasa_paths):
        stem = Path(raw_path).stem
        try:
            mat = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
            cycles = np.atleast_1d(mat[stem].cycle)
        except Exception:
            soh_map[raw_path] = {}
            continue

        caps: dict[int, float] = {}
        for idx, cycle in enumerate(cycles):
            if str(cycle.type) == "discharge":
                try:
                    caps[idx] = float(np.atleast_1d(cycle.data.Capacity)[-1])
                except Exception:
                    pass

        if not caps:
            soh_map[raw_path] = {}
            continue

        max_cap = max(caps.values())
        soh_map[raw_path] = {idx: cap / max_cap for idx, cap in caps.items()}

    return soh_map


@torch.no_grad()
def collect(model, dataset: WindowPairDataset, soh_map: dict, n_samples: int, batch_size: int, device: str):
    """encoder를 돌려 (z_array, meta_list) 반환."""
    n = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), size=n, replace=False)

    zs, metas = [], []
    for i in range(0, n, batch_size):
        batch_idx = indices[i : i + batch_size]
        xs = torch.stack([dataset[int(j)]["x"] for j in batch_idx]).to(device)
        tokens = model.tokenizer(xs)
        z = model.encoder(tokens).mean(dim=1)   # (B, D)
        zs.append(z.cpu())
        metas.extend(get_meta(dataset, soh_map, int(j)) for j in batch_idx)

    return torch.cat(zs, dim=0).numpy(), metas


def get_meta(dataset: WindowPairDataset, soh_map: dict, idx: int) -> dict:
    rec_idx, start, _ = dataset.index[idx]
    record = dataset.records[rec_idx]
    meta = {
        "dataset":    record.dataset,
        "battery_id": record.group_id,
        "cycle_index": None,
        "operation":   None,
        "soh":         None,
    }
    if record.dataset == "nasa":
        for seg in record.metadata.get("segments", []):
            if seg["start"] <= start < seg["start"] + seg["length"]:
                cycle_idx = seg["cycle_index"]
                meta["cycle_index"] = cycle_idx
                meta["operation"]   = seg["operation"]
                cycle_soh = soh_map.get(record.raw_path, {})
                if cycle_soh:
                    nearest = min(cycle_soh, key=lambda c: abs(c - cycle_idx))
                    meta["soh"] = cycle_soh[nearest]
                break
    return meta


def pca_2d(z: np.ndarray):
    z_c = z - z.mean(0)
    _, s, Vt = np.linalg.svd(z_c, full_matrices=False)
    coords = z_c @ Vt[:2].T
    var_ratio = (s[:2] ** 2) / ((s ** 2).sum() + 1e-8)
    return coords, var_ratio


def tsne_2d(z: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z)


def scatter_categorical(ax, coords, labels, title, xlabel, ylabel):
    cats = sorted(set(v for v in labels if v is not None))
    n = max(len(cats), 1)
    cmap_name = "tab20" if n > 10 else "tab10"
    palette = plt.colormaps[cmap_name](np.linspace(0, 1, n))
    color_map = dict(zip(cats, palette))
    c = [color_map.get(v, (0.5, 0.5, 0.5, 0.3)) for v in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=c, s=3, alpha=0.5)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=6, label=cat)
               for cat, col in color_map.items()]
    ax.legend(handles=handles, fontsize=6, ncol=max(1, n // 10))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)


def scatter_continuous(ax, coords, values, title, xlabel, ylabel):
    valid = np.array([v is not None for v in values])
    if not valid.any():
        ax.set(title=f"{title} (no data)", xlabel=xlabel, ylabel=ylabel)
        return
    c = np.array([v for v in values if v is not None], dtype=float)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], c=c, s=3, alpha=0.5, cmap="plasma")
    plt.colorbar(sc, ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)


def make_figure(pca_coords, tsne_coords, var_ratio, metas) -> plt.Figure:
    projs = [
        (pca_coords,  "PCA",   f"PC1 ({var_ratio[0]:.1%})", f"PC2 ({var_ratio[1]:.1%})"),
        (tsne_coords, "t-SNE", "t-SNE 1",                   "t-SNE 2"),
    ]
    color_specs = [
        ("dataset",    [m["dataset"]     for m in metas], "categorical", None),
        ("operation",  [m["operation"]   for m in metas], "categorical", None),
        ("battery_id", [m["battery_id"]  for m in metas], "categorical", None),
        ("cycle",      [m["cycle_index"] for m in metas], "continuous",  "plasma"),
        ("SoH",        [m["soh"]         for m in metas], "continuous",  "RdYlGn"),
    ]

    n_rows = len(color_specs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))

    for row, (name, values, kind, cmap) in enumerate(color_specs):
        for col, (coords, proj_name, xl, yl) in enumerate(projs):
            ax = axes[row, col]
            title = f"{proj_name} – {name}"
            if kind == "categorical":
                scatter_categorical(ax, coords, values, title, xl, yl)
            else:
                scatter_continuous(ax, coords, values, title, xl, yl)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = JEPA.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()

    dm = PretrainLoader(
        data_dir=args.data_dir,
        datasets=args.datasets,
        seq_len=model.hparams.seq_len,
        processed_dir_name=args.processed_dir_name,
        batch_size=1,
        num_workers=0,
    )
    dm.setup()
    dataset = {"train": dm.train_ds, "val": dm.val_ds, "test": dm.test_ds}[args.split]

    print("SoH 로딩...")
    soh_map = load_soh_map(dm.records)

    print(f"Collecting {args.n_samples} embeddings from {args.split} split ({len(dataset)} windows)...")
    z, metas = collect(model, dataset, soh_map, args.n_samples, args.batch_size, device)

    print("PCA...")
    pca_coords, var_ratio = pca_2d(z)

    print("t-SNE...")
    tsne_coords = tsne_2d(z)

    fig = make_figure(pca_coords, tsne_coords, var_ratio, metas)

    try:
        from clearml import Task
        task = Task.init(
            project_name="BESS-JEPA",
            task_name="validate_representation",
            reuse_last_task_id=False,
        )
        task.get_logger().report_matplotlib_figure("Representations", "PCA+t-SNE", fig, 0)
        print(f"ClearML task {task.id} 에 리포트 완료.")
    except Exception as e:
        out = Path("representation_validation.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"ClearML 없음 ({e}). {out}에 저장.")

    plt.close(fig)


if __name__ == "__main__":
    main()
