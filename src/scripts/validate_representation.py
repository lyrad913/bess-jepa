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

from context_mask_jepa import ContextMaskJEPA
from context_mask_projection_jepa import ContextMaskProjectionJEPA
from jepa import JEPA
from masked_autoencoder import MaskedAutoencoder
from projection_jepa import ProjectionJEPA
from pretrain_loader import PretrainLoader, WindowPairDataset
from ts_jepa import TSJEPA
from ts_jepa_sigreg import TSJEPASIGReg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--model-type",
        default="jepa",
        choices=[
            "jepa",
            "projection_jepa",
            "context_mask_jepa",
            "context_mask_projection_jepa",
            "ts_jepa",
            "ts_jepa_sigreg",
            "mae",
        ],
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--datasets", default="mixed")
    p.add_argument("--processed-dir-name", default="processed")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", default="representation_validation.png")
    p.add_argument(
        "--projections",
        nargs="+",
        default=["pca", "tsne"],
        choices=["pca", "tsne", "umap"],
        help="Which 2D projections to plot.",
    )
    p.add_argument(
        "--color-by",
        nargs="+",
        default=["dataset", "operation", "battery_id", "cycle", "soh"],
        choices=["dataset", "operation", "battery_id", "cycle", "soh"],
        help="Metadata fields used to color the plots.",
    )
    return p.parse_args()


def load_model(model_type: str, checkpoint: str, device: str):
    model_classes = {
        "jepa": JEPA,
        "projection_jepa": ProjectionJEPA,
        "context_mask_jepa": ContextMaskJEPA,
        "context_mask_projection_jepa": ContextMaskProjectionJEPA,
        "ts_jepa": TSJEPA,
        "ts_jepa_sigreg": TSJEPASIGReg,
        "mae": MaskedAutoencoder,
    }
    model_cls = model_classes[model_type]
    return model_cls.load_from_checkpoint(checkpoint, map_location=device)


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


def umap_2d(z: np.ndarray) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise ImportError("UMAP을 쓰려면 `uv add umap-learn`으로 의존성을 추가해야 합니다.") from e
    return umap.UMAP(n_components=2, random_state=42).fit_transform(z)


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


def scatter_continuous(ax, coords, values, title, xlabel, ylabel, cmap="plasma"):
    valid = np.array([v is not None for v in values])
    if not valid.any():
        ax.set(title=f"{title} (no data)", xlabel=xlabel, ylabel=ylabel)
        return
    c = np.array([v for v in values if v is not None], dtype=float)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], c=c, s=3, alpha=0.5, cmap=cmap)
    plt.colorbar(sc, ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)


def make_figure(coords_by_projection, var_ratio, metas, projections: list[str], color_by: list[str]) -> plt.Figure:
    all_projs = {
        "pca": (coords_by_projection.get("pca"), "PCA", f"PC1 ({var_ratio[0]:.1%})", f"PC2 ({var_ratio[1]:.1%})"),
        "tsne": (coords_by_projection.get("tsne"), "t-SNE", "t-SNE 1", "t-SNE 2"),
        "umap": (coords_by_projection.get("umap"), "UMAP", "UMAP 1", "UMAP 2"),
    }
    projs = [all_projs[name] for name in projections]

    all_color_specs = {
        "dataset": ("dataset", [m["dataset"] for m in metas], "categorical", None),
        "operation": ("operation", [m["operation"] for m in metas], "categorical", None),
        "battery_id": ("battery_id", [m["battery_id"] for m in metas], "categorical", None),
        "cycle": ("cycle", [m["cycle_index"] for m in metas], "continuous", "plasma"),
        "soh": ("SoH", [m["soh"] for m in metas], "continuous", "RdYlGn"),
    }
    color_specs = [
        all_color_specs[name]
        for name in color_by
    ]

    n_rows = len(projs)
    n_cols = len(color_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for row, (coords, proj_name, xl, yl) in enumerate(projs):
        for col, (name, values, kind, cmap) in enumerate(color_specs):
            ax = axes[row, col]
            title = f"{proj_name} – {name}"
            if kind == "categorical":
                scatter_categorical(ax, coords, values, title, xl, yl)
            else:
                scatter_continuous(ax, coords, values, title, xl, yl, cmap=cmap)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.model_type, args.checkpoint, device)
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

    coords_by_projection = {}
    var_ratio = np.array([0.0, 0.0])

    if "pca" in args.projections:
        print("PCA...")
        coords_by_projection["pca"], var_ratio = pca_2d(z)
    elif "pca" not in args.projections:
        _, var_ratio = pca_2d(z)

    if "tsne" in args.projections:
        print("t-SNE...")
        coords_by_projection["tsne"] = tsne_2d(z)

    if "umap" in args.projections:
        print("UMAP...")
        coords_by_projection["umap"] = umap_2d(z)

    fig = make_figure(coords_by_projection, var_ratio, metas, args.projections, args.color_by)

    try:
        from clearml import Task
        task = Task.init(
            project_name="BESS-JEPA",
            task_name="validate_representation",
            reuse_last_task_id=False,
        )
        task.get_logger().report_matplotlib_figure("Representations", "+".join(args.projections), fig, 0)
        print(f"ClearML task {task.id} 에 리포트 완료.")
    except Exception as e:
        out = Path(args.output)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"ClearML 없음 ({e}). {out}에 저장.")

    plt.close(fig)


if __name__ == "__main__":
    main()
