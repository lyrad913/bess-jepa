import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR / "models"))
sys.path.insert(0, str(SRC_DIR / "data"))

from hybrid_jepa import HybridJEPA
from pretrain_loader import PretrainLoader


class HybridGapDataset(Dataset):
    """Map loader gaps to hybrid semantics and inject masked samples.

    Base PretrainLoader gap r means y starts at (r + 1) windows after x.
    HybridJEPA gap h means:
      h == 0: masked in-window objective
      h >= 1: y starts h windows after x

    Therefore future samples use h = r + 1. Masked samples keep the same x
    and ignore y inside the model.
    """

    def __init__(self, base: Dataset, masked_fraction: float, seed: int):
        self.base = base
        self.masked_fraction = masked_fraction
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base[idx]
        if self._use_masked_objective(idx):
            item["gap"] = torch.zeros_like(item["gap"])
        else:
            item["gap"] = item["gap"] + 1
        return item

    def _use_masked_objective(self, idx: int) -> bool:
        if self.masked_fraction <= 0:
            return False
        if self.masked_fraction >= 1:
            return True
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)
        return bool(torch.rand((), generator=generator).item() < self.masked_fraction)


class HybridPretrainLoader(PretrainLoader):
    def __init__(self, *args, masked_fraction: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_fraction = masked_fraction

    def _dataset_for_split(self, split: str) -> Dataset:
        base = super()._dataset_for_split(split)
        return HybridGapDataset(base, masked_fraction=self.masked_fraction, seed=self.seed + self._split_offset(split))

    @staticmethod
    def _split_offset(split: str) -> int:
        return {"train": 0, "val": 10_000_000, "test": 20_000_000}.get(split, 30_000_000)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )


@hydra.main(config_path="../../../config", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)

    try:
        from clearml import Task

        task = Task.init(project_name="BESS-JEPA", task_name="pretrain_hybrid_jepa")
        task.connect(OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        print(f"ClearML unavailable: {e}")

    dm = HybridPretrainLoader(**cfg.data)
    model = HybridJEPA(**cfg.model)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=TensorBoardLogger(save_dir="logs/pretrain", name="hybrid_jepa"),
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename="best-hybrid-jepa"),
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
