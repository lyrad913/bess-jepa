import sys
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR / "models"))
sys.path.insert(0, str(SRC_DIR / "data"))

from context_query_projection_jepa import ContextQueryProjectionJEPA
from pretrain_loader import PretrainLoader


@hydra.main(config_path="../../../config", config_name="pretrain_context_query_projection_jepa", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)

    try:
        from clearml import Task

        task = Task.init(project_name="BESS-JEPA", task_name="pretrain_context_query_projection_jepa")
        task.connect(OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        print(f"ClearML unavailable: {e}")

    dm = PretrainLoader(**cfg.data)
    model = ContextQueryProjectionJEPA(**cfg.model)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=TensorBoardLogger(save_dir="logs/pretrain", name="context_query_projection_jepa"),
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="best-context-query-projection-jepa",
            ),
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
