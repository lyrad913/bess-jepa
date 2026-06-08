import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from clearml import Task

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from jepa import JEPA
from pretrain_loader import PretrainLoader


@hydra.main(config_path="../../config", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)

    task = Task.init(project_name="BESS-JEPA", task_name="pretrain")
    task.connect(OmegaConf.to_container(cfg, resolve=True))

    dm = PretrainLoader(**cfg.data)
    model = JEPA(**cfg.model)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename="best"),
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
