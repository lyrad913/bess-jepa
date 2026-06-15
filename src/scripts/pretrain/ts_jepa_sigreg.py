import sys
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf, open_dict

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR / "models"))
sys.path.insert(0, str(SRC_DIR / "data"))

from pretrain_loader import PretrainLoader
from ts_jepa_sigreg import TSJEPASIGReg


@hydra.main(config_path="../../../config", config_name="pretrain_ts_jepa", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.trainer.seed)

    if float(cfg.model.sigreg_lambda) == 0.0:
        with open_dict(cfg):
            cfg.model.sigreg_lambda = 0.1

    try:
        from clearml import Task

        task = Task.init(project_name="BESS-JEPA", task_name="pretrain_ts_jepa_sigreg")
        task.connect(OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        print(f"ClearML unavailable: {e}")

    dm = PretrainLoader(**cfg.data)
    model = TSJEPASIGReg(**cfg.model)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=TensorBoardLogger(save_dir="logs/pretrain", name="ts_jepa_sigreg"),
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=cfg.trainer.patience, mode="min"),
            ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename="best-ts-jepa-sigreg"),
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
