from pathlib import Path

from datetime import datetime, timedelta

import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
    BatchSizeFinder,
    LearningRateFinder,
)


################# TRAIN #################
@rank_zero_only
def maybe_save_checkpoint(trainer: Trainer):
    print("Saving checkpoint...")
    ckpt_path = (
        Path(trainer.log_dir)
        / "checkpoints"
        / f"e{trainer.current_epoch}_last_at_{get_curr_time_w_random_shift()}.ckpt"
    )
    if trainer.global_rank == 0:
        trainer.save_checkpoint(ckpt_path)


def init_log_directory(timestamp: str, log_dir: str = "./logs") -> tuple:
    log_dir = Path(log_dir) / timestamp
    ckpt_dir = log_dir / "checkpoints"

    for d in [log_dir, ckpt_dir]:
        if not d.exists():
            d.mkdir(parents=True)
    return log_dir, ckpt_dir


def get_logger(log_dir: Path, name: str = "", version: str = "") -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=log_dir.as_posix(),
        name=name,
        version=version,
        log_graph=False,
        default_hp_metric=False,
    )
    return logger


def get_callbacks(ckpt_dir: Path, save_top_k: int = 3) -> list:
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=ckpt_dir.as_posix(),
            filename="{epoch}-{step}-{val_loss:.3f}",
            monitor="val_loss_epoch",
            mode="min",
            save_top_k=save_top_k,
            save_on_train_epoch_end=True,
        ),
        EarlyStopping(monitor="val_loss_epoch", patience=3, mode="min", verbose=True),
        # DeviceStatsMonitor(cpu_stats=True),
        # BatchSizeFinder(),
        # LearningRateFinder(num_training_steps=200),
    ]
    return callbacks


################# MISC #################
def get_curr_time_w_random_shift() -> str:
    # shifting for a random number of seconds so that exp folder names coincide less often
    now = datetime.now() - timedelta(seconds=np.random.randint(60))
    return now.strftime("%y-%m-%dT%H-%M-%S")
