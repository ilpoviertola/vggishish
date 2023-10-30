from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import GreatesHitDataset


class GreatesHitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        split_dir: str,
        batch_size: int,
        num_workers: int,
        label_file: str = None,
        pin_memory: bool = False,
        transforms = None
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_dir = Path(split_dir)
        self.label_file_path = Path(label_file) if label_file is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transforms = transforms

        assert self.batch_size > 0, "batch size must be greater than 0"
        assert self.num_workers >= 0, "num_workers must be greater than or equal to 0"

        self.datasets = {}

    def setup(self, stage=None) -> None:
        for split in ["train", "validation", "test"]:
            self.datasets[split] = GreatesHitDataset(
                split=split,
                split_dir_path=self.split_dir,
                data_dir_path=self.data_dir,
                transforms=self.transforms,
                label_file_path=self.label_file_path
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
