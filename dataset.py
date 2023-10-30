from pathlib import Path
import json

import numpy as np
from torch.utils.data import Dataset
import torch


class GreatesHitDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_dir_path: Path,
        split_dir_path: Path,
        transforms=None,
        label_file_path: Path = None,
    ):
        super().__init__()
        assert data_dir_path.exists(), f"data_dir_path ({data_dir_path}) does not exist"
        assert (
            data_dir_path.is_dir()
        ), f"data_dir_path ({data_dir_path}) has to be directory"
        assert split in ["train", "test", "validation"], f"invalid split ({split})"
        assert (
            split_dir_path.exists()
        ), f"split_dir_path ({split_dir_path}) does not exist"
        assert (
            split_dir_path.is_dir()
        ), f"split_dir_path ({split_dir_path}) has to be directory"
        split_file_path = split_dir_path / f"{split}.txt"
        assert (
            split_file_path.exists()
        ), f"split_file_path ({split_file_path}) does not exist"
        self.transforms = transforms
        if label_file_path is None:
            label_file_path = Path(self.data_dir_path / "labels.json")
        assert (
            label_file_path.exists()
        ), f"label_file_path ({label_file_path}) does not exist"
        assert (
            label_file_path.is_file()
        ), f"label_file_path ({label_file_path}) has to be file"

        self.dataset = []
        with open(split_file_path, encoding="utf-8") as f:
            within_split = f.read().splitlines()

        for basename in within_split:
            files = self._get_all_files_with_same_basename(basename, data_dir_path)
            self.dataset += files

        with open(label_file_path, encoding="utf-8") as json_file:
            self.filename2label = json.load(json_file)

    def __getitem__(self, index: int) -> dict:
        data_path: Path = self.dataset[index]
        spec = np.load(data_path.as_posix())
        spec = self.transforms(spec) if self.transforms is not None else spec
        return {
            "spec": torch.from_numpy(spec),
            "label": self._label2code(data_path.stem),
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def _label2code(self, filename: str):
        label = self.filename2label[filename]
        if label == "hit":
            return 0
        else:  # assume scratch
            return 1

    @staticmethod
    def _get_all_files_with_same_basename(basename: str, data_dir: Path):
        all_files = data_dir.glob(f"{basename}*")
        return list(all_files)
