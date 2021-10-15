import cv2
import torch
import numpy as np
import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Union, Tuple, Callable
from pandas import DataFrame
from pathlib import Path

from tumor.dataset.data_utils import sample_normal


DOUBLE_TENSOR = Tuple[Tensor, Tensor]
TRANSFORMS_TYPE = Callable[[Tensor], Tensor]


class BrainData(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        df: Union[str, Path],
        transforms: TRANSFORMS_TYPE,
        num_ex: int,
        color_code: int,
    ) -> None:  ### DATAFRAME WITH IMAGES AND LABELS COLS
        self.data_path = data_path
        df = pd.read_csv(df)
        self.ids = df["id"].to_list()
        self.labels = df["label"].to_list()
        self.transforms = transforms
        self.num_ex = num_ex
        self.color_code = color_code

    def __getitem__(self, idx: int) -> DOUBLE_TENSOR:
        id_imgs = list(
            (Path(self.data_path) / str(self.ids[idx]).zfill(5)).rglob("*.png")
        )
        img_3d = sample_normal(id_imgs, self.num_ex, self.transforms, self.color_code)

        label = torch.tensor(self.labels[idx])
        return img_3d.float(), label.float()

    def __len__(self) -> int:
        return len(self.labels)
