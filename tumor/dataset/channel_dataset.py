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


class ChannelBrainData(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        df: Union[str, Path],
        transforms: TRANSFORMS_TYPE,
        num_ex: int,
    ) -> None:  ### DATAFRAME WITH IMAGES AND LABELS COLS
        self.data_paths = data_paths
        df = pd.read_csv(df)
        self.ids = df["id"].to_list()
        self.labels = df["label"].to_list()
        self.transforms = transforms
        self.num_ex = num_ex

    def __getitem__(self, idx: int) -> DOUBLE_TENSOR:
        imgs = []
        for projection in self.data_paths:
            id_imgs = list(
                (Path(projection) / str(self.ids[idx]).zfill(5)).rglob("*.png")
            )
            imgs.append(
                sample_normal(id_imgs, self.num_ex, self.transforms, 6).squeeze(0)
            )

        imgs = np.stack(imgs)
        imgs = torch.tensor(imgs)
        imgs = imgs.permute(1, 0, 2, 3)

        label = torch.tensor(self.labels[idx])
        return imgs.float(), label.float()

    def __len__(self) -> int:
        return len(self.labels)
