import torch
import pandas as pd

from albumentations import Compose
from tumor.dataset.base_dataset import BrainData
from albumentations.pytorch.transforms import ToTensorV2


params = {
    "df": pd.read_csv("/workspace/tests/test_df.csv"),
    "transforms": Compose(ToTensorV2()),
}


def test_init():
    dataset = BrainData(**params)
    assert isinstance(dataset.imgs, list)
    assert isinstance(dataset.labels, list)


def test_labels_range():
    dataset = BrainData(**params)
    labels_range = [0, 1]
    labels = []

    for label in dataset.labels:
        if label in labels_range:
            labels.append(label)

    assert len(dataset.labels) == len(labels)
