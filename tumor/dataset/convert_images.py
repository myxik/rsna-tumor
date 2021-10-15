import cv2
import hydra
import pydicom
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from pydicom.pixel_data_handlers.util import apply_voi_lut
from typing import Tuple, Union
from pathlib import Path
from omegaconf import DictConfig

from tumor.dataset.data_utils import convert_images, make_df


@hydra.main(config_path="/workspace/configs", config_name="config.yaml")
def converter(cfg: DictConfig) -> None:
    cfg = cfg.data_prep
    save_path = cfg.save_path
    Path(save_path).mkdir(exist_ok=True, parents=True)
    data_path = cfg.data_path
    projections = cfg.projections
    img_size = tuple(cfg.img_size)
    voi_lut = cfg.voi_lut
    base_df = cfg.base_df
    split = cfg.split

    base_df = pd.read_csv(base_df)
    labels = base_df["MGMT_value"].to_numpy()
    ids = base_df["BraTS21ID"].to_numpy()
    labels_dict = dict(zip(*[ids, labels]))

    if split == "cross-validation":
        skf = StratifiedKFold(n_splits=cfg.split_params.n_folds)
        paths = convert_images(data_path, projections, save_path, img_size, voi_lut)
        for idx, (train, val) in enumerate(skf.split(ids, labels)):
            train_ids = ids[train]
            train_labels = labels[train]

            pd.DataFrame({"id": train_ids, "label": train_labels}).to_csv(
                str(Path(save_path) / f"train_{idx}.csv"), index=False
            )

            val_ids = ids[val]
            val_labels = labels[val]

            pd.DataFrame({"id": val_ids, "label": val_labels}).to_csv(
                str(Path(save_path) / f"val_{idx}.csv"), index=False
            )
    else:
        paths = convert_images(data_path, projections, save_path, img_size, voi_lut)
        make_df(paths, labels_dict, save_path)


if __name__ == "__main__":
    converter()
