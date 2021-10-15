import cv2
import torch
import pydicom
import numpy as np
import pandas as pd

from pydicom.pixel_data_handlers.util import apply_voi_lut
from typing import Callable, Tuple, Union, List, Dict
from pathlib import Path
from tqdm import tqdm
from torch import Tensor


def load_dicom_image(
    path: Union[str, Path], img_size: Tuple[int] = (512, 512), voi_lut: bool = True
) -> None:
    """SOURCE: https://www.kaggle.com/rluethy/efficientnet3d-with-one-mri-type#Functions-to-load-images"""
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = cv2.resize(data, img_size)
    return data


def convert_images(
    folder_path: Union[str, Path],
    projections: List[str],
    save_path: Union[str, Path],
    img_size: Tuple[int] = (512, 512),
    voi_lut: bool = True,
) -> List[str]:
    paths = []
    folder_path = Path(folder_path)
    save_path = Path(save_path)
    tq = tqdm(folder_path.iterdir(), desc=f"Processing images")
    for data_path in tq:
        for projection in projections:
            dcm_paths = list((Path(data_path) / projection).rglob("*.dcm"))
            for dcm in dcm_paths:
                patient_id = str(dcm.parents[1].stem)
                (save_path / f"{patient_id}/{projection}").mkdir(
                    exist_ok=True, parents=True
                )
                cv2.imwrite(
                    str(save_path / f"{patient_id}/{projection}/{dcm.stem}.png"),
                    load_dicom_image(dcm, img_size, voi_lut),
                )
    return paths


def make_df(
    paths: List[str], labels_dict: Dict[int, int], save_path: Union[str, Path]
) -> None:
    labels = []
    for path in paths:
        patient_id = int(Path(path).parents[1].stem)
        labels.append(labels_dict[patient_id])

    df = pd.DataFrame({"images": paths, "labels": labels})
    save_path = Path(save_path)

    df.to_csv(str(save_path / "dataframe.csv"), index=False)


def get_id(elem: Path) -> int:
    return int(str(elem.stem).replace("Image-", ""))


def sample_normal(
    id_paths: List[Path],
    num_samples: int,
    transforms: Callable[[Tensor], Tensor],
    color_code: int,
) -> Tensor:
    id_paths = sorted(id_paths, key=get_id)
    id_paths = np.array(id_paths)
    dist = np.random.normal(loc=0.0, scale=1.0, size=len(id_paths))
    indices = np.argsort(dist)[::-1]
    indices = sorted(indices[:num_samples])
    # indices = sorted(indices)

    id_paths = id_paths[indices]
    # id_paths = id_paths[:num_samples]
    img_3d = []
    for img_path in id_paths:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, color_code)

        if transforms:
            img = transforms(image=img)["image"]
        img_3d.append(img)

    img_3d = np.stack(img_3d)
    if color_code == cv2.COLOR_BGR2RGB:
        img_3d = torch.tensor(img_3d)
    else:
        img_3d = torch.tensor(img_3d)
        img_3d = img_3d.permute(1, 0, 2, 3)
    return img_3d
