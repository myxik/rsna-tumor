# from pathlib import Path

# min_imgs = 100000
# workdir = Path('/workspace/data/tw1')
# for patiend_id in workdir.iterdir():
#     num_imgs = len(list((patiend_id / "T1w").rglob("*.png")))
#     if num_imgs < min_imgs:
#         min_imgs = num_imgs
#         min_id = patiend_id
# print(min_imgs)
# print(min_id)

# import torch
# from tumor.model.gru_2d import GRUover2d

# device = torch.device("cuda")

# model = GRUover2d("tf_efficientnetv2_s", False, 512, 512)
# model.to(device)
# a = torch.Tensor(1, 16, 3, 300, 300).to(device)

# out = model(a)

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# transforms = A.load("/workspace/configs/transforms/augs.yaml", data_format="yaml")
# print(transforms)

# transforms = A.Compose([
#   ToTensorV2(always_apply=True, p=1.0, transpose_mask=False),
# ], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})

# A.save(transforms, "/workspace/configs/transforms/test_augs.yaml", data_format="yaml")

import cv2

print(cv2.COLOR_BGR2RGB)
print(cv2.COLOR_BGR2GRAY)
