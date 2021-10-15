import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch_3d import EfficientNet3D
from torch import Tensor


class EffNet(nn.Module):
    def __init__(self, model_name: str = "efficientnet-b7") -> None:
        super().__init__()
        self.model = EfficientNet3D.from_name(
            model_name, override_params={"num_classes": 2}, in_channels=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x
