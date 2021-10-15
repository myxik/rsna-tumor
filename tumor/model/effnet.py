import timm
import torch
import torch.nn as nn

from torch import Tensor


class EffNet(nn.Module):
    def __init__(self, model_name: str, pretrained: bool) -> None:
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if "efficientnet" in model_name:
            self.backbone.classifier = nn.Linear(
                self.backbone.classifier.in_features, 1
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.softmax(x)
        return x
