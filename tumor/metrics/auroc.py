from torchmetrics import AUROC
from torch import Tensor


class ROCAUC(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.metric = AUROC(num_classes=2, average="macro")

    def update(self, prediction: Tensor, target: Tensor) -> None:
        target = target.long()
        self.metric.update(prediction, target)

    def compute(self) -> Tensor:
        try:
            return self.metric.compute()
        except:
            return Tensor([0.0])
