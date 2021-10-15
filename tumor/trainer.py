import torch

from pytorch_lightning import LightningModule
from typing import Any, Callable, List, Tuple, Dict
from torch import Tensor


LOSS_TYPE = Callable[[Tensor, Tensor], Tensor]
METRICS_TYPE = List[Any]
SAMPLE_TYPE = Tuple[Tensor, Tensor]


class Lit(LightningModule):
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        criterion: LOSS_TYPE,
        metrics: METRICS_TYPE,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metrics = metrics

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: SAMPLE_TYPE, batch_idx: int) -> Tensor:
        image, labels = batch
        outputs = self.model(image)
        outputs = outputs.squeeze(1)

        loss = self.criterion(outputs, labels)
        outputs = torch.sigmoid(outputs)
        self.update_metrics(outputs, labels, "train")
        metrics = self.compute_metrics("train")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: SAMPLE_TYPE, batch_idx: int) -> Tensor:
        image, labels = batch
        outputs = self.model(image)
        outputs = outputs.squeeze(1)

        loss = self.criterion(outputs, labels)
        outputs = torch.sigmoid(outputs)
        self.update_metrics(outputs, labels, "val")
        metrics = self.compute_metrics("val")

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_epoch=True)

    def update_metrics(self, outputs: Tensor, labels: Tensor, mode: str) -> None:
        for metric in self.metrics:
            if mode in metric.name:
                metric.update(outputs, labels)

    def compute_metrics(self, mode: str) -> Dict[str, Tensor]:
        metric_name = []
        metric_value = []
        for metric in self.metrics:
            if mode in metric.name:
                metric_name.append(metric.name)
                metric_value.append(metric.compute())
        return dict(zip(*[metric_name, metric_value]))

    def configure_optimizers(self) -> Any:
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
                "strict": True,
                "name": "ReduceOnPlateau",
            },
        }

    def soft_ensemble(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=1)
