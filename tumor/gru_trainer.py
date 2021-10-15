import torch

from typing import Any, Callable, List, Tuple, Dict
from torch import Tensor

from tumor.trainer import Lit


LOSS_TYPE = Callable[[Tensor, Tensor], Tensor]
METRICS_TYPE = List[Any]
SAMPLE_TYPE = Tuple[Tensor, Tensor]


class GRULit(Lit):
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        criterion: LOSS_TYPE,
        metrics: METRICS_TYPE,
    ) -> None:
        super().__init__(model, optimizer, scheduler, criterion, metrics)

    def training_step(self, batch: SAMPLE_TYPE, batch_idx: int) -> Tensor:
        image, labels = batch
        image = image.squeeze(0)

        labels = torch.full(
            (image.shape[0],), fill_value=labels.item(), device=self.device
        )
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
        image = image.squeeze(0)

        labels = torch.full(
            (image.shape[0],), fill_value=labels.item(), device=self.device
        )
        outputs = self.model(image)
        outputs = outputs.squeeze(1)

        loss = self.criterion(outputs, labels)
        outputs = torch.sigmoid(outputs)
        self.update_metrics(outputs, labels, "val")
        metrics = self.compute_metrics("val")

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_epoch=True)
