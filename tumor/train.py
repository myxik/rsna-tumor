import hydra
import albumentations as A

from shutil import rmtree
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from albumentations.pytorch.transforms import ToTensorV2

from tumor.trainer import Lit
from tumor.utils.helper import load_obj, load_list_obj


@hydra.main(config_path="/workspace/configs", config_name="config.yaml")
def run(cfg: DictConfig) -> None:
    train_transforms = A.load(cfg.general.train_transforms, data_format="yaml")
    val_transforms = A.load(cfg.general.test_transforms, data_format="yaml")

    traindataset = load_obj(cfg.dataset.class_name)(
        transforms=train_transforms, **cfg.dataset.train.params
    )
    valdataset = load_obj(cfg.dataset.class_name)(
        transforms=val_transforms, **cfg.dataset.val.params
    )

    trainloader = load_obj(cfg.dataloader.class_name)(
        dataset=traindataset, **cfg.dataloader.train.params
    )
    valloader = load_obj(cfg.dataloader.class_name)(
        dataset=valdataset, **cfg.dataloader.val.params
    )

    model = load_obj(cfg.model.class_name)(**cfg.model.params)

    optimizer = load_obj(cfg.optimizer.class_name)(
        model.parameters(), **cfg.optimizer.params
    )
    scheduler = load_obj(cfg.scheduler.class_name)(optimizer, **cfg.scheduler.params)
    criterion = load_obj(cfg.general.criterion)(**cfg.general.criterion_params)

    metrics = load_list_obj(cfg.metrics.list)
    callbacks = load_list_obj(cfg.callbacks.list)
    exp_dir = Path(cfg.logger.params.save_dir) / (
        cfg.general.experiment_name + "/" + cfg.general.version
    )
    if exp_dir.exists():
        rmtree(exp_dir)

    logger = load_obj(cfg.logger.class_name)(
        name=cfg.general.experiment_name,
        version=cfg.general.version,
        **cfg.logger.params
    )

    module = load_obj(cfg.general.trainer)(
        model, optimizer, scheduler, criterion, metrics
    )
    tr = Trainer(
        gpus=cfg.general.gpu_count,
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.general.max_epochs,
    )
    tr.fit(module, trainloader, valloader)


if __name__ == "__main__":
    run()
