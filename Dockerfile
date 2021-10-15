FROM myxik/myxik_container:latest
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

RUN pip uninstall nvidia-tensorboard nvidia-tensorboard-plugin-dlprof -y
RUN pip install kaggle pytest pydicom torchmetrics black
RUN pip install pytorch-lightning timm
RUN pip install -U albumentations[imgaug] --ignore-installed ruamel.yaml
RUN pip install git+https://github.com/shijianjian/EfficientNet-PyTorch-3D
RUN pip install git+https://github.com/rwightman/pytorch-image-models.git
