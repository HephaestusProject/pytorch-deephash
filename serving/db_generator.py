import base64
import torch
import torchvision
import omegaconf
import json
import numpy as np

from omegaconf import OmegaConf
from torchvision import transforms
from fastapi import HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict
from io import BytesIO
from PIL import Image
from bitarray import bitarray

from src.model.net import DeepHash


class Request(BaseModel):
    base64_image_string: str
    image_name: str
    label: str


class Response(BaseModel):
    prediction: str


class Handler(object):
    def __init__(self):
        """
        instantiation deep learning model 
        1. declare weight path variable
        2. instantiation deep learning model
        3. load weight and
        """
        config = self.get_config(root_dir="..")
        weight_file = Path(
            "DeepHash_epoch=65-tr_loss=0.00-val_loss=0.23-tr_acc=0.00-val_acc=0.93.ckpt"
        )
        self.model: DeepHash = DeepHash(config=config)
        self.model.load_state_dict(
            torch.load(str(weight_file), map_location=torch.device("cpu")), strict=False
        )
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(227),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        self.database = {}

    def __call__(self, request: Request) -> str:
        """
        inference
        Args:
            request (Request): 딥러닝 모델 추론을 위한 입력 데이터(single image)
        Returns:
            (Response): 딥러닝 모델 추론 결과
        """
        base64image: str = request.base64_image_string
        image_name: str = request.image_name
        label: str = request.label
        inputs: torch.Tensor = self._preprocessing(base64image)
        coarse_grain_tensor, fine_grain_tensor = self.model.inference(inputs)

        binary_hash = bitarray(coarse_grain_tensor.squeeze().data.numpy().astype(np.bool).tolist())
        str_binary_hash = "".join([str(int(i)) for i in binary_hash.tolist()])

        fine_grain = fine_grain_tensor.squeeze().data.numpy().tolist()

        self.register(
            image_name=image_name, binary_hash=str_binary_hash, fine_grain=fine_grain, label=label
        )

        # return Response(prediction=prediction)

    def _preprocessing(self, base64image: str) -> torch.Tensor:
        """
        base64로 encoding된 single image를 torch.tensor로 변환
        Args:
            base64image (str): base64로 encoding된 이미지
        Returns:
            (torch.Tensor): torch tensor 이미지
        """

        try:
            image = Image.open(BytesIO(base64.b64decode(base64image, validate=True)))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image string: {e}")

        inputs = self.preprocessor(image).unsqueeze(0)

        return inputs

    def get_config(self, root_dir: str) -> omegaconf.DictConfig:

        config = OmegaConf.create()
        config_dir = Path(root_dir) / Path("conf")
        model_config = OmegaConf.load(config_dir / Path("model/model.yml"))
        dataset_config = OmegaConf.load(config_dir / Path("dataset/dataset.yml"))
        runner_config = OmegaConf.load(config_dir / Path("runner/runner.yml"))

        config.update(model=model_config, dataset=dataset_config, runner=runner_config)
        OmegaConf.set_readonly(config, True)

        return config

    def register(self, image_name, binary_hash, fine_grain, label):
        if binary_hash not in self.db.keys():
            self.db.update({binary_hash: []})

        self.db[binary_hash].append(
            {"name": image_name, "label": label, "latent_vector": fine_grain}
        )

    @property
    def db(self):
        return self.database


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader

    config_dir = Path("..") / Path("conf")
    model_config = OmegaConf.load(config_dir / Path("model/model.yml"))
    dataset_config = OmegaConf.load(config_dir / Path("dataset/dataset.yml"))
    runner_config = OmegaConf.load(config_dir / Path("runner/runner.yml"))
    config = OmegaConf.create()
    config.update(model=model_config, dataset=dataset_config, runner=runner_config)
    OmegaConf.set_readonly(config, True)

    handler = Handler()

    test_dataset = datasets.CIFAR10(
        root=str((Path("..") / Path(config.dataset.params.path.test)).absolute()),
        train=False,
        transform=None,
        download=True,
    )

    train_dataset = datasets.CIFAR10(
        root=str((Path("..") / Path(config.dataset.params.path.train)).absolute()),
        train=True,
        transform=None,
        download=True,
    )

    for idx, (image, label) in enumerate(train_dataset):
        image_name = f"idx_{idx}_label_{label}.png"
        # image = image.resize((227, 227), Image.ANTIALIAS)
        # image.save(f"train_images/idx_{idx}_label_{label}.png")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        bytes_image = buffer.getvalue()
        base64_image_string = base64.b64encode(bytes_image).decode("utf-8")
        label = str(label)

        inputs = Request(
            base64_image_string=base64_image_string, image_name=image_name, label=label
        )
        handler(inputs)

    with open("database.json", "w") as json_file:
        json.dump(handler.db, json_file)
