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

        with open("database.json") as json_file:
            self.database = json.load(json_file)

    def __call__(self, request: Request) -> str:
        """
        inference
        Args:
            request (Request): 딥러닝 모델 추론을 위한 입력 데이터(single image)
        Returns:
            (Response): 딥러닝 모델 추론 결과
        """
        base64image: str = request.base64_image_string
        inputs: torch.Tensor = self._preprocessing(base64image)
        coarse_grain_tensor, fine_grain_tensor = self.model.inference(inputs)

        input_binary_hash = bitarray(
            coarse_grain_tensor.squeeze().data.numpy().astype(np.bool).tolist()
        )
        str_binary_hash = "".join([str(int(i)) for i in input_binary_hash.tolist()])

        fine_grain = fine_grain_tensor.squeeze().data.numpy()

        hamming_distances = []
        binary_hashs = list(self.database.keys())
        for target_str_binary_hash in binary_hashs:
            target_binary_hash = bitarray(target_str_binary_hash)
            bit_diff = target_binary_hash ^ input_binary_hash
            hamming_distance = bit_diff.count()
            hamming_distances.append(hamming_distance)

        np_hamming_distances = np.array(hamming_distances)
        minimum_distance = min(hamming_distances)
        candidates_indexs = list(np.where(np_hamming_distances == minimum_distance))[0].tolist()

        if len(candidates_indexs) < 2:
            final_hash = binary_hashs[candidates_indexs[0]]
            result = self.database[final_hash]
            print("final")
            print(result)
            exit()

        detail_distances = []
        for candidate_index in candidates_indexs:
            search_key = binary_hashs[candidate_index]
            detail_candidate = self.database[search_key][0]
            latent_vector = np.array(detail_candidate["latent_vector"])
            euclidean_distance = np.linalg.norm(fine_grain - latent_vector)
            detail_distances.append(euclidean_distance)

        min_val = min(detail_distances)
        final_search_key = detail_distances.index(min_val)
        final_hash = candidates_indexs.index(final_search_key)
        result = self.database[final_hash]
        print("final")
        print(result)

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

    for idx, (image, label) in enumerate(test_dataset):
        if idx > 0:
            break
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        bytes_image = buffer.getvalue()
        base64_image_string = base64.b64encode(bytes_image).decode("utf-8")
        print(f"label : {label}")

        inputs = Request(base64_image_string=base64_image_string)
        handler(inputs)
