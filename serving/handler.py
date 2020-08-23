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
            "../DeepHash_epoch=65-tr_loss=0.00-val_loss=0.23-tr_acc=0.00-val_acc=0.93.ckpt"
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

        with open("../database.json") as json_file:
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

        # 분할정복으로 풀어야함
        # 1. coarse grain으로 hamming distance를 구함
        # 2. hamming distance기반으로 euclidean distance를 구함
        # =============================================

        # 1. coarse grain을 binaryhash로 변경한다.
        # 2. 변경된 binaryhash과 데이터베이스의 binaryhash간의 hamming distance를 구한다.
        #     2.1. 같은 hamming distance값을 갖는 애들을 뭉친다.
        # 3. top-k에 부합하는 hamming distance값들을 뭉친다.
        coarse_grain = coarse_grain_tensor.squeeze().data.numpy().astype(np.bool)
        fine_grain = fine_grain_tensor.squeeze().data.numpy()
        top_k = 3

        hamming_distances = []
        registered_binaryhashs: List = list(self.database.keys())
        for target_binaryhash in registered_binaryhashs:
            target_binaryhash = np.array([int(i) for i in list(target_binaryhash)], dtype=np.bool)
            bit_diff = target_binaryhash ^ coarse_grain
            hamming_distance = len(bit_diff == True)
            hamming_distances.append(hamming_distance)

        tree = {}
        hamming_distances = np.array(hamming_distances)
        fisrt_nodes = list(set(hamming_distances))
        for node in fisrt_nodes:
            keys = list(np.where(hamming_distances == node)[0])
            tree.update({node: {"indexs": keys}})

        sorted_index_hamming_distance = np.argsort(hamming_distances)

        # candidates 구함
        # coarse grain으로부터 구한 후보군들 중에 어디까지 fine grain feature를 이용해서 euclidean distance를 구할 것인지 계산해야함
        candidates_indexes = []
        count = 0
        while len(candidates_indexes) < top_k:
            hamming_distance_index = sorted_index_hamming_distance[count]
            hamming_distance = hamming_distances[hamming_distance_index]
            candidates_indexes.extend(tree[hamming_distance]["indexs"])
            count += 1

        euclidean_distances = []
        for canditate_index in candidates_indexes:
            candidate_binaryhash = registered_binaryhashs[canditate_index]
            target_fine_grain = np.array(self.database[candidate_binaryhash]["fine_grain"])
            euclidean_distance = np.linalg.norm(fine_grain - target_fine_grain)
            euclidean_distances.append(euclidean_distance)

        euclidean_distances = np.array(euclidean_distances)
        sorted_index_euclidean_distances = np.argsort(euclidean_distances)

        result_indexes = np.array([])
        for i in range(top_k):
            index = sorted_index_euclidean_distances[i]
            index = np.where(euclidean_distances == euclidean_distances[index])[0]
            result_indexes = np.append(result_indexes, index)

        # 유클리디안 결과
        candidates = sorted_index_euclidean_distances[0:top_k]
        for candidate in candidates:
            print(euclidean_distances[candidate])

        result_indexes = result_indexes.astype(np.int)
        for index in result_indexes:
            candidate_binaryhash = registered_binaryhashs[index]
            result = self.database[candidate_binaryhash]

            target_fine_grain = np.array(result["fine_grain"])
            target_binaryhash = np.array([int(i) for i in list(target_binaryhash)], dtype=np.bool)
            print(f"hamming distance : {len((target_binaryhash ^ coarse_grain) == True)}")
            print(f"euclidean distance : {np.linalg.norm(fine_grain - target_fine_grain)}")
            print(f"name : {result['name']}")
            print(f"label : {result['label']}")

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
