"""
Usage:
    main.py build_database [options]
    main.py build_database (-h | --help)
Options:
    --dataset-config <dataset config path>  Path to YAML file for dataset configuration  [default: conf/dataset/dataset.yml] [type: path]
    --model-config <model config path>  Path to YAML file for model configuration  [default: conf/model/model.yml] [type: path]
    --runner-config <runner config path>  Path to YAML file for model configuration  [default: conf/runner/runner.yml] [type: path]    
    --database-config <database config path>  Path to YAML file for database configuration  [default: conf/database/database.yml] [type: path]
    --test-config <test config path>  Path to YAML file for test configuration  [default: conf/test/test.yml] [type: path]

    -h --help  Show this.
"""
import json
import time
from functools import wraps
from pathlib import Path
from typing import Dict, List, Tuple

import numpy
import numpy as np
import omegaconf
import torch
import torchvision
import torchvision.transforms as transforms
from bitarray import bitarray
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset.dataset_registry import DataSetRegistry
from src.model.net import DeepHash


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Function called {elapsed_time} times. ")

        return ret

    return with_profiling


# TODO. train.py와 중복됨
def get_config(hparams: Dict) -> omegaconf.DictConfig:
    config = OmegaConf.create()

    config_dir = Path(".")
    model_config = OmegaConf.load(config_dir / hparams.get("--model-config"))
    dataset_config = OmegaConf.load(config_dir / hparams.get("--dataset-config"))
    runner_config = OmegaConf.load(config_dir / hparams.get("--runner-config"))
    test_config = OmegaConf.load(config_dir / hparams.get("--test-config"))
    database_config = OmegaConf.load(config_dir / hparams.get("--database-config"))

    config.update(
        model=model_config,
        dataset=dataset_config,
        runner=runner_config,
        test=test_config,
        database=database_config,
    )
    OmegaConf.set_readonly(config, True)

    return config


def build_model(config: omegaconf.OmegaConf):
    model: DeepHash = DeepHash(config=config)
    weight_filepath = Path(config.test.params.weight_path)
    model.load_state_dict(
        torch.load(str(weight_filepath), map_location=torch.device("cpu")), strict=False
    )

    return model


def build_dataset(config: omegaconf.DictConfig) -> DataLoader:
    preprocessor: torchvision.transforms = transforms.Compose(
        [
            transforms.Resize(227),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset: torch.utils.data = DataSetRegistry.get(config.dataset.type)
    dataset: torch.utils.data = dataset(
        root=config.dataset.params.path.train, train=True, transform=preprocessor, download=True,
    )

    return dataset


def extract_binaryhash(tensor: torch.Tensor):
    float_array: numpy.array = tensor.squeeze().data.numpy()
    bool_array: numpy.array = float_array.astype(np.bool)

    binary_hash: str = "".join([str(int(i)) for i in bool_array])

    return binary_hash


def build_database(dataset: torch.utils.data, model: torch.nn.Module) -> Dict:
    database: Dict = {}

    for idx, (image, label) in enumerate(dataset):
        image_name: str = f"idx_{idx}_label_{label}.png"

        image: torch.Tensor = image.unsqueeze(0)
        coarse_grain, fine_grain = model.inference(image)
        fine_grain: List = fine_grain.squeeze().data.numpy().tolist()

        binaryhash: str = extract_binaryhash(coarse_grain)

        database[binaryhash] = {"name": image_name, "label": label, "fine_grain": fine_grain}

    return database


def save_database(config: omegaconf.DictConfig, database: Dict) -> None:
    save_dir = Path(config.database.params.save_dir)

    with save_dir.open(mode="w") as json_file:
        json.dump(database, json_file)

    return


@profile
def generate_database(hparams: Dict):
    
    config: omegaconf.DictConfig = get_config(hparams=hparams)
    model: DeepHash = build_model(config=config)
    dataset: torch.utils.data = build_dataset(config=config)
    database: Dict = build_database(dataset=dataset, model=model)
    save_database(config=config, database=database)
    print("Generated Database")
