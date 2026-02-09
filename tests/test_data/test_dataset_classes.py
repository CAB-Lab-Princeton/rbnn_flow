import os, sys
import numpy
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.rbnn_flow.utils.configuration import get_config
from src.rbnn_flow.data.dataset_factories import build_rbnn_dataloader

CONFIG_FILE_PATH = "/n/fs/masonj687/rbnn_flow/tests/test_data/test_config.yaml"

def test_supervised_and_labels():
    """"""
    config = get_config(config_file_path=CONFIG_FILE_PATH)
    config.dataset['supervised'] = True
    config.dataset['labeled'] = True 

    dataloaders = build_rbnn_dataloader(config=config)
    traindl, _, _ = dataloaders

    x = next(iter(traindl))
    assert len(x) == 4
    
def test_content_labels():
    """"""
    config = get_config(config_file_path=CONFIG_FILE_PATH)
    config.dataset['supervised'] = False
    config.dataset['labeled'] = True 

    dataloaders = build_rbnn_dataloader(config=config)
    traindl, _, _ = dataloaders

    x = next(iter(traindl))
    assert len(x) == 2

def test_only_images():
    """"""
    config = get_config(config_file_path=CONFIG_FILE_PATH)
    config.dataset['supervised'] = False
    config.dataset['labeled'] = False 

    dataloaders = build_rbnn_dataloader(config=config)
    traindl, _, _ = dataloaders

    x = next(iter(traindl))
    assert not isinstance(x, list)

def test_ground_truth():
    """"""
    config = get_config(config_file_path=CONFIG_FILE_PATH)
    config.dataset['supervised'] = True
    config.dataset['labeled'] = False 

    dataloaders = build_rbnn_dataloader(config=config)
    traindl, _, _ = dataloaders

    x = next(iter(traindl))
    assert len(x) == 3


if __name__ == "__main__":
    test_supervised_and_labels()
    test_content_labels()
    test_only_images()
    test_ground_truth()