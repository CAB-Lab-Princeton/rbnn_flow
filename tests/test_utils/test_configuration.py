import os, sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.rbnn_flow.utils.configuration import load_config, get_config

def test_get_config_str():
    """"""
    config_file_path = 'tests/test_utils/test_config.yaml'
    config_dict = load_config(config_file_path=config_file_path)
    config = get_config(config_file_path=config_file_path)

    for key in config_dict.keys():
        assert hasattr(config, key), f"dict has key `{key}` but config does not: {dir(config)}."

def test_get_config_path():
    """"""
    config_file_path = Path('tests/test_utils/test_config.yaml')
    config_dict = load_config(config_file_path=config_file_path)
    config = get_config(config_file_path=config_file_path)

    for key in config_dict.keys():
        assert hasattr(config, key), f"dict has key `{key}` but config does not: {dir(config)}."