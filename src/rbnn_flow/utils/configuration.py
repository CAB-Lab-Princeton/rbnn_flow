import yaml
import os
from dataclasses import dataclass
from pathlib import Path

class Config:
    """
    Class object for configuraton files.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

def load_config(config_file_path: str| Path):
    """"""
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_config(config_file_path: str| Path):
    """
    Function to load cofig object from yaml file.

    ** Args: **
        `confi_file_path` (str | Path)

    ** Return: **
        `object` (Config)
    """

    config = load_config(config_file_path=config_file_path)
    return Config(**config)
            
