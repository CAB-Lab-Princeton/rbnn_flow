import sys, os
import numpy as np
from copy import deepcopy
from enum import Enum

from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.rbnn_flow.utils.general import setup_reproducibility
from src.rbnn_flow.utils.data import shuffle_and_split

from src.rbnn_flow.data.dataset_classes import LowDimDataset, RBNNImageDataset
from src.rbnn_flow.utils.configuration import Config

# Auxiliary functions

def get_low_dim_dataset(config: Config) -> tuple[LowDimDataset, ...]:
    """
    Factory function that builds LowDimDatasets for train, test, and validation splits.

    ** Args: **
        `args` (ArgumentParser): experiment parameters

    ** Returns: **
        `tuple` (tuple[LowDimDataset]) : Train, test, and validation datasets -- based on user-defined splits.
    """
    # Load data from file
    raw_data = np.load(config.dataset['data_dir'], allow_pickle=True)

    setup_reproducibility(config.training['seed'])
    rd_split_R = shuffle_and_split(raw_data=raw_data['R'], test_split=config.dataset['test_split'],\
                                    val_split=config.dataset['val_split'])

    setup_reproducibility(config.training['seed'])
    rd_split_pi = shuffle_and_split(raw_data=raw_data['pi'], test_split=config.dataset['test_split'],\
                                     val_split=config.dataset['val_split'])

    datasets = tuple([LowDimDataset(data_R=rd_split_R[i], data_pi=rd_split_pi[i], seq_len=config.dataset['seq_len']) for i in range(len(rd_split_R))])
    return datasets

def build_low_dim_dataloader(config: Config) -> tuple[DataLoader, ...]:
    """
    Factory functions for low-dimensional datasets (LowDimDataset).

    ...

    Args:
        args (ArgumentParser): experiment parameters

    """
    trainds, testds, valds = get_low_dim_dataset(config=config)
    
    # Dataset kwargs 
    train_kwargs = config.dataloader
    val_kwargs = config.dataloader
    test_kwargs = config.dataloader

    # Create dataloaders
    traindataloader = DataLoader(
        trainds, **train_kwargs
        )
    testdataloader = DataLoader(
        testds, **test_kwargs
        )
    valdataloader = DataLoader(
        valds, **val_kwargs
        )

    return (traindataloader, testdataloader, valdataloader)

def get_rbnn_dataset(config: Config) -> tuple[RBNNImageDataset, ...]:
    """"""
    dataset_type = ["train", "test", "val"]

    # load raw data
    raw_data = np.load(config.dataset['data_dir'], allow_pickle=True)[()]
    raw_datasets = [raw_data['data'][type_] for type_ in dataset_type]

    if config.dataset['labeled']:
        labels = [raw_data['labels'][type_] for type_ in dataset_type]
        
        if config.dataset['supervised']:
            R_gt = [raw_data['R_gt'][type_] for type_ in dataset_type]
            pi_gt = [raw_data['pi_gt'][type_] for type_ in dataset_type]

            datasets = tuple([RBNNImageDataset(data=raw_datasets[i], R_gt=R_gt[i], pi_gt=pi_gt[i], \
                                               content_labels=labels[i], seq_len=config.dataset['seq_len']) for i in range(len(dataset_type))])
            return datasets
        else:
            datasets = tuple([RBNNImageDataset(data=raw_datasets[i], R_gt=None, pi_gt=None, \
                                               content_labels=labels[i], seq_len=config.dataset['seq_len']) for i in range(len(dataset_type))])
            return datasets
    else:
        if config.dataset['supervised']:
            R_gt = [raw_data['R_gt'][type_] for type_ in dataset_type]
            pi_gt = [raw_data['pi_gt'][type_] for type_ in dataset_type]

            datasets = tuple([RBNNImageDataset(data=raw_datasets[i], R_gt=R_gt[i], pi_gt=pi_gt[i], \
                                               content_labels=None, seq_len=config.dataset['seq_len']) for i in range(len(dataset_type))])
            return datasets
        else:
            datasets = tuple([RBNNImageDataset(data=raw_datasets[i], R_gt=None, pi_gt=None, \
                                               content_labels=None, seq_len=config.dataset['seq_len']) for i in range(len(dataset_type))])
            return datasets
        
def build_rbnn_dataloader(config: Config) -> tuple[DataLoader, ...]:
    """
    Factory function to make image dataloader (RBNNImageDataset).
    """

    trainds, testds, valds = get_rbnn_dataset(config=config)


    # Dataset kwargs 
    train_kwargs = config.dataloader
    val_kwargs = config.dataloader
    test_kwargs = config.dataloader

    # Create dataloaders
    traindataloader = DataLoader(
        trainds, **train_kwargs
        )
    testdataloader = DataLoader(
        testds, **test_kwargs
        )
    valdataloader = DataLoader(
        valds, **val_kwargs
        )

    return (traindataloader, testdataloader, valdataloader)