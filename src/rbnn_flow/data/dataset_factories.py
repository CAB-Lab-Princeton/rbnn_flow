import numpy as np
from torch.utils.data import DataLoader

from src.rbnn_flow.utils.general import setup_reproducibility
from src.rbnn_flow.utils.data import shuffle_and_split
from src.rbnn_flow.data.dataset_classes import LowDimDataset, RBNNImageDataset
from src.rbnn_flow.utils.configuration import Config

_SPLITS = ["train", "test", "val"]


def _build_dataloaders(datasets: tuple, dataloader_config: dict) -> tuple[DataLoader, ...]:
    return tuple(DataLoader(ds, **dict(dataloader_config)) for ds in datasets)


def get_low_dim_dataset(config: Config) -> tuple[LowDimDataset, ...]:
    """
    Factory function that builds LowDimDatasets for train, test, and validation splits.

    **Args:**
        `config` (Config): Experiment configuration.

    **Returns:**
        `tuple[LowDimDataset, ...]`: Train, test, and validation datasets.
    """
    raw_data = np.load(config.dataset['data_dir'], allow_pickle=True)

    # Reset seed before each split so R and pi are shuffled identically.
    setup_reproducibility(config.training['seed'])
    rd_split_R = shuffle_and_split(raw_data=raw_data['R'],
                                   test_split=config.dataset['test_split'],
                                   val_split=config.dataset['val_split'])

    setup_reproducibility(config.training['seed'])
    rd_split_pi = shuffle_and_split(raw_data=raw_data['pi'],
                                    test_split=config.dataset['test_split'],
                                    val_split=config.dataset['val_split'])

    return tuple(
        LowDimDataset(data_R=rd_split_R[i], data_pi=rd_split_pi[i], seq_len=config.dataset['seq_len'])
        for i in range(len(rd_split_R))
    )


def build_low_dim_dataloader(config: Config) -> tuple[DataLoader, ...]:
    """Factory function for low-dimensional dataloaders (LowDimDataset)."""
    return _build_dataloaders(get_low_dim_dataset(config), config.dataloader)


def get_rbnn_dataset(config: Config) -> tuple[RBNNImageDataset, ...]:
    """
    Factory function that builds RBNNImageDatasets for train, test, and validation splits.

    **Args:**
        `config` (Config): Experiment configuration.

    **Returns:**
        `tuple[RBNNImageDataset, ...]`: Train, test, and validation datasets.
    """
    raw_data = np.load(config.dataset['data_dir'], allow_pickle=True)[()]
    raw_datasets = [raw_data['data'][s] for s in _SPLITS]

    R_gt   = [raw_data['R_gt'][s]    for s in _SPLITS] if config.dataset['supervised'] else [None] * 3
    pi_gt  = [raw_data['pi_gt'][s]   for s in _SPLITS] if config.dataset['supervised'] else [None] * 3
    labels = [raw_data['labels'][s]  for s in _SPLITS] if config.dataset['labeled']    else [None] * 3

    return tuple(
        RBNNImageDataset(data=raw_datasets[i], R_gt=R_gt[i], pi_gt=pi_gt[i],
                         content_labels=labels[i], seq_len=config.dataset['seq_len'])
        for i in range(3)
    )


def build_rbnn_dataloader(config: Config) -> tuple[DataLoader, ...]:
    """Factory function for image dataloaders (RBNNImageDataset)."""
    return _build_dataloaders(get_rbnn_dataset(config), config.dataloader)
