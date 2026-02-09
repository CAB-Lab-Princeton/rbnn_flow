"""
Author(s): Justice Mason
"""
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser

from src.rbnn_flow.utils.general import setup_reproducibility

def shuffle_and_split(raw_data: np.ndarray, test_split: float, val_split: float) -> list[np.ndarray]:
    """
    Function to shuffle and split the dataset.

    ...

    **Args:**
    -   `raw_data` (np.ndarray):
    -   `test_split` (float)
    -   `val_split` (float)
    """
    n_traj = raw_data.shape[0]

    test_len = int(test_split * n_traj)
    val_len = int((1. - test_split) * val_split * n_traj)
    train_len = int((1. - test_split) * (1. - val_split) * n_traj)
    
    np.random.shuffle(raw_data)
        
    rd_split = np.split(np.float64(raw_data), [train_len, train_len + val_len, train_len + val_len + test_len], axis=0)
    return rd_split

# Functions to construct datasets

def construct_joint_image_datasets(args: ArgumentParser) -> tuple[np.ndarray, ...]:
    """
    Function to 
    """
    # Pull out args
    data_dir_list = args.data_dir_list
    test_split = args.test_split
    val_split = args.val_split

    # Initialize dictionaries
    x_dict = {'train':[], 'test':[], 'val':[]}
    y_dict = {'train':[], 'test':[], 'val':[]}
    
    # Loop through data directories and load datasets
    class_idx = 0

    for data_dir in tqdm(data_dir_list):

        # Load data
        raw_data = np.load(data_dir + 'data.npy', allow_pickle=True)

        # Set seed for CNNS
        setup_reproducibility(seed=args.seed, cnn_reprod=True)

        # Split and shuffle data
        data_split = shuffle_and_split(raw_data=raw_data, test_split=test_split, val_split=val_split)
        
        x_dict['train'].append(data_split[0])
        x_dict['test'].append(data_split[1])
        x_dict['val'].append(data_split[2])

        y_dict['train'].append(class_idx * np.ones(data_split[0].shape[:2], dtype=int))
        y_dict['test'].append(class_idx * np.ones(data_split[1].shape[:2], dtype=int))
        y_dict['val'].append(class_idx * np.ones(data_split[2].shape[:2], dtype=int))

        class_idx +=1
    
    for key in x_dict.keys():
        x_dict[key] = np.concatenate(x_dict[key], axis=0)
        y_dict[key] = np.concatenate(y_dict[key], axis=0)
    
    return x_dict, y_dict

def construct_joint_datasets_with_gt(args):
    """"""
    # Pull out args
    data_dir_list = args.data_dir_list
    test_split = args.test_split
    val_split = args.val_split

    # Initialize dictionaries
    x_dict = {'train':[], 'test':[], 'val':[]}
    y_dict = {'train':[], 'test':[], 'val':[]}
    R_dict = {'train':[], 'test':[], 'val':[]}
    pi_dict = {'train':[], 'test':[], 'val':[]}
    
    # Loop through data directories and load datasets
    class_idx = 0

    for data_dir in tqdm(data_dir_list):

        # Load data
        data_dir_data = data_dir + '/data.npy'
        data_dir_R = data_dir + '/data_R.npy'
        data_dir_pi = data_dir + '/data_pi.npy'

        raw_data = np.load(data_dir_data, allow_pickle=True)
        raw_R = np.load(data_dir_R, allow_pickle=True)
        raw_pi = np.load(data_dir_pi, allow_pickle=True)

        # Set seed 
        setup_reproducibility(seed=args.seed)
        data_split = shuffle_and_split(raw_data=raw_data, test_split=test_split, val_split=val_split)

        setup_reproducibility(seed=args.seed)
        R_split = shuffle_and_split(raw_data=raw_R, test_split=test_split, val_split=val_split)
        
        setup_reproducibility(seed=args.seed)
        pi_split = shuffle_and_split(raw_data=raw_pi, test_split=test_split, val_split=val_split)

        x_dict['train'].append(data_split[0])
        x_dict['test'].append(data_split[1])
        x_dict['val'].append(data_split[2])

        y_dict['train'].append(class_idx * np.ones(data_split[0].shape[:2], dtype=int))
        y_dict['test'].append(class_idx * np.ones(data_split[1].shape[:2], dtype=int))
        y_dict['val'].append(class_idx * np.ones(data_split[2].shape[:2], dtype=int))

        R_dict['train'].append(R_split[0])
        R_dict['test'].append(R_split[1])
        R_dict['val'].append(R_split[2])

        pi_dict['train'].append(pi_split[0])
        pi_dict['test'].append(pi_split[1])
        pi_dict['val'].append(pi_split[2])

        class_idx +=1
    
    for key in x_dict.keys():
        x_dict[key] = np.concatenate(x_dict[key], axis=0)
        y_dict[key] = np.concatenate(y_dict[key], axis=0)
        R_dict[key] = np.concatenate(R_dict[key], axis=0)
        pi_dict[key] = np.concatenate(pi_dict[key], axis=0)

    return x_dict, y_dict, R_dict, pi_dict
    