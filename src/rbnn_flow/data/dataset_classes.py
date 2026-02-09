"""
Author(s): Justice Mason (jjmason)

"""
import numpy as np

from typing import Union, Optional, List
from torch.utils.data import DataLoader, Dataset
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Measurement Dataset Class
class LowDimDataset(BaseModel, Dataset):
    """
    Dataset class specfic to the rigid body ODE network.
    
    ...
    
    **Attributes:**
    - `data` (np.ndarray): Full trajectories generated using Euler's equations.
    - `seq_len` (int, default=2): Length of each sample trajectory in the dataset.
        
    **Methods:**
    - __len__()
    - __getitem__(idx)
    
    **Notes:**
    """
    
    data_R: np.ndarray = Field(..., description="Numpy array containing full trajectories for .")
    data_pi: np.ndarray = Field(..., description="Numpy array containing full trajectories for angular momentum.")
    seq_len: int = Field(gt=0, default=2)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __len__(self) -> int:
        """
        Computes total number of trajectory samples of length seq_len in the dataset.
        
        ...
        
        """
        num_traj, traj_len, _, _  = self.data_R.shape
        num_samples = num_traj * (traj_len - self.seq_len + 1)
        return num_samples
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, ...]:
        """
        Returns specific trajectory subsequence corresponding to idx.
        
        ...
        
        **Args:**
        - `idx` (int): Sample's index.
        
        **Returns:**
        - sample (torch.Tensor): Returns sample of length seq_len with 
        
        **Notes:**
        """
        assert idx < self.__len__(), "Index is out of range."
        _, traj_len, _, _  = self.data_R.shape
        
        traj_idx, seq_idx = divmod(idx, traj_len - self.seq_len + 1)
        sample_R = self.data_R[traj_idx, seq_idx:seq_idx+self.seq_len, ...]
        sample_pi = self.data_pi[traj_idx, seq_idx:seq_idx+self.seq_len, ...]
        
        sample = (sample_R, sample_pi)
        
        return sample

# Image Dataset Class
class RBNNImageDataset(BaseModel, Dataset):
    """
    Dataset class for image + content information + GT project. Used to join separate datasets.
    
    ...
    
   **Attributes:**
    -   `data` (np.ndarray): Full trajectories generated using Euler's equations.
    -   `content_labels` (List[str]): Content labels of 
    -   `seq_len` (int, default=2): Length of each sample trajectory in the dataset.
    -   `R_gt` (np.ndarray):
    -   `pi_gt` (np.ndarray):
        
    **Methods:**
    -   `__len__()`
    -   `__getitem__(idx)`
    
    **Notes:**
    """
    data: np.ndarray = Field(..., description="Numpy array of full trajectories generated using Euler's equations.")
    content_labels: Optional[np.ndarray] = Field(None, description="Content labels attached to each example -- optional.")
    R_gt: Optional[np.ndarray] = Field(None, description="Numpy array of full trajectory of ground-truth orientation matrices.")
    pi_gt: Optional[np.ndarray] = Field(None, description="Numpy array of full trajectory of ground-truth angular momentum.")
    seq_len: int = Field(gt=0, default=2, description="Sequence length of each traning example.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __len__(self) -> int:
        """
        """
        num_traj, traj_len, _, _, _ = self.data.shape
        length = num_traj * (traj_len - self.seq_len + 1)  
        return length
        
    def __getitem__(self, idx) -> tuple[Union[np.ndarray, str], ...]:
        """
        """
        assert idx < self.__len__(),  "Index is out of range."
        traj_len = self.data.shape[1]
        traj_idx, seq_idx = divmod(idx, traj_len - self.seq_len + 1)
        
        if self.content_labels is None:
            if (self.R_gt is None) or (self.pi_gt is None):
                sample = self.data[traj_idx, seq_idx:seq_idx+self.seq_len,...]
                return sample
            else:
                sample_x = self.data[traj_idx, seq_idx:seq_idx+self.seq_len,...]
                sample_R_gt = self.R_gt[traj_idx, seq_idx:seq_idx+self.seq_len,...]
                sample_pi_gt = self.pi_gt[traj_idx, seq_idx:seq_idx+self.seq_len,...]
                return (sample_x, sample_R_gt, sample_pi_gt)
        else:
            if (self.R_gt is None) or (self.pi_gt is None):
                sample_x = self.data[traj_idx, seq_idx:seq_idx+self.seq_len,...]
                sample_y = self.content_labels[traj_idx, seq_idx:seq_idx+self.seq_len]
                return (sample_x, sample_y)
            else:
                sample_x = self.data[traj_idx, seq_idx:seq_idx+self.seq_len,...]
                sample_R_gt = self.R_gt[traj_idx, seq_idx:seq_idx+self.seq_len,...]
                sample_pi_gt = self.pi_gt[traj_idx, seq_idx:seq_idx+self.seq_len,...]
                sample_y = self.content_labels[traj_idx, seq_idx:seq_idx+self.seq_len]
                return (sample_x, sample_R_gt, sample_pi_gt, sample_y)