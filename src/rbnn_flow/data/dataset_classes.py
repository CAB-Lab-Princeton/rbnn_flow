"""
Author(s): Justice Mason (jjmason)

"""
import numpy as np

from typing import Union, Optional
from torch.utils.data import Dataset
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# Measurement Dataset Class
class LowDimDataset(BaseModel, Dataset):
    """
    Dataset class specfic to the rigid body ODE network.

    ...

    **Attributes:**
    - `data_R` (np.ndarray): Full trajectories generated using Euler's equations.
    - `data_pi` (np.ndarray): Full trajectories of angular momentum.
    - `seq_len` (int, default=2): Length of each sample trajectory in the dataset.

    **Methods:**
    - __len__()
    - __getitem__(idx)
    """

    data_R: np.ndarray = Field(..., description="Numpy array containing full trajectories for orientation.")
    data_pi: np.ndarray = Field(..., description="Numpy array containing full trajectories for angular momentum.")
    seq_len: int = Field(gt=0, default=2)
    _num_traj: int = PrivateAttr()
    _traj_len: int = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        self._num_traj, self._traj_len = self.data_R.shape[:2]

    def __len__(self) -> int:
        return self._num_traj * (self._traj_len - self.seq_len + 1)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, ...]:
        if idx >= len(self):
            raise IndexError(f"index {idx} out of range")
        traj_idx, seq_idx = divmod(idx, self._traj_len - self.seq_len + 1)
        sl = slice(seq_idx, seq_idx + self.seq_len)
        return self.data_R[traj_idx, sl], self.data_pi[traj_idx, sl]


# Image Dataset Class
class RBNNImageDataset(BaseModel, Dataset):
    """
    Dataset class for image + content information + GT project. Used to join separate datasets.

    ...

   **Attributes:**
    -   `data` (np.ndarray): Full trajectories of images.
    -   `content_labels` (np.ndarray, optional): Content labels attached to each example.
    -   `seq_len` (int, default=2): Length of each sample trajectory in the dataset.
    -   `R_gt` (np.ndarray, optional): Ground-truth orientation matrices.
    -   `pi_gt` (np.ndarray, optional): Ground-truth angular momentum.

    **Methods:**
    -   `__len__()`
    -   `__getitem__(idx)`
    """
    data: np.ndarray = Field(..., description="Numpy array of full trajectories of images.")
    content_labels: Optional[np.ndarray] = Field(None, description="Content labels attached to each example.")
    R_gt: Optional[np.ndarray] = Field(None, description="Ground-truth orientation matrices.")
    pi_gt: Optional[np.ndarray] = Field(None, description="Ground-truth angular momentum.")
    seq_len: int = Field(gt=0, default=2, description="Sequence length of each training example.")
    _num_traj: int = PrivateAttr()
    _traj_len: int = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        self._num_traj, self._traj_len = self.data.shape[:2]

    def __len__(self) -> int:
        return self._num_traj * (self._traj_len - self.seq_len + 1)

    def __getitem__(self, idx: int) -> tuple[Union[np.ndarray, str], ...]:
        if idx >= len(self):
            raise IndexError(f"index {idx} out of range")
        traj_idx, seq_idx = divmod(idx, self._traj_len - self.seq_len + 1)
        sl = slice(seq_idx, seq_idx + self.seq_len)

        sample_x = self.data[traj_idx, sl]
        has_gt = (self.R_gt is not None) and (self.pi_gt is not None)

        if self.content_labels is None:
            if not has_gt:
                return sample_x
            return (sample_x, self.R_gt[traj_idx, sl], self.pi_gt[traj_idx, sl])
        else:
            if not has_gt:
                return (sample_x, self.content_labels[traj_idx, sl])
            return (sample_x, self.R_gt[traj_idx, sl], self.pi_gt[traj_idx, sl], self.content_labels[traj_idx, sl])
