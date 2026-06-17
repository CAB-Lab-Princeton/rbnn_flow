import os, sys
import numpy as np
import pytest
from unittest.mock import patch
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.rbnn_flow.data.dataset_factories import (
    get_low_dim_dataset,
    build_low_dim_dataloader,
    get_rbnn_dataset,
    build_rbnn_dataloader,
)
from src.rbnn_flow.data.dataset_classes import LowDimDataset, RBNNImageDataset
from src.rbnn_flow.utils.configuration import Config

# ── constants ─────────────────────────────────────────────────────────────────
# N_TRAJ=100 with test_split=0.2, val_split=0.1 gives exact integer splits
# (72 train, 8 val, 20 test) so shuffle_and_split produces exactly 3 pieces.
N_TRAJ   = 100
TRAJ_LEN = 20
SEQ_LEN  = 5
BATCH    = 4
C, W, H  = 3, 32, 32
SPLITS   = ["train", "test", "val"]


# ── shared helpers ────────────────────────────────────────────────────────────

def _config(labeled: bool = True, supervised: bool = True) -> Config:
    return Config(
        dataset={
            'data_dir': 'dummy.npy',
            'test_split': 0.2,
            'val_split': 0.1,
            'seq_len': SEQ_LEN,
            'labeled': labeled,
            'supervised': supervised,
        },
        dataloader={
            'batch_size': BATCH,
            'num_workers': 0,   # no worker processes in tests
            'shuffle': False,
            'pin_memory': False,
            'drop_last': False,
        },
        training={'seed': 0},
    )


def _low_dim_npz() -> dict:
    """Simulates a .npz with 'R' and 'pi' keys (as returned by np.load)."""
    rng = np.random.default_rng(0)
    return {
        'R':  rng.random((N_TRAJ, TRAJ_LEN, 3, 3)),
        'pi': rng.random((N_TRAJ, TRAJ_LEN, 3)),
    }


def _rbnn_npz() -> np.ndarray:
    """Simulates np.load(...)[()]: a 0-d object array wrapping the data dict."""
    rng = np.random.default_rng(0)
    n = N_TRAJ // 3
    inner = {
        'data':   {s: rng.random((n, TRAJ_LEN, C, W, H)).astype(np.float32) for s in SPLITS},
        'labels': {s: np.zeros((n, TRAJ_LEN), dtype=int)                     for s in SPLITS},
        'R_gt':   {s: rng.random((n, TRAJ_LEN, 3, 3))                        for s in SPLITS},
        'pi_gt':  {s: rng.random((n, TRAJ_LEN, 3))                           for s in SPLITS},
    }
    return np.array(inner)  # 0-d object array; inner[()] returns inner


def _low_dim_triple() -> tuple[LowDimDataset, ...]:
    """Pre-built LowDimDataset (train, test, val) for DataLoader tests."""
    rng = np.random.default_rng(0)
    R  = rng.random((N_TRAJ, TRAJ_LEN, 3, 3))
    pi = rng.random((N_TRAJ, TRAJ_LEN, 3))
    make = lambda n: LowDimDataset(data_R=R[:n], data_pi=pi[:n], seq_len=SEQ_LEN)
    return make(72), make(20), make(8)


def _rbnn_triple(**kwargs) -> tuple[RBNNImageDataset, ...]:
    """Pre-built RBNNImageDataset (train, test, val) for DataLoader tests."""
    rng = np.random.default_rng(0)
    imgs = rng.random((N_TRAJ, TRAJ_LEN, C, W, H)).astype(np.float32)
    make = lambda n: RBNNImageDataset(data=imgs[:n], seq_len=SEQ_LEN, **kwargs)
    return make(72), make(20), make(8)


# ── get_low_dim_dataset ───────────────────────────────────────────────────────

class TestGetLowDimDataset:

    def test_returns_three_lowdimdatasets(self):
        with patch('src.rbnn_flow.data.dataset_factories.np.load', return_value=_low_dim_npz()), \
             patch('src.rbnn_flow.data.dataset_factories.setup_reproducibility'):
            datasets = get_low_dim_dataset(_config())
        assert len(datasets) == 3
        assert all(isinstance(ds, LowDimDataset) for ds in datasets)

    def test_seq_len_respected(self):
        with patch('src.rbnn_flow.data.dataset_factories.np.load', return_value=_low_dim_npz()), \
             patch('src.rbnn_flow.data.dataset_factories.setup_reproducibility'):
            train_ds, _, _ = get_low_dim_dataset(_config())
        R, pi = train_ds[0]
        assert R.shape[0]  == SEQ_LEN
        assert pi.shape[0] == SEQ_LEN

    def test_r_shape(self):
        with patch('src.rbnn_flow.data.dataset_factories.np.load', return_value=_low_dim_npz()), \
             patch('src.rbnn_flow.data.dataset_factories.setup_reproducibility'):
            train_ds, _, _ = get_low_dim_dataset(_config())
        R, _ = train_ds[0]
        assert R.shape == (SEQ_LEN, 3, 3)

    def test_split_sizes_sum_to_total(self):
        with patch('src.rbnn_flow.data.dataset_factories.np.load', return_value=_low_dim_npz()), \
             patch('src.rbnn_flow.data.dataset_factories.setup_reproducibility'):
            train_ds, test_ds, val_ds = get_low_dim_dataset(_config())
        window = TRAJ_LEN - SEQ_LEN + 1
        assert len(train_ds) + len(test_ds) + len(val_ds) == N_TRAJ * window

    def test_r_and_pi_trajectory_counts_match(self):
        """R and pi must be split with the same seed so trajectories remain paired."""
        with patch('src.rbnn_flow.data.dataset_factories.np.load', return_value=_low_dim_npz()), \
             patch('src.rbnn_flow.data.dataset_factories.setup_reproducibility'):
            train_ds, _, _ = get_low_dim_dataset(_config())
        assert train_ds.data_R.shape[0] == train_ds.data_pi.shape[0]


# ── build_low_dim_dataloader ──────────────────────────────────────────────────

class TestBuildLowDimDataloader:

    def test_returns_three_dataloaders(self):
        with patch('src.rbnn_flow.data.dataset_factories.get_low_dim_dataset', return_value=_low_dim_triple()):
            loaders = build_low_dim_dataloader(_config())
        assert len(loaders) == 3
        assert all(isinstance(dl, DataLoader) for dl in loaders)

    def test_batch_shape(self):
        with patch('src.rbnn_flow.data.dataset_factories.get_low_dim_dataset', return_value=_low_dim_triple()):
            train_dl, _, _ = build_low_dim_dataloader(_config())
        R_batch, pi_batch = next(iter(train_dl))
        assert R_batch.shape  == (BATCH, SEQ_LEN, 3, 3)
        assert pi_batch.shape == (BATCH, SEQ_LEN, 3)

    def test_dataloader_kwargs_are_independent(self):
        """Mutating config.dataloader after building must not affect the loaders."""
        config = _config()
        with patch('src.rbnn_flow.data.dataset_factories.get_low_dim_dataset', return_value=_low_dim_triple()):
            train_dl, test_dl, val_dl = build_low_dim_dataloader(config)
        config.dataloader['batch_size'] = 999
        assert train_dl.batch_size == BATCH
        assert test_dl.batch_size  == BATCH
        assert val_dl.batch_size   == BATCH


# ── get_rbnn_dataset ──────────────────────────────────────────────────────────

class TestGetRBNNDataset:

    def _get(self, labeled: bool, supervised: bool) -> tuple[RBNNImageDataset, ...]:
        with patch('src.rbnn_flow.data.dataset_factories.np.load', return_value=_rbnn_npz()):
            return get_rbnn_dataset(_config(labeled=labeled, supervised=supervised))

    def test_returns_three_rbnnimage_datasets(self):
        datasets = self._get(labeled=True, supervised=True)
        assert len(datasets) == 3
        assert all(isinstance(ds, RBNNImageDataset) for ds in datasets)

    def test_supervised_and_labeled_returns_four_tuple(self):
        train_ds, _, _ = self._get(labeled=True, supervised=True)
        assert len(train_ds[0]) == 4    # (x, R_gt, pi_gt, label)

    def test_supervised_no_labels_returns_three_tuple(self):
        train_ds, _, _ = self._get(labeled=False, supervised=True)
        assert len(train_ds[0]) == 3    # (x, R_gt, pi_gt)

    def test_labeled_no_supervision_returns_two_tuple(self):
        train_ds, _, _ = self._get(labeled=True, supervised=False)
        assert len(train_ds[0]) == 2    # (x, label)

    def test_no_labels_no_supervision_returns_array(self):
        train_ds, _, _ = self._get(labeled=False, supervised=False)
        assert isinstance(train_ds[0], np.ndarray)  # bare x, not wrapped in a tuple

    def test_seq_len_respected(self):
        train_ds, _, _ = self._get(labeled=False, supervised=False)
        assert train_ds[0].shape[0] == SEQ_LEN

    def test_image_spatial_shape(self):
        train_ds, _, _ = self._get(labeled=False, supervised=False)
        assert train_ds[0].shape == (SEQ_LEN, C, W, H)


# ── build_rbnn_dataloader ─────────────────────────────────────────────────────

class TestBuildRBNNDataloader:

    def test_returns_three_dataloaders(self):
        with patch('src.rbnn_flow.data.dataset_factories.get_rbnn_dataset', return_value=_rbnn_triple()):
            loaders = build_rbnn_dataloader(_config())
        assert len(loaders) == 3
        assert all(isinstance(dl, DataLoader) for dl in loaders)

    def test_image_batch_shape(self):
        with patch('src.rbnn_flow.data.dataset_factories.get_rbnn_dataset',
                   return_value=_rbnn_triple()):
            train_dl, _, _ = build_rbnn_dataloader(_config(labeled=False, supervised=False))
        batch = next(iter(train_dl))
        assert batch.shape == (BATCH, SEQ_LEN, C, W, H)

    def test_supervised_and_labeled_batch_length(self):
        rng = np.random.default_rng(0)
        imgs   = rng.random((N_TRAJ, TRAJ_LEN, C, W, H)).astype(np.float32)
        labels = np.zeros((N_TRAJ, TRAJ_LEN), dtype=int)
        R_gt   = rng.random((N_TRAJ, TRAJ_LEN, 3, 3))
        pi_gt  = rng.random((N_TRAJ, TRAJ_LEN, 3))
        make = lambda n: RBNNImageDataset(
            data=imgs[:n], content_labels=labels[:n],
            R_gt=R_gt[:n], pi_gt=pi_gt[:n], seq_len=SEQ_LEN
        )
        triple = make(72), make(20), make(8)
        with patch('src.rbnn_flow.data.dataset_factories.get_rbnn_dataset', return_value=triple):
            train_dl, _, _ = build_rbnn_dataloader(_config(labeled=True, supervised=True))
        batch = next(iter(train_dl))
        assert len(batch) == 4  # (x, R_gt, pi_gt, label)

    def test_dataloader_kwargs_are_independent(self):
        """Mutating config.dataloader after building must not affect the loaders."""
        config = _config()
        with patch('src.rbnn_flow.data.dataset_factories.get_rbnn_dataset', return_value=_rbnn_triple()):
            train_dl, test_dl, val_dl = build_rbnn_dataloader(config)
        config.dataloader['batch_size'] = 999
        assert train_dl.batch_size == BATCH
        assert test_dl.batch_size  == BATCH
        assert val_dl.batch_size   == BATCH
