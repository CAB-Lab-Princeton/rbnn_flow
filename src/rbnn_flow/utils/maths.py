from __future__ import annotations

import torch
import numpy as np


def pd_matrix(diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    """
    Constructs a positive-definite matrix via M = L @ L^T + I, where L is
    lower-triangular with `diag` on the diagonal and `off_diag` below it.

    Parameters
    ----------
    diag : torch.Tensor
        Diagonal elements, shape (D,) or (B, D).
    off_diag : torch.Tensor
        Strictly lower-triangular elements, shape (D*(D-1)//2,) or (B, D*(D-1)//2).

    Returns
    -------
    torch.Tensor
        PD matrix of shape (1, D, D) or (B, D, D).
    """
    if diag.dim() == 1:
        diag = diag.unsqueeze(0)
        off_diag = off_diag.unsqueeze(0)

    n = diag.shape[-1]
    rows, cols = torch.tril_indices(n, n, offset=-1, device=diag.device)

    L = torch.diag_embed(diag)      # [B, D, D] — batched diagonal construction
    L[:, rows, cols] = off_diag     # fill strict lower triangle in one shot

    return L @ L.mT + torch.eye(n, dtype=diag.dtype, device=diag.device)


def vee_map(R: torch.Tensor | np.ndarray, mode: str = "torch") -> torch.Tensor | np.ndarray:
    """
    Vee map: extracts the axial vector from a skew-symmetric matrix.
    Supports batched inputs; output device matches input.
    """
    if mode == "torch":
        return torch.stack((-R[..., 1, 2], R[..., 0, 2], -R[..., 0, 1]), dim=-1)
    return np.array([-R[1, 2], R[0, 2], -R[0, 1]])


def hat_map(a: torch.Tensor | np.ndarray, mode: str = "torch") -> torch.Tensor | np.ndarray:
    """
    Hat map: embeds a vector into so(3) as a skew-symmetric matrix.
    Supports batched inputs; output device and dtype match input.
    """
    if mode == "torch":
        zeros = torch.zeros_like(a[..., 0])
        return torch.stack([
            torch.stack([ zeros,       -a[..., 2],  a[..., 1]], dim=-1),
            torch.stack([ a[..., 2],   zeros,      -a[..., 0]], dim=-1),
            torch.stack([-a[..., 1],   a[..., 0],  zeros     ], dim=-1),
        ], dim=-2)
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def skew(v: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(v, torch.Tensor):
        zeros = torch.zeros_like(v[..., 0])
        return torch.stack([
            torch.stack([ zeros,      -v[..., 2],  v[..., 1]], dim=-1),
            torch.stack([ v[..., 2],   zeros,     -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1],   v[..., 0],  zeros    ], dim=-1),
        ], dim=-2)
    zeros = np.zeros_like(v[..., 0])
    return np.stack([
        np.stack([ zeros,      -v[..., 2],  v[..., 1]], axis=-1),
        np.stack([ v[..., 2],   zeros,     -v[..., 0]], axis=-1),
        np.stack([-v[..., 1],   v[..., 0],  zeros    ], axis=-1),
    ], axis=-2)

def make_skew(A: torch.Tensor) -> torch.Tensor:
    """
    Transforms a matrix into a skew-symmetric matrix.
    """
    return 0.5 * (A - A.mT)

def clamp_vectors_batch(vectors: torch.Tensor, max_norm: float, eps: float = 1e-8) -> torch.Tensor:
    # Compute the L2 norm along the last dimension (assuming each row is a vector)
    norms = vectors.norm(p=2, dim=-1, keepdim=True)
    # Compute per-vector scaling factor and ensure no vector gets amplified if its norm is already low.
    scales = torch.clamp(max_norm / (norms + eps), max=1.0)
    return vectors * scales

def logm_taylor(M: torch.Tensor, n: int = 3):
    """"""
    # grab size of matrix
    dim = M.shape[-1]

    # calculate A matrix
    A = torch.einsum('bij, bjk -> bik', M - torch.eye(dim, device=M.device), torch.linalg.inv(M + torch.eye(dim, device=M.device)))

    # initialize log
    logM = torch.zeros_like(M).to(M.device)

    for i in range(n):
        idx = 2 * i + 1
        A_i = torch.linalg.matrix_power(A, n=idx)
        
        # update logM
        logM += (1./idx) * A_i
    
    logM = 2 * logM
    return logM

def safe_atanh(x, eps=1e-6):
    # Clamp the input to be within the valid range
    x_clamped = torch.clamp(x, -1 + eps, 1 - eps)
    return torch.atanh(x_clamped)