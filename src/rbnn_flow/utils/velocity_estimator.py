import torch
import numpy as np


from src.rbnn_flow.utils.maths import make_skew, clamp_vectors_batch, vee_map

def finite_difference(R: torch.Tensor, dt: float, method: str = "central"):
    """
    Compute the finite difference approximation of dR/dt for a batch of rotation matrices.
    
    Args:
        R (torch.Tensor): Tensor of shape (batch, time, 3, 3), containing a trajectory of rotation matrices.
        dt (float): Time step between consecutive matrices.
        method (str): Differencing method: "forward", "backward", "central", "five-point".
    
    Returns:
        torch.Tensor: Approximated time derivative of R, shape (batch, time-?, 3, 3).
    """
    if method == "forward":
        return (R[:, 1:] - R[:, :-1]) / dt  # Shape (batch, T-1, 3, 3)
    
    elif method == "backward":
        return (R[:, 1:] - R[:, :-1]) / dt  # Shape (batch, T-1, 3, 3) (same as forward but reversed in usage)
    
    elif method == "central":
        return (R[:, 2:] - R[:, :-2]) / (2 * dt)  # Shape (batch, T-2, 3, 3)
    
    elif method == "five-point":
        if R.shape[1] < 5:
            raise ValueError("Five-point stencil requires at least 5 time steps.")
        return (-R[:, 4:] + 8 * R[:, 3:-1] - 8 * R[:, 1:-3] + R[:, :-4]) / (12 * dt)  # Shape (batch, T-4, 3, 3)
    
    else:
        raise ValueError("Invalid method. Choose from 'forward', 'backward', 'central', 'five-point'.")


def compute_angular_velocity(R, dt, method="central"):
    """
    Compute the angular velocity ω(t) from rotation matrices using finite difference methods.

    Args:
        R (torch.Tensor): Shape (B, T, 3, 3), batched rotation matrices.
        dt (float): Time step.
        method (str): Finite difference method: "forward", "backward", "central", "five-point".
    
    Returns:
        torch.Tensor: Angular velocity vectors ω(t), shape (B, T-?, 3).
    """
    dR_dt = finite_difference(R, dt, method)  # Compute dR/dt
    R_trimmed = R[:, 1:-1] if method == "central" else R[:, :-1]  # Align with dR/dt shape
    
    if method == "five-point":
        R_trimmed = R[:, 2:-2]

    skew_matrix_ = torch.matmul(R_trimmed.transpose(-2, -1), dR_dt)  # Compute skew-symmetric part
    skew_matrix = make_skew(skew_matrix_)
    return clamp_vectors_batch(vee_map(R=skew_matrix, mode="torch", device=R.device), max_norm=100.)