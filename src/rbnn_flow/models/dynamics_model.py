from abc import ABC, abstractmethod
from typing import Callable
import math

import torch
import torch.nn as nn
from torch.nn import Module, Parameter

from src.rbnn_flow.nn.mlp import MLP
from src.rbnn_flow.utils.maths import pd_matrix
from src.rbnn_flow.utils.velocity_estimator import compute_angular_velocity

class DynamicsModel(ABC):
    def __init__(self, integrator: Callable = None):
        self.integrator = integrator

    @abstractmethod
    def _prediction_loop(self, x0, seq_len, dt):
        """"""
        pass

    @abstractmethod
    def estimate_velocity(self, x):
        """"""
        pass

    def __call__(self, x):
        """"""
        pass 


class VanilaNNDynamicsModel(DynamicsModel):
    def __init__(self, 
                network: Module, 
                integrator: Callable = None):
        super().__init__(integrator)
        self.network = network

    def __call__(self, x: torch.Tensor, dt: float, seq_len: int = 3) -> torch.Tensor:
        """"""
        x_init = x[..., 0, :]
        xpred = self.network(x_init, dt, seq_len)

        return xpred


class SO3DynamicsModel(nn.Module):
    def __init__(self,
                integrator: Callable,\
                velocity_estimator: Callable,\
                dtype: torch.dtype = torch.float32,\
                device: str = 'cuda:0',\
                gravity: bool = False):
        super().__init__(integrator)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.velocity_estimator = velocity_estimator
        self.dtype = dtype
        self.gravity = gravity

        self._initialize_dynamics_params()
    
    def _initialize_dynamics_params(self):
        """"""
        scale = math.sqrt(3)
        self.I_diag = Parameter(torch.rand(3, dtype=self.dtype, device=self.device) / scale)
        self.I_off_diag = Parameter(torch.rand(3, dtype=self.dtype, device=self.device) / scale)

        self.V = MLP(indim=9, hiddim=250, outdim=1, n_hidden=2, bias_bool=False) if self.gravity else None
            
    def _compute_moment_of_inertia(self) -> torch.Tensor:
        """"""
        moi = pd_matrix(diag=self.I_diag, off_diag=self.I_off_diag)
        return moi
    
    def estimate_velocity(self, x: torch.Tensor, dt: float = 1e-3) -> torch.Tensor:
        return self.estimate_velocity(x, dt)
    
    def forward(self, x: torch.Tensor, dt: float, seq_len: int) -> torch.Tensor:
        """"""
        # ---  Compute MOI --- 
        moi = self._compute_moment_of_inertia()

        # --- Compute Initialization ---
        x_init, xdot_init = x[0][..., 0, :], x[1][..., 0, :]
        V = self.V if self.gravity else None
        
        x_pred, xdot_pred = self.integrator(x_init, xdot_init, V, moi, dt, seq_len)
        return (x_pred, xdot_pred)