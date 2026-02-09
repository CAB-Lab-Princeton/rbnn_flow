from abc import ABC, abstractmethod
from typing import Callable


import numpy as np
import torch
from torch.nn import Module, Parameter

from src.rbnn_flow.nn.mlp import MLP
from src.rbnn_flow.utils.maths import pd_matrix

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
    def __init__(self, integrator: Callable = None, network: Module = None):
        super.__init__(integrator)
        assert network != None
        self.network = network

    def __call__(self, x: torch.Tensor, dt: float, seq_len: int = 3) -> torch.Tensor:
        """"""
        split_ind = x.shape[-1] // 2
        xpos, xvel = x[..., :split_ind], x[..., split_ind:]
        xpos_init, xvel_init = xpos[..., 0, :], xvel[..., 0, :]

        x_init = torch.concat((xpos_init, xvel_init), dim=-1)
        xpred = self.network(x_init, dt, seq_len)

        return xpred


class SO3DynamicsModel(DynamicsModel):
    def __init__(self, integrator: Callable, dtype = torch.float32,
                        device: str = 'cuda:0', gravity: bool = False):
        super.__init__(integrator)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.gravity = gravity

        self._initialize_dynamics_params()
    
    def _initialize_dynamics_params(self):
        """"""
        self.I_diag = Parameter(torch.rand(3, dtype=self.dtype) / torch.sqrt(3), requires_grad=True).to(self.device)
        self.I_off_diag = Parameter(torch.rand(3, dtype=self.dtype) / torch.sqrt(3), requires_grad=True).to(self.device)

        if self.gravity:
            self.V = MLP(indim=9, hiddim=250, outdim=1,
                         n_hidden=2, bias_bool=False)
            
    def _compute_moment_of_inertia(self) -> torch.Tensor:
        """"""
        moi = pd_matrix(diag=self.I_diag, off_diag=self.I_off_diag).to(self.device)
        return moi
    
    def __call__(self, x: torch.Tensor, dt: float, seq_len: int) -> torch.Tensor:
        """"""
        # compute moi 
        moi = self._compute_moment_of_inertia()

        # compute init
        x_init, xdot_init = x[0][..., 0, :], x[1][..., 0, :]

        # predict
        if self.gravity:
            V = self.V
        else:
            V = None
        
        x_pred, xdot_pred = self.integrator(x_init, xdot_init, V, moi, dt, seq_len)
        return (x_pred, xdot_pred)