from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import dataclass

import torch
import numpy as np
from torch.nn import Module

from ..utils.configuration import Config


class PGNNModel(ABC):
    """"""
    def __init__(self, config: ModelConfig):
        
        self.encoder = None
        self.decoder = None
        self.dynamics = None
    
    @abstractmethod
    def _data_preprocess(self, x: torch.Tensor):
        """@private"""
        pass

    @abstractmethod
    def _data_postproccess(self, x: torch.Tensor):
        """@private"""
        pass

    @abstractmethod
    def _encode(self, x: torch.Tensor):
        """@private"""
        pass
    
    @abstractmethod
    def _decode(self, z: torch.Tensor):
        """@private"""
        pass

    @abstractmethod
    def _flow(self, z: torch.Tensor, inverse: bool = False):
        """@private"""
        pass

    @abstractmethod
    def _dynamics_update(self, z0: torch.Tensor):
        """@private"""
        pass

    @abstractmethod
    def train(self, batch: torch.Tensor):
        pass
    
    @abstractmethod
    def inference(self, x: torch.Tensor, dt: float):
        pass 


class RBNNBaseModel(PGNNModel, Module):
    """"""
    def __init__(self, config: Config):
                
                '''
                encoder: Module | Callable = False,
                decoder: Module | Callable = False,
                dynamics: Callable | Module = None,
                high_dimensional: bool = False,
                content_dimension: bool = False):
                '''

        super.__init__()
        self.high_dimensional = config.high_dim
        self.content_dimension = config.has_content_dim
    
        # --- Define encoder ---
        # --- Define decoder ---
        # --- Define dynamics model ---
    
    @classmethod
    def from_config(self):
        pass

    def _data_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape data for convolution operators if the input data is high-dimensional.
        """
        if self.high_dimensional:
            bs, seq_len, C, W, H = x.shape
            return x.reshape(bs * seq_len, 1, C, W, H)
        return x

    def _data_postproccess(self, x: torch.Tensor, batch_size: int, seq_len: int = 2) -> torch.Tensor:
        """
        Convert high-dimensional data from 
        """
        if self.high_dimensional:
            C, W, H = x.shape[1:]
            return x.reshape(batch_size, seq_len, C, W, H)
        return x
    
    def _encode(self, x: torch.Tensor):
        """@private"""
        return self.encoder(**x)
    
    def _decode(self, z: torch.Tensor):
        """@private"""
        return self.decoder(**z)
    
    def _dynamics_update(self, z0: torch.Tensor, **kwargs):
        """@private"""
        return self.dynamics(z0, self.dt, **kwargs)
    
    def forward(self, x: torch.Tensor, dt: float, seq_len: int = 3) -> torch.Tensor:
        """
        
        Args:
            x:      (B, T, C, W, H)
            dt: 
            seq_len: 
        Returns:
            output_dict: 
        """
        batch_size = x.shape[0] 

        # Encode input sequence
        x_obs = self.data_preprocess(x=x)
        if self.high_dimensional and self.content_dimension:
            z_content, R_enc = self.encode(x=x_obs)
            z_content_rs = z_content.reshape(batch_size, -1, self.encoder.content_dim)
        elif self.high_dimensional and not self.content_dimension:
            R_enc = self.encode(x=x_obs)

        R_enc = R_enc.double()
        R_enc_rs = R_enc.reshape(batch_size, -1, 3, 3)

        #  --- Estimate angular velocity ---
        omega_enc = self.velocity_estimator(z_seq=R_enc_rs).double()
        pi_enc = torch.einsum('ij, btj -> bti', moi, omega_enc)

        # Define the intial condition --- R: [B, T, 3, 3] pi: [B, T, 3]
        R_0 = R_enc_rs[:, 0, ...]
        pi_0 = pi_enc[:, 0, ...]

        # Integrate full trajectory
        R_dyn, pi_dyn = self.integrator.integrate(pi_init=pi_enc[:, 0, ...], R_init=R_0, V=self.V, moi=moi, timestep=self.dt, traj_len=seq_len)

        # Reshape dynamics-based prediction
        R_dyn_rs = R_dyn.reshape(-1, 3, 3)
        zC_rs = z_content_rs.reshape(-1, self.encoder.content_dim)

        # Decode autoencoder-based and dynamics-based predictions in a single batched call
        xhat_combined = self.decode(
            z_content=zC_rs.repeat(2, 1),
            z_dyn=torch.cat([R_dyn_rs, R_enc], dim=0)
        )
        xhat_dyn_, xhat_recon_ = xhat_combined.chunk(2, dim=0)

        # Post-process step
        xhat_dyn = self.data_postprocess(xhat=xhat_dyn_, batch_size=batch_size, seq_len=seq_len)
        xhat_recon = self.data_postprocess(xhat=xhat_recon_, batch_size=batch_size, seq_len=seq_len)