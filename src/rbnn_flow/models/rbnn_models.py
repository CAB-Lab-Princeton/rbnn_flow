from abc import ABC, abstractmethod
from typing import Callable

import torch
import numpy as np
from torch.nn import Module

class PGNNModel(ABC):
    """"""
    def __init__(self,
                encoder: Module | Callable = None,
                decoder: Module | Callable = None,
                dynamics: Callable = None):
        
        self.encoder = encoder
        self.decoder = decoder
        self.dynamics = dynamics
    
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
    def __init__(self, encoder: Module | Callable = False,\
                decoder: Module | Callable = False,\
                dynamics: Callable | Module = None,
                high_dimensional: bool = False):
        super.__init__(encoder, decoder, dynamics)
        assert dynamics != None
        self.high_dimensional = high_dimensional 
    
    def _data_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        if self.high_dimensional:
            pass
        pass

    def _encode(self, x: torch.Tensor):
        """@private"""
        return self.encoder(x)
    
    def _decode(self, z: torch.Tensor):
        """@private"""
        return self.decoder(z)
    
    def _dynamics_update(self, z0: torch.Tensor, **kwargs):
        """@private"""
        return self.dynamics(z0, self.dt, **kwargs)
    
    def forward(self, x: torch.Tensor, dt: float, seq_len: int = 3) -> torch.Tensor:
        """"""
        # encode
        if self.encoder:
            pass

        # dynamics

        # decode
        if self.decoder:
            pass