import torch
import torch.nn as nn
import numpy as np
import pytest

from enum import Enum

from utils.general import choose_nonlinearity

class Nonlinearity(Enum):
   TANH = 'tanh'
   RELU = 'relu'
   LRELU = 'leakyrelu'
   SIG = 'sigmoid'

class MLP(nn.Module):
    def __init__(self, indim: int, hiddim: int, outdim:int, n_hidden: int = 1,\
            nonlinearity: Nonlinearity = Nonlinearity.RELU, bias_bool: bool = True):

        # append initial layer and nonlinearity
        self.nonlinearity = choose_nonlinearity(nonlin=nonlinearity)
        layers = [nn.Linear(indim, hiddim), self.nonlinearity]

        # append hidden layers
        for l in range(n_hidden):
            layers.append(nn.Linear(hiddim, hiddim))
            layers.append(self.nonlinearity)

        # append last layer
        layers.append(nn.Linear(hiddim, outdim))
        self.model = torch.nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Method to initial weights properly for each layer.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """"""
        out = self.model(x)
        return out