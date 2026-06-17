from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch
import torch.nn as nn

def compute_feature_size(d_in: int,
                        num_layers: int,
                        downsample: bool = True,
                        pad: int = 1,
                        k_size: int = 3,
                        stride: int = 2):
    """"""
    while num_layers > 0:
        if downsample:
            d_out = np.floor(float(d_in + (2 * pad) - k_size)/stride) + 1.0
        else:
            d_out = (d_in - 1) * stride - 2 * pad + k_size

        num_layers -= 1
        return compute_feature_size(d_out, num_layers, downsample, pad, k_size, stride)
    return int(d_in)

class ResidualBlock(nn.Module):
    """Standard two-layer residual block with batch normalization."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(p=dropout)
        self.relu  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)
    

class DownsampleBlock(nn.Module):
    """
    Standard encoder block -- conv2d --> BatchNorm2d --> activation.
    Stride of the conv2d layer is always set to 2; padding is 1.

    Args:
        c_in: int
        c_out: int
        kernel_size: int
        stride: int
        padding: int
    """
    def __init__(self, 
                 c_in: int,
                 c_out: int,
                 kernel_size: int,
                 stride: int = 2,
                 padding: int = 1,
    ):
        super().__init__()
        self.downsampler = nn.Sequential(
            nn.Conv2d(c_in,
                    c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
            ),
            nn.BatchNorm2d(c_out),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, C_in, W, H)
        Returns:
            x_ds:   (B, C_out, W_out, H_out)
        """
        return self.downsampler(x)
    

class UpsampleBlock(nn.Module):
    """
    Standard upsampling block using Conv2dTranspose.

    Args:
        c_in: int
        c_out: int
        kernel_size: int
        stride: int
        padding: int
    """
    def __init__(self, 
                 c_in: int,
                 c_out: int,
                 kernel_size: int,
                 stride: int = 2,
                 padding: int = 1,
    ):
        super().__init__()
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(c_in,
                    c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
            ),
            nn.BatchNorm2d(c_out),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, C_in, W, H)
        Returns:
            x_us:   (B, C_out, W_out, H_out)
        """
        return self.upsampler(x)
    

class CNNEncoder(nn.Module):
    """
    Standard CNN encoder.
    """
    def __init__(self,
                 in_channels: int,
                 base_channels: int,
                 d_model: int,
                 kernel_size: int = 3,
                 num_layers: int = 2, 
                 dropout: float = 0.1,
                 has_residuals: bool = True,
                 image_size: int = 64,

    ):
        super().__init__()
        # --- Define channels sizes ---
        c_list = base_channels * (2 ** np.arange(num_layers + 1))

        self.initial_downsample = DownsampleBlock(
                            c_in=in_channels,
                            c_out=c_list[0],
                            kernel_size=2*kernel_size+1,
                            stride=2,
                            padding=3,
        )

        if has_residuals:
            modulelist_flattened = chain.from_iterable([[DownsampleBlock(c_list[i], c_list[i+1], kernel_size), ResidualBlock(c_list[i+1], dropout)] for i in range(num_layers)])
            self.downsample_layers  = nn.ModuleList(modulelist_flattened)

        else:
            self.downsample_layers = nn.ModuleList([DownsampleBlock(c_list[i], c_list[i+1], kernel_size) for i in range(num_layers)])
        
        # --- Define the Linear layer ---
        fm_size = compute_feature_size(int(image_size/2), num_layers, True, k_size=kernel_size)
        in_dim = c_list[-1] * (fm_size ** 2)
        hidden_dim = 2 * in_dim if 2*in_dim > 256 else 256

        self.projection = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                        nn.GELU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.GELU(),
                                        nn.Linear(hidden_dim, d_model))
        
        self._out_channels = c_list[-1]   
        self._d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B*T, C, H, W)
        Returns:
            embedding:      (B*T, d_model)
        """
        x = self.initial_downsample(x)

        for layer in self.downsample_layers:
            x = layer(x)

        x = self.projection(x.flatten(1))
        return x
    

class CNNDecoder(nn.Module):
    def __init__(self,
                 base_channels: int,
                 d_model: int,
                 kernel_size: int = 4,
                 num_layers: int = 2, 
                 dropout: float = 0.1,
                 has_residuals: bool = True,
    ):
        super().__init__()

        # --- Define channel sizes ---
        c_list = base_channels * (2 ** np.arange(num_layers + 1))
        c_list = c_list[::-1]

        #  --- Define upsample ---
        self.input_proj = nn.Conv2d(d_model, c_list[0], kernel_size=1)

        if has_residuals:
            modulelist_flattened = chain.from_iterable([[UpsampleBlock(c_list[i], c_list[i+1], kernel_size), ResidualBlock(c_list[i+1], dropout)] for i in range(num_layers)])
            self.upsample_layers = nn.ModuleList(list(modulelist_flattened))
        else:
            self.upsample_layers = nn.ModuleList([UpsampleBlock(c_list[i], c_list[i+1], kernel_size, 2, 1) for i in range(num_layers)])

        self.head = nn.Conv2d(c_list[-1], 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor
        """
        x = self.input_proj(x)

        for layer in self.upsample_layers:
            x = layer(x) 
        return self.head(x)
        