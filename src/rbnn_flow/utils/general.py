import numpy as np
import torch
import random

def choose_nonlinearity(nonlin: str = "tanh"):
    """"""
    if nonlin.lower() == 'tanh':
        nl = torch.nn.Tanh()
    elif nonlin.lower() == 'relu':
        nl = torch.nn.ReLU()
    elif nonlin.lower() == 'leakyrelu':
        nl = torch.nn.LeakyReLU()
    elif nonlin.lower() == 'sigmoid':
        nl = torch.nn.Sigmoid()
    
    return nl

# Reproducibility Utility Functions
def setup_reproducibility(seed: int, cnn_reprod: bool = False):
    """
    Function to set-up reproducibility of the networks.

    **Args:**
    -   `seed` (int): 
    -   `cnn_reprod` (bool):

    **Returns:**
    """
    if cnn_reprod:
        # Set benchmark to be constant and deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)