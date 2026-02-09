import torch

def pd_matrix(diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    """
    Function constructing postive-definite matrix from diag/off-diag entries.
    
    ...
    
    Parameters
    ----------
    diag : torch.Tensor
        Diagonal elements of PD matrix.
        
    off-diag: torch.Tensor
        Off-diagonal elements of PD matrix.
        
    Returns
    -------
    matrix_pd : torch.Tensor
        Calculated PD matrix.
        
    Notes
    -----
    
    """
    if len(diag.shape)==1:
        batch_dim = 1
        diag_dim = diag.shape[0]
        diag = diag[None, ...]
    else:
        batch_dim  = diag.shape[0]
        diag_dim = diag.shape[1]

    L_ = []
    for i in range(batch_dim):
        L = torch.diag_embed(diag[i, ...])

        ind = np.tril_indices(diag_dim, k=-1)
        flat_ind  = np.ravel_multi_index(ind, (diag_dim, diag_dim))
        
        L = torch.flatten(L, start_dim=0)
        L[flat_ind] = off_diag[i, ...]
        L = torch.reshape(L, (diag_dim, diag_dim))
        
        #import pdb; pdb.set_trace()
        matrix_pd = L @ L.T + (1.0 * torch.eye(3, dtype=diag.dtype, device=diag.device))
        L_.append(matrix_pd)
    
    M = torch.stack(L_, dim=0)
    return M