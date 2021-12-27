import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F


TORCH_VERSION = torch.__version__


class SVD(nn.Module):
    """
    Singular value decomposition layer
    
    Examples
    --------
    >>> A = torch.rand(126, 100, 20).to('cuda')
    >>> U, S, V = SVD()(A)
    >>> A_ = torch.matmul(U, torch.matmul(S, V.transpose(-1, -2)))
    >>> print(torch.dist(A_, A))
    """
    def __init__(self):
        super(SVD, self).__init__()
    
    def forward(self, A):
        """
        Inputs
        ------
        A: [b, m, n]
        
        Outputs
        -------
        U: [b, m, n]
        S: [b, n, n]
        V: [b, n, n]
        """
        return self.svd_(A)
        
    @staticmethod
    def svd_(A):
        """
        Parameters
        ----------
        A: torch.FloatTensor
            A tensor of shape [b, m, n].

        Returns
        -------
        U: [b, m, n]
        S: [b, n, n]
        V: [b, n, n]

        References
        ----------
        1. https://www.youtube.com/watch?v=pSbafxDHdgE&t=205s
        2. https://www2.math.ethz.ch/education/bachelor/lectures/hs2014/other/linalg_INFK/svdneu.pdf
        """
        U, S, V = torch.svd_lowrank(A, q=A.shape[-1])
        S = torch.diag_embed(S)
        return U, S, V


def singular_value_cumsum(S):
    """
    Parameters
    ----------
    S: [b, n, n]
        S is a diagonal matrix whose off-diagonal entries are all equal to zero.
    """
    numerator = torch.diagonal(S, dim1=-2, dim2=-1).cumsum(dim=1)
    denominator = torch.diagonal(S, dim1=-2, dim2=-1).sum(dim=1, keepdim=True)
    return torch.div(numerator, denominator)


def compute_nu(S, k):
    """
    Parameters
    ----------
    S: torch.tensor
        Diagonal matrix of shape [b, n, n]
    k: int
        Rank estimation smaller than n.

    Returns
    -------
    nu: torch.tensor
        Frobenius norm ratio of shape [b, 1]
        It can be futher used to compute the efficient rank estimate (Zhang, 2017)
    """
    S_flat = torch.diagonal(S, dim1=-2, dim2=-1)
    nu = S_flat[:, :k].norm(p=2, dim=1) / S_flat.norm(p=2, dim=1)
    nu = nu.view(-1, 1)
    return nu