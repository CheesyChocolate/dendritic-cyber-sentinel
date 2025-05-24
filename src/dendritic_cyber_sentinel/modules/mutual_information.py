"""
Mutual Information calculation module.

This module implements mutual information calculation between two signals or images,
originally developed by Jose Delpiano (2015) and modified by Andrew Hill (2011).
"""

import numpy as np
from typing import Union, Optional


def hist2d(A: np.ndarray, B: np.ndarray, L: int = 256) -> np.ndarray:
    """
    Calculate the joint histogram of two images or signals.
    
    This is a Python implementation of the MATLAB hist2.m function.
    
    Parameters:
    -----------
    A : np.ndarray
        First input signal/image
    B : np.ndarray  
        Second input signal/image
    L : int, optional
        Number of bins for each matrix (default: 256)
        
    Returns:
    --------
    np.ndarray
        Joint histogram of matrices A and B
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    
    # Handle empty arrays
    if A.size == 0 or B.size == 0:
        return np.zeros((L, L))
    
    # Get min/max values
    ma = np.min(A)
    MA = np.max(A)
    mb = np.min(B)
    MB = np.max(B)
    
    # Handle constant arrays
    if MA == ma:
        MA = ma + 1
    if MB == mb:
        MB = mb + 1
    
    # Scale and round to fit in {0,...,L-1}
    A_scaled = np.round((A - ma) * (L - 1) / (MA - ma))
    B_scaled = np.round((B - mb) * (L - 1) / (MB - mb))
    
    # Ensure values are within bounds
    A_scaled = np.clip(A_scaled, 0, L - 1).astype(int)
    B_scaled = np.clip(B_scaled, 0, L - 1).astype(int)
    
    # Initialize histogram
    n = np.zeros((L, L))
    
    # Calculate joint histogram
    for i in range(L):
        mask = (A_scaled.flatten() == i)
        if np.any(mask):
            B_subset = B_scaled.flatten()[mask]
            hist_counts = np.bincount(B_subset, minlength=L)
            n[i, :len(hist_counts)] = hist_counts[:L]
    
    return n


def mutual_information(A: np.ndarray, B: np.ndarray, L: int = 256) -> float:
    """
    Calculate mutual information between two images or signals.
    
    This is a Python implementation of the MATLAB mi.m function.
    
    Parameters:
    -----------
    A : np.ndarray
        First input signal/image
    B : np.ndarray
        Second input signal/image  
    L : int, optional
        Number of bins for histograms (default: 256)
        
    Returns:
    --------
    float
        Mutual information value
        
    Notes:
    ------
    Assumption: 0*log(0)=0
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    
    # Handle empty arrays
    if A.size == 0 or B.size == 0:
        return 0.0
    
    # Calculate marginal histograms
    na, _ = np.histogram(A.flatten(), bins=L, range=(np.min(A), np.max(A)))
    na_sum = np.sum(na)
    if na_sum == 0:
        return 0.0
    na = na / na_sum
    
    nb, _ = np.histogram(B.flatten(), bins=L, range=(np.min(B), np.max(B)))
    nb_sum = np.sum(nb)
    if nb_sum == 0:
        return 0.0
    nb = nb / nb_sum
    
    # Calculate joint histogram
    n2 = hist2d(A, B, L)
    n2_sum = np.sum(n2)
    if n2_sum > 0:
        n2 = n2 / n2_sum
    
    # Calculate mutual information
    I = _minf(n2, np.outer(na, nb))
    
    return np.sum(I)


def _minf(pab: np.ndarray, papb: np.ndarray) -> np.ndarray:
    """
    Helper function for mutual information calculation.
    
    Parameters:
    -----------
    pab : np.ndarray
        Joint probability distribution
    papb : np.ndarray
        Product of marginal probability distributions
        
    Returns:
    --------
    np.ndarray
        Mutual information components
    """
    # Find support (avoid log(0))
    threshold = 1e-12
    mask = (papb > threshold) & (pab > threshold)
    
    # Initialize result
    result = np.zeros_like(pab)
    
    # Calculate mutual information where both probabilities are non-zero
    if np.any(mask):
        result[mask] = pab[mask] * np.log2(pab[mask] / papb[mask])
    
    return result


# Alias for compatibility
mi = mutual_information 