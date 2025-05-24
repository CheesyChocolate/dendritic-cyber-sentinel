"""
Wavelet utilities for the Dendritic Cyber Sentinel.

This module provides custom wavelet transform implementations needed for the MRA S-dDCA algorithm.
"""

import numpy as np
import pywt


def modwt(data: np.ndarray, wavelet: str, level: int = None) -> list:
    """
    Maximal Overlap Discrete Wavelet Transform (MODWT).
    
    This is a custom implementation of MODWT since PyWavelets doesn't provide it natively.
    The implementation is based on the algorithm described in:
    Percival, D. B., & Walden, A. T. (2000). Wavelet methods for time series analysis.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    wavelet : str
        Wavelet to use (e.g., 'db1', 'sym2')
    level : int, optional
        Number of decomposition levels
        
    Returns:
    --------
    list
        List of MODWT coefficients at each level
    """
    # Convert input to numpy array
    data = np.asarray(data)
    
    # Get wavelet filters
    wavelet_obj = pywt.Wavelet(wavelet)
    h = wavelet_obj.dec_lo  # Low-pass (scaling) filter
    g = wavelet_obj.dec_hi  # High-pass (wavelet) filter
    
    # Normalize filters for MODWT (divide by sqrt(2))
    h_tilde = h / np.sqrt(2)
    g_tilde = g / np.sqrt(2)
    
    # Determine maximum level if not specified
    if level is None:
        level = int(np.floor(np.log2(len(data))))
    
    # Initialize coefficients list
    coeffs = []
    
    # Current data to process
    v_j = data.copy()
    
    # Perform MODWT for each level
    for j in range(1, level + 1):
        # Calculate periodization factor
        L = len(h_tilde)
        N = len(v_j)
        
        # Initialize detail and approximation coefficients
        w_j = np.zeros(N)
        v_j1 = np.zeros(N)
        
        # Apply filters (circular convolution)
        for t in range(N):
            for l in range(L):
                # Circular indexing
                k = (t - l * (2**(j-1))) % N
                
                # Apply high-pass filter for details
                w_j[t] += g_tilde[l] * v_j[k]
                
                # Apply low-pass filter for approximations
                v_j1[t] += h_tilde[l] * v_j[k]
        
        # Store detail coefficients
        coeffs.append(w_j)
        
        # Update data for next level
        v_j = v_j1.copy()
    
    # Add approximation coefficients at the final level
    coeffs.append(v_j)
    
    return coeffs


def modwt_alternative(data: np.ndarray, wavelet: str, level: int = None) -> list:
    """
    Alternative implementation of MODWT using stationary wavelet transform (SWT).
    
    This is a fallback method that uses PyWavelets' SWT which is similar to MODWT
    but with some differences in normalization.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    wavelet : str
        Wavelet to use (e.g., 'db1', 'sym2')
    level : int, optional
        Number of decomposition levels
        
    Returns:
    --------
    list
        List of coefficients at each level
    """
    # Determine maximum level if not specified
    if level is None:
        level = pywt.swt_max_level(len(data))
    else:
        level = min(level, pywt.swt_max_level(len(data)))
    
    # Ensure level is at least 1
    level = max(1, level)
    
    # Compute SWT
    coeffs = pywt.swt(data, wavelet, level=level)
    
    # Extract and return only detail coefficients plus final approximation
    result = [detail for _, detail in coeffs]
    result.append(coeffs[-1][0])  # Add final approximation
    
    return result 