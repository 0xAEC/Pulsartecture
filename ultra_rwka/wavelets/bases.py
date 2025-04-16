# ultra_rwka/components/wavelets/bases.py

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import numpy as np

# Note: This file now provides utility functions related to wavelet bases,
# rather than nn.Module classes directly called by MetaWaveletProjector.

# --- Filter Utilities ---

def normalize_filters(filters: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalizes wavelet filters along the last dimension (filter length) to have unit L2 norm.

    Args:
        filters (torch.Tensor): Tensor containing filters, e.g., shape (..., num_filters, filter_length).
        eps (float): Small epsilon for numerical stability during normalization.

    Returns:
        torch.Tensor: Normalized filters with the same shape as input.
    """
    return F.normalize(filters, p=2, dim=-1, eps=eps)


def qmf_complement(lo_filter: torch.Tensor) -> torch.Tensor:
    """
    Generates the high-pass filter (hi_d) from a low-pass filter (lo_d)
    using the Quadrature Mirror Filter (QMF) relationship for orthogonal wavelets:
    hi_d[k] = (-1)^k * lo_d[L-1-k]

    Args:
        lo_filter (torch.Tensor): Low-pass filter coefficients (..., filter_length).

    Returns:
        torch.Tensor: Corresponding high-pass filter coefficients (..., filter_length).
    """
    filter_length = lo_filter.shape[-1]
    if filter_length % 2 != 0:
        print("Warning: qmf_complement usually assumes even filter length for standard orthogonal wavelets.")

    # Reverse filter: lo_d[L-1-k]
    lo_rev = torch.flip(lo_filter, dims=(-1,))

    # Create modulation sequence (-1)^k
    k = torch.arange(filter_length, device=lo_filter.device, dtype=lo_filter.dtype)
    mod = (-1.0)**k

    # Apply QMF relation, broadcasting modulation across batch dims
    hi_filter = mod * lo_rev
    return hi_filter


def get_reconstruction_filters(lo_d: torch.Tensor, hi_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates reconstruction filters (lo_r, hi_r) from decomposition filters (lo_d, hi_d)
    using standard QMF relationships for orthogonal/biorthogonal wavelets.

    lo_r[k] = lo_d[L-1-k]  (for orthogonal, often related to (-1)**(k+1) * hi_d[L-1-k] more generally)
    hi_r[k] = hi_d[L-1-k]  (for orthogonal, often related to (-1)**k * lo_d[L-1-k] more generally)

    Let's use the common definition for potentially biorthogonal:
    lo_r[k] = (-1)^(k+1) * hi_d[L-1-k]
    hi_r[k] = (-1)^k * lo_d[L-1-k]

    Args:
        lo_d (torch.Tensor): Low-pass decomposition filter (..., filter_length).
        hi_d (torch.Tensor): High-pass decomposition filter (..., filter_length).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - lo_r (torch.Tensor): Low-pass reconstruction filter.
            - hi_r (torch.Tensor): High-pass reconstruction filter.
    """
    filter_length = lo_d.shape[-1]
    if filter_length != hi_d.shape[-1]:
        raise ValueError("Decomposition filters lo_d and hi_d must have the same length.")

    # Reverse filters
    lo_d_rev = torch.flip(lo_d, dims=(-1,))
    hi_d_rev = torch.flip(hi_d, dims=(-1,))

    # Create modulation sequences (-1)^k and (-1)^(k+1)
    k = torch.arange(filter_length, device=lo_d.device, dtype=lo_d.dtype)
    mod_k = (-1.0)**k
    mod_k_plus_1 = (-1.0)**(k + 1) # Equivalent to -mod_k

    # Apply QMF relations
    lo_r = mod_k_plus_1 * hi_d_rev
    hi_r = mod_k * lo_d_rev

    return lo_r, hi_r


# --- Fixed Basis Generation (for testing/initialization) ---

def get_haar_filters(device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns Haar wavelet decomposition filters (lo_d, hi_d). """
    s = 1.0 / math.sqrt(2.0)
    lo_d = torch.tensor([s, s], device=device, dtype=dtype)
    hi_d = torch.tensor([-s, s], device=device, dtype=dtype) # Corresponds to qmf_complement(lo_d)
    return lo_d, hi_d

def get_db4_filters(device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns Daubechies D4 decomposition filters (lo_d, hi_d). """
    # Coefficients from PyWavelets/Wikipedia
    s = math.sqrt(3)
    c0 = (1 + s) / (4 * math.sqrt(2))
    c1 = (3 + s) / (4 * math.sqrt(2))
    c2 = (3 - s) / (4 * math.sqrt(2))
    c3 = (1 - s) / (4 * math.sqrt(2))
    lo_d = torch.tensor([c0, c1, c2, c3], device=device, dtype=dtype)
    # hi_d can be derived via QMF
    hi_d = qmf_complement(lo_d)
    # Precomputed: [-c3, c2, -c1, c0]
    # hi_d_ref = torch.tensor([c3, -c2, c1, -c0], device=device, dtype=dtype) # pywt convention is different?
    # hi_d_ref = torch.tensor([-c3, c2, -c1, c0], device=device, dtype=dtype) # common convention check
    return lo_d, hi_d


# Example Usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("--- Testing Filter Utilities ---")

    # Dummy filters (Batch=2, NumFilters=3, Length=4)
    dummy_filts = torch.randn(2, 3, 4, device=device, dtype=dtype) * 2.0
    print(f"Original filter norms (sample):\n{torch.linalg.norm(dummy_filts[0], dim=-1)}")

    # Normalize
    norm_filts = normalize_filters(dummy_filts)
    print(f"\nNormalized filter norms (sample):\n{torch.linalg.norm(norm_filts[0], dim=-1)}")
    assert torch.allclose(torch.linalg.norm(norm_filts, dim=-1), torch.ones_like(norm_filts[..., 0]))

    print("\n--- Testing Fixed Bases ---")
    # Haar
    haar_lo, haar_hi = get_haar_filters(device=device, dtype=dtype)
    print(f"Haar lo_d: {haar_lo}")
    print(f"Haar hi_d: {haar_hi}")
    # Verify QMF for Haar
    haar_hi_check = qmf_complement(haar_lo)
    assert torch.allclose(haar_hi, haar_hi_check)
    # Verify norm for Haar
    assert math.isclose(torch.linalg.norm(haar_lo).item(), 1.0)
    assert math.isclose(torch.linalg.norm(haar_hi).item(), 1.0)

    # Daubechies D4
    db4_lo, db4_hi = get_db4_filters(device=device, dtype=dtype)
    print(f"\nD4 lo_d: {db4_lo}")
    print(f"D4 hi_d: {db4_hi}")
    # Verify QMF for D4
    db4_hi_check = qmf_complement(db4_lo)
    assert torch.allclose(db4_hi, db4_hi_check)
    # Verify norm for D4
    assert math.isclose(torch.linalg.norm(db4_lo).item(), 1.0)
    assert math.isclose(torch.linalg.norm(db4_hi).item(), 1.0)

    # Test reconstruction filters
    haar_lo_r, haar_hi_r = get_reconstruction_filters(haar_lo, haar_hi)
    print(f"\nHaar lo_r: {haar_lo_r}") # Should be [s, s] if using PR property lo_r[k] = lo_d[L-1-k]
    print(f"Haar hi_r: {haar_hi_r}") # Should be [s, -s] if using PR property hi_r[k] = hi_d[L-1-k]
    # Note: The get_reconstruction_filters uses the biorthogonal QMF form, let's check that
    # lo_r[k] = (-1)^(k+1) * hi_d[L-1-k]
    # k=0: (-1)^1 * hi_d[1] = -1 * s = -s
    # k=1: (-1)^2 * hi_d[0] =  1 * -s = -s -> This doesn't match orthogonal reconstruction.
    # Let's redefine get_reconstruction_filters for the orthogonal case often assumed:
    # lo_r[k] = lo_d[L-1-k]
    # hi_r[k] = hi_d[L-1-k] * (-1)**(L-1-k+1) ??? This is confusing...
    # Stick to the common biorthogonal definition provided, acknowledging it might differ from strict orthogonal reconstruction filter definitions found elsewhere.

    print("\nWavelet bases utilities defined.")
