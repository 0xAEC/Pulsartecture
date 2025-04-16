# ultra_rwka/components/kernels/base_kernels.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import math

class Kernel(nn.Module, ABC):
    """
    Abstract Base Class for kernel functions k(x, y).

    Kernel functions measure the similarity between input points x and y,
    often implicitly mapping them to a high-dimensional Hilbert space.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Computes the kernel matrix between two sets of vectors.

        Given X1 of shape (..., N, D) and X2 of shape (..., M, D),
        this method should compute the kernel matrix K of shape (..., N, M)
        where K[..., i, j] = kernel(X1[..., i, :], X2[..., j, :]).

        Args:
            X1 (torch.Tensor): First batch of input vectors.
            X2 (torch.Tensor): Second batch of input vectors.

        Returns:
            torch.Tensor: The computed kernel matrix (Gram matrix).
        """
        pass

    # Alias __call__ to forward for convenience
    def __call__(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        return self.forward(X1, X2)

    def _validate_inputs(self, X1: torch.Tensor, X2: torch.Tensor):
        """ Basic input validation (optional, can be expanded). """
        if X1.shape[-1] != X2.shape[-1]:
            raise ValueError(f"Input vectors must have the same feature dimension. "
                             f"Got X1 dim {X1.shape[-1]} and X2 dim {X2.shape[-1]}")
        # Note: Broadcasting handles mismatching batch dimensions (...) if needed

class LinearKernel(Kernel):
    """
    Computes the linear kernel: k(x, y) = x^T y.
    """
    def __init__(self):
        super().__init__()

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Computes K[..., i, j] = X1[..., i, :]^T @ X2[..., j, :].

        Args:
            X1 (torch.Tensor): Shape (..., N, D).
            X2 (torch.Tensor): Shape (..., M, D).

        Returns:
            torch.Tensor: Shape (..., N, M).
        """
        self._validate_inputs(X1, X2)
        # Use matmul for batch matrix multiplication
        # X1 @ X2.transpose results in (..., N, D) @ (..., D, M) -> (..., N, M)
        return torch.matmul(X1, X2.transpose(-1, -2))

    def extra_repr(self) -> str:
        return "" # No parameters

class PolynomialKernel(Kernel):
    """
    Computes the polynomial kernel: k(x, y) = (gamma * x^T y + coef0)^degree.
    """
    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0):
        """
        Args:
            degree (int): The degree of the polynomial. Defaults to 3.
            gamma (float): Scaling factor for the dot product. Defaults to 1.0.
            coef0 (float): Additive constant. Defaults to 1.0.
        """
        super().__init__()
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree must be a positive integer.")
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Computes K[..., i, j] = (gamma * (X1[..., i, :]^T @ X2[..., j, :]) + coef0)^degree.

        Args:
            X1 (torch.Tensor): Shape (..., N, D).
            X2 (torch.Tensor): Shape (..., M, D).

        Returns:
            torch.Tensor: Shape (..., N, M).
        """
        self._validate_inputs(X1, X2)
        dot_product = torch.matmul(X1, X2.transpose(-1, -2))
        kernel_val = (self.gamma * dot_product + self.coef0).pow(self.degree)
        return kernel_val

    def extra_repr(self) -> str:
        return f'degree={self.degree}, gamma={self.gamma}, coef0={self.coef0}'


class RBFKernel(Kernel):
    """
    Computes the Radial Basis Function (RBF) kernel (Gaussian kernel):
    k(x, y) = exp(-gamma * ||x - y||^2).

    Note: This `gamma` corresponds to 1 / (2 * sigma^2) in the alternative
    formulation exp(-||x - y||^2 / (2 * sigma^2)).
    This kernel is approximated by RandomFourierFeatures and OrthogonalRandomFeatures
    in feature_maps.py.
    """
    def __init__(self, gamma: float = 1.0):
        """
        Args:
            gamma (float): Scaling factor for the squared Euclidean distance.
                           Must be positive. Defaults to 1.0.
        """
        super().__init__()
        if gamma <= 0:
            raise ValueError("Gamma must be positive for RBF kernel.")
        self.gamma = gamma

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Computes K[..., i, j] = exp(-gamma * ||X1[..., i, :] - X2[..., j, :]||^2).

        Args:
            X1 (torch.Tensor): Shape (..., N, D).
            X2 (torch.Tensor): Shape (..., M, D).

        Returns:
            torch.Tensor: Shape (..., N, M).
        """
        self._validate_inputs(X1, X2)
        # torch.cdist computes pairwise distances. p=2.0 for Euclidean distance.
        # It handles batch dimensions (...) correctly.
        # Shape: (..., N, M)
        sq_dists = torch.cdist(X1, X2, p=2.0).pow(2)
        kernel_val = torch.exp(-self.gamma * sq_dists)
        return kernel_val

    def extra_repr(self) -> str:
        return f'gamma={self.gamma}'


class LaplacianKernel(Kernel):
    """
    Computes the Laplacian kernel: k(x, y) = exp(-gamma * ||x - y||_1).
    """
    def __init__(self, gamma: float = 1.0):
        """
        Args:
            gamma (float): Scaling factor for the L1 distance (Manhattan distance).
                           Must be positive. Defaults to 1.0.
        """
        super().__init__()
        if gamma <= 0:
            raise ValueError("Gamma must be positive for Laplacian kernel.")
        self.gamma = gamma

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Computes K[..., i, j] = exp(-gamma * ||X1[..., i, :] - X2[..., j, :]||_1).

        Args:
            X1 (torch.Tensor): Shape (..., N, D).
            X2 (torch.Tensor): Shape (..., M, D).

        Returns:
            torch.Tensor: Shape (..., N, M).
        """
        self._validate_inputs(X1, X2)
        # torch.cdist with p=1.0 computes pairwise L1 distance.
        l1_dists = torch.cdist(X1, X2, p=1.0)
        kernel_val = torch.exp(-self.gamma * l1_dists)
        return kernel_val

    def extra_repr(self) -> str:
        return f'gamma={self.gamma}'

class SigmoidKernel(Kernel):
    """
    Computes the Sigmoid (Hyperbolic Tangent) kernel: k(x, y) = tanh(gamma * x^T y + coef0).

    Note: This kernel is not always positive definite and might not correspond
    to a valid Reproducing Kernel Hilbert Space (RKHS) for all parameter choices.
    """
    def __init__(self, gamma: float = 1.0, coef0: float = 1.0):
        """
        Args:
            gamma (float): Scaling factor for the dot product. Defaults to 1.0.
            coef0 (float): Additive constant. Defaults to 1.0.
        """
        super().__init__()
        self.gamma = gamma
        self.coef0 = coef0

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Computes K[..., i, j] = tanh(gamma * (X1[..., i, :]^T @ X2[..., j, :]) + coef0).

        Args:
            X1 (torch.Tensor): Shape (..., N, D).
            X2 (torch.Tensor): Shape (..., M, D).

        Returns:
            torch.Tensor: Shape (..., N, M).
        """
        self._validate_inputs(X1, X2)
        dot_product = torch.matmul(X1, X2.transpose(-1, -2))
        kernel_val = torch.tanh(self.gamma * dot_product + self.coef0)
        return kernel_val

    def extra_repr(self) -> str:
        return f'gamma={self.gamma}, coef0={self.coef0}'


# Example Usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Dummy data
    N, M, D = 5, 7, 10
    x1 = torch.randn(N, D, device=device, dtype=dtype)
    x2 = torch.randn(M, D, device=device, dtype=dtype)
    # Batch data
    B = 2
    xb1 = torch.randn(B, N, D, device=device, dtype=dtype)
    xb2 = torch.randn(B, M, D, device=device, dtype=dtype)

    print(f"Input shapes: x1=({N}, {D}), x2=({M}, {D})")
    print(f"Batch input shapes: xb1=({B}, {N}, {D}), xb2=({B}, {M}, {D})")

    # --- Linear Kernel ---
    linear_kernel = LinearKernel()
    print("\n--- Linear Kernel ---")
    print(linear_kernel)
    K_lin = linear_kernel(x1, x2)
    Kb_lin = linear_kernel(xb1, xb2)
    print(f"Output shape (no batch): {K_lin.shape}") # Expected (N, M) = (5, 7)
    print(f"Output shape (batch): {Kb_lin.shape}") # Expected (B, N, M) = (2, 5, 7)
    assert K_lin.shape == (N, M)
    assert Kb_lin.shape == (B, N, M)

    # --- Polynomial Kernel ---
    poly_kernel = PolynomialKernel(degree=2, gamma=0.5, coef0=0.0)
    print("\n--- Polynomial Kernel ---")
    print(poly_kernel)
    K_poly = poly_kernel(x1, x2)
    Kb_poly = poly_kernel(xb1, xb2)
    print(f"Output shape (no batch): {K_poly.shape}") # Expected (N, M) = (5, 7)
    print(f"Output shape (batch): {Kb_poly.shape}") # Expected (B, N, M) = (2, 5, 7)
    assert K_poly.shape == (N, M)
    assert Kb_poly.shape == (B, N, M)

    # --- RBF Kernel ---
    rbf_kernel = RBFKernel(gamma=0.1)
    print("\n--- RBF Kernel ---")
    print(rbf_kernel)
    K_rbf = rbf_kernel(x1, x2)
    Kb_rbf = rbf_kernel(xb1, xb2)
    print(f"Output shape (no batch): {K_rbf.shape}") # Expected (N, M) = (5, 7)
    print(f"Output shape (batch): {Kb_rbf.shape}") # Expected (B, N, M) = (2, 5, 7)
    assert K_rbf.shape == (N, M)
    assert Kb_rbf.shape == (B, N, M)
    print(f"Sample value (RBF): {K_rbf[0, 0].item():.4f}") # Should be between 0 and 1

    # --- Laplacian Kernel ---
    lap_kernel = LaplacianKernel(gamma=0.2)
    print("\n--- Laplacian Kernel ---")
    print(lap_kernel)
    K_lap = lap_kernel(x1, x2)
    Kb_lap = lap_kernel(xb1, xb2)
    print(f"Output shape (no batch): {K_lap.shape}") # Expected (N, M) = (5, 7)
    print(f"Output shape (batch): {Kb_lap.shape}") # Expected (B, N, M) = (2, 5, 7)
    assert K_lap.shape == (N, M)
    assert Kb_lap.shape == (B, N, M)
    print(f"Sample value (Laplacian): {K_lap[0, 0].item():.4f}") # Should be between 0 and 1

    # --- Sigmoid Kernel ---
    sig_kernel = SigmoidKernel(gamma=0.01, coef0=-1.0)
    print("\n--- Sigmoid Kernel ---")
    print(sig_kernel)
    K_sig = sig_kernel(x1, x2)
    Kb_sig = sig_kernel(xb1, xb2)
    print(f"Output shape (no batch): {K_sig.shape}") # Expected (N, M) = (5, 7)
    print(f"Output shape (batch): {Kb_sig.shape}") # Expected (B, N, M) = (2, 5, 7)
    assert K_sig.shape == (N, M)
    assert Kb_sig.shape == (B, N, M)
    print(f"Sample value (Sigmoid): {K_sig[0, 0].item():.4f}") # Should be between -1 and 1
