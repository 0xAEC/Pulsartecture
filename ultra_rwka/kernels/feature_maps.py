# ultra_rwka/components/kernels/feature_maps.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from abc import ABC, abstractmethod

# Helper for stable division
def _safe_divide(num, den, eps=1e-10):
    return num / (den + eps)

# Base class for feature maps
class FeatureMap(nn.Module, ABC):
    """Abstract Base Class for kernel feature map approximations."""
    def __init__(self, in_dim: int, feature_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, feature_dim={self.feature_dim}'

# --- Random Feature Maps ---

@torch.jit.script
def _rff_forward_script(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """ JIT-scriptable core computation for RFF/ORF """
    projection = torch.matmul(x, W) + b
    # Use torch.cat instead of list concatenation for JIT compatibility
    phi = torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)
    phi = phi * scale_factor
    return phi

class RandomFourierFeatures(FeatureMap):
    """
    Approximates the Gaussian RBF kernel exp(-gamma^2||x-y||^2/2) using
    Random Fourier Features (RFFs). [cite: 59, 194]

    Uses standardized sampling N(0, gamma^2 * I) for weights `W` and U(0, 2*pi) for bias `b`.
    """
    def __init__(self,
                 in_dim: int,
                 feature_dim: int,
                 gamma: float = 1.0,
                 learnable: bool = False,
                 device=None,
                 dtype=None):
        """
        Args:
            in_dim (int): Input feature dimension.
            feature_dim (int): Output feature dimension (d_k). Must be even.
            gamma (float): RBF kernel bandwidth parameter (sigma=1/gamma). Defaults to 1.0.
            learnable (bool): If True, make projection matrix W learnable (deviates from standard RFF). Defaults to False.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__(in_dim, feature_dim)
        if feature_dim % 2 != 0:
            raise ValueError(f"feature_dim (d_k={feature_dim}) must be even for RFFs using sin/cos pairs.")

        self.gamma = gamma
        self.learnable = learnable
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        num_projections = feature_dim // 2
        scale = gamma # Standard deviation for Gaussian sampling

        # Sample projection matrix W and bias b
        W_init = torch.randn(in_dim, num_projections, **self.factory_kwargs) * scale
        b_init = torch.rand(num_projections, **self.factory_kwargs) * 2 * math.pi

        if learnable:
            warnings.warn("Making RFF projection matrix W learnable. This deviates from standard RFF theory.")
            self.W = nn.Parameter(W_init)
            self.b = nn.Parameter(b_init)
        else:
            self.register_buffer('W', W_init)
            self.register_buffer('b', b_init)

        # Scaling factor: sqrt(2 / d_k) = sqrt(1 / num_projections)
        self.scale_factor = math.sqrt(1.0 / num_projections)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RFF transformation. Shape: (*, in_dim) -> (*, feature_dim).
        """
        if x.device != self.W.device or x.dtype != self.W.dtype:
            x = x.to(device=self.W.device, dtype=self.W.dtype)

        # Use JIT-scripted function for potential optimization
        return _rff_forward_script(x, self.W, self.b, self.scale_factor)

    def extra_repr(self) -> str:
        return f'{super().extra_repr()}, gamma={self.gamma}, learnable={self.learnable}'


class OrthogonalRandomFeatures(FeatureMap):
    """
    Approximates the Gaussian RBF kernel using Orthogonal Random Features (ORF).
    ORF aims to reduce the variance of the RFF approximation by using orthogonal projection matrices.
    """
    def __init__(self,
                 in_dim: int,
                 feature_dim: int,
                 gamma: float = 1.0,
                 learnable: bool = False, # Orthogonal constraint is hard to maintain if learnable
                 device=None,
                 dtype=None):
        """
        Args:
            in_dim (int): Input feature dimension.
            feature_dim (int): Output feature dimension (d_k). Must be even.
            gamma (float): RBF kernel bandwidth parameter (sigma=1/gamma). Defaults to 1.0.
            learnable (bool): If True, makes W learnable, breaking orthogonality. Defaults to False.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__(in_dim, feature_dim)
        if feature_dim % 2 != 0:
            raise ValueError(f"feature_dim (d_k={feature_dim}) must be even for ORF using sin/cos pairs.")
        if learnable:
            warnings.warn("Making ORF projection matrix W learnable. Orthogonality constraint will likely be broken during training.")

        self.gamma = gamma
        self.learnable = learnable
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        num_projections = feature_dim // 2

        # Generate orthogonal matrix W using QR decomposition of a Gaussian matrix
        # Sample a matrix larger than needed if in_dim < num_projections
        matrix_dim = max(in_dim, num_projections)
        gaussian_matrix = torch.randn(matrix_dim, matrix_dim, **self.factory_kwargs)
        q, _ = torch.linalg.qr(gaussian_matrix)
        W_orth = q[:in_dim, :num_projections] # Select the target dimensions

        # Scale W by gamma * sqrt(input_dim) (different scaling factor theory for ORF)
        # Scaling factor for individual projections ~ N(0, gamma^2), var(W_ij) = gamma^2 / D
        # Another view: sample chi-distributed lengths. Simpler: Scale orthogonal matrix.
        # Let's stick closer to RFF scaling for simplicity here, controlled by gamma.
        # Variance of projection x @ W still depends on ||x||^2 * E[W_i^2]
        W_init = W_orth * gamma # Apply bandwidth scaling

        # Bias b ~ Uniform(0, 2*pi)
        b_init = torch.rand(num_projections, **self.factory_kwargs) * 2 * math.pi

        if learnable:
            self.W = nn.Parameter(W_init)
            self.b = nn.Parameter(b_init)
        else:
            self.register_buffer('W', W_init)
            self.register_buffer('b', b_init)

        # Scaling factor sqrt(2 / d_k) = sqrt(1 / num_projections)
        self.scale_factor = math.sqrt(1.0 / num_projections)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ORF transformation. Shape: (*, in_dim) -> (*, feature_dim).
        """
        if x.device != self.W.device or x.dtype != self.W.dtype:
            x = x.to(device=self.W.device, dtype=self.W.dtype)

        # Use JIT-scripted function for potential optimization
        return _rff_forward_script(x, self.W, self.b, self.scale_factor)

    def extra_repr(self) -> str:
        return f'{super().extra_repr()}, gamma={self.gamma}, learnable={self.learnable}'

# --- Learnable Feature Maps ---

class LearnablePositiveMap(FeatureMap):
    """
    A learnable feature map using an MLP followed by a positivity-ensuring activation.
    Useful when the desired kernel is non-standard or needs to be learned.
    Positivity might be beneficial for FA-KLA implementation details[cite: 78].
    """
    _supported_activations = {'elu+1', 'relu', 'softplus', 'gelu'} # GELU is not strictly positive but often used

    def __init__(self,
                 in_dim: int,
                 feature_dim: int,
                 hidden_dim: int | None = None,
                 num_layers: int = 1,
                 activation: str = 'elu+1',
                 use_bias: bool = True,
                 initialize_weights: bool = True,
                 device=None,
                 dtype=None):
        """
        Args:
            in_dim (int): Input feature dimension.
            feature_dim (int): Output feature dimension (d_k).
            hidden_dim (int | None): Hidden dimension for MLP if num_layers > 1. Defaults to max(in_dim, feature_dim).
            num_layers (int): Number of layers in the MLP (>= 1). Defaults to 1.
            activation (str): Activation function for the *final* layer.
                              Options: 'elu+1', 'relu', 'softplus', 'gelu'. Defaults to 'elu+1'.
            use_bias (bool): Whether to use bias terms in linear layers. Defaults to True.
            initialize_weights (bool): If True, apply Kaiming initialization. Defaults to True.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__(in_dim, feature_dim)
        self.activation_type = activation.lower()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.activation_type not in self._supported_activations:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {self._supported_activations}")

        _hidden_dim = hidden_dim if hidden_dim is not None else max(in_dim, feature_dim)

        layers = []
        current_dim = in_dim

        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            output_dim = feature_dim if is_last_layer else _hidden_dim
            linear_layer = nn.Linear(current_dim, output_dim, bias=use_bias, **self.factory_kwargs)

            if initialize_weights:
                # Kaiming uniform initialization often works well for ReLU/variants
                nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5)) # Default PyTorch Linear init a
                if use_bias:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(linear_layer.bias, -bound, bound)

            layers.append(linear_layer)

            if not is_last_layer:
                # Intermediate activation (GELU is a good default)
                layers.append(nn.GELU())
                current_dim = _hidden_dim
            else:
                # Final activation handled in forward pass
                pass

        self.mlp = nn.Sequential(*layers)

        # Select the final activation function
        if self.activation_type == 'elu+1':
            self.activation_fn = lambda x: F.elu(x) + 1.0
        elif self.activation_type == 'relu':
            self.activation_fn = F.relu
        elif self.activation_type == 'softplus':
            self.activation_fn = F.softplus
        elif self.activation_type == 'gelu': # Included for flexibility, though not strictly positive
            self.activation_fn = F.gelu


    # Apply JIT scripting to the forward method
    # @torch.jit.script_method # -> Error with lambda fn in newer torch, apply script later if needed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the learnable feature map transformation. Shape: (*, in_dim) -> (*, feature_dim).
        """
        if x.device != next(self.parameters()).device or x.dtype != next(self.parameters()).dtype:
             x = x.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)

        projected = self.mlp(x)
        phi = self.activation_fn(projected)
        return phi

    def extra_repr(self) -> str:
        # Find Linear layers to report layers correctly
        num_linear_layers = sum(1 for m in self.mlp if isinstance(m, nn.Linear))
        return f'{super().extra_repr()}, activation={self.activation_type}, mlp_layers={num_linear_layers}'


# Potentially add other feature maps like PolynomialFeatures or LearnableFourierFeatures (SIREN-style)
# class PolynomialFeatures(FeatureMap): ...
# class LearnableFourierFeatures(FeatureMap): ...


# Example usage (can be removed or kept for illustration)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    in_d = 64
    feat_d = 128
    batch_size = 4
    seq_len = 10

    dummy_input = torch.randn(batch_size, seq_len, in_d, device=device, dtype=dtype)

    print("--- Testing RFF ---")
    rff = RandomFourierFeatures(in_d, feat_d, gamma=0.5, device=device, dtype=dtype)
    print(rff)
    rff_out = rff(dummy_input)
    print("Input Shape:", dummy_input.shape)
    print("RFF Output Shape:", rff_out.shape)
    print("RFF Output Sample:", rff_out.view(-1)[:8])

    print("\n--- Testing ORF ---")
    orf = OrthogonalRandomFeatures(in_d, feat_d, gamma=0.5, device=device, dtype=dtype)
    print(orf)
    orf_out = orf(dummy_input)
    print("ORF Output Shape:", orf_out.shape)
    print("ORF Output Sample:", orf_out.view(-1)[:8])
    # Verify orthogonality (approximate due to float precision)
    # W_orf = orf.W.cpu().float().detach().numpy()
    # print("ORF W^T @ W approx Identity:", np.allclose(W_orf.T @ W_orf, np.eye(feat_d // 2), atol=1e-6))


    print("\n--- Testing LearnablePositiveMap (ELU+1) ---")
    lpm_elu = LearnablePositiveMap(in_d, feat_d, activation='elu+1', num_layers=2, device=device, dtype=dtype)
    print(lpm_elu)
    lpm_elu_out = lpm_elu(dummy_input)
    print("LPM (ELU+1) Output Shape:", lpm_elu_out.shape)
    print("LPM (ELU+1) Output Sample:", lpm_elu_out.view(-1)[:8])
    print("LPM (ELU+1) Min Value:", torch.min(lpm_elu_out).item()) # Should be >= 0

    print("\n--- Testing LearnablePositiveMap (ReLU) ---")
    lpm_relu = LearnablePositiveMap(in_d, feat_d, activation='relu', num_layers=1, device=device, dtype=dtype)
    print(lpm_relu)
    lpm_relu_out = lpm_relu(dummy_input)
    print("LPM (ReLU) Output Shape:", lpm_relu_out.shape)
    print("LPM (ReLU) Output Sample:", lpm_relu_out.view(-1)[:8])
    print("LPM (ReLU) Min Value:", torch.min(lpm_relu_out).item()) # Should be >= 0

    # Check JIT scripting works for RFF/ORF forward
    try:
        scripted_rff_forward = torch.jit.script(_rff_forward_script)
        print("\nJIT scripting for _rff_forward_script successful.")
        jit_out = scripted_rff_forward(dummy_input, rff.W, rff.b, rff.scale_factor)
        assert torch.allclose(jit_out, rff_out)
        print("JIT output matches eager output.")
    except Exception as e:
        print("\nJIT scripting for _rff_forward_script failed:", e)
