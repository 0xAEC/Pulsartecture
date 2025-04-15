# ultra_rwka/components/projections.py

import torch
import torch.nn as nn
import math
from typing import Optional, Type, Union

class LinearProjection(nn.Module):
    """
    A flexible linear projection layer with optional activation and layer normalization.
    Includes weight initialization options. Designed for creating Q, K, V projections,
    memory write vectors, i-DAM keys/content projections, etc., as needed by
    various Ultra-RWKA components[cite: 34, 36, 43, 49].
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 use_bias: bool = True,
                 activation_cls: Optional[Type[nn.Module]] = None,
                 use_norm: bool = False,
                 norm_eps: float = 1e-5,
                 norm_affine: bool = True, # Whether LayerNorm has learnable affine params
                 initialize: Union[str, bool] = 'kaiming_uniform', # 'xavier_uniform', 'kaiming_uniform', False
                 device=None,
                 dtype=None):
        """
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            use_bias (bool): Whether the linear layer uses a bias term. Defaults to True.
            activation_cls (Optional[Type[nn.Module]]): Optional activation function class
                (e.g., nn.GELU, nn.ReLU) to apply after the projection. Defaults to None (linear).
            use_norm (bool): Whether to apply Layer Normalization after the projection
                (and before activation if activation exists). Defaults to False.
            norm_eps (float): Epsilon for Layer Normalization. Defaults to 1e-5.
            norm_affine (bool): If True, LayerNorm has learnable affine parameters. Defaults to True.
            initialize (Union[str, bool]): Weight initialization scheme for the linear layer.
                Options: 'xavier_uniform', 'kaiming_uniform', False (uses default PyTorch init).
                Defaults to 'kaiming_uniform'.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.use_norm = use_norm
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # --- Build Layers ---
        # Initialize linear layer
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias, **self.factory_kwargs)

        # Initialize optional Layer Normalization
        self.norm: Optional[nn.LayerNorm] = None
        if use_norm:
            # LayerNorm is typically applied over the feature dimension (-1)
            self.norm = nn.LayerNorm(out_dim, eps=norm_eps, elementwise_affine=norm_affine, **self.factory_kwargs)

        # Initialize optional Activation
        self.activation: Optional[nn.Module] = None
        if activation_cls is not None:
            self.activation = activation_cls()

        # --- Initialization ---
        if isinstance(initialize, str):
            self._initialize_weights(initialize)
        elif initialize is True: # Allow boolean True to default to kaiming
             self._initialize_weights('kaiming_uniform')
        # Else: use default PyTorch initialization if initialize is False or None

    def _initialize_weights(self, scheme: str):
        """ Applies weight initialization to the linear layer. """
        scheme = scheme.lower()
        if scheme == 'kaiming_uniform':
            # Kaiming uniform for ReLU/variants (uses sqrt(5) like default Linear init)
            nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
            if self.use_bias and self.linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.linear.bias, -bound, bound)
        elif scheme == 'xavier_uniform':
            # Xavier uniform for tanh/sigmoid (gain=1)
            nn.init.xavier_uniform_(self.linear.weight)
            if self.use_bias and self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError(f"Unsupported initialization scheme: {scheme}. Choose 'xavier_uniform', 'kaiming_uniform', or False.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the projection, optional normalization, and optional activation.

        Args:
            x (torch.Tensor): Input tensor of shape (*, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (*, out_dim).
        """
        # Ensure type matches layer parameters for robustness
        expected_dtype = self.linear.weight.dtype
        if x.dtype != expected_dtype:
            x = x.to(dtype=expected_dtype)

        # 1. Linear Projection
        x = self.linear(x)

        # 2. Optional Layer Normalization
        # Note: Order matters - norm is usually applied before activation
        if self.norm is not None:
            x = self.norm(x)

        # 3. Optional Activation
        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        """ Provides a detailed string representation of the module configuration. """
        s = f'in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.use_bias}'
        if self.norm is not None:
            s += f', norm=True (eps={self.norm.eps}, affine={self.norm.elementwise_affine})'
        if self.activation is not None:
            s += f', activation={self.activation.__class__.__name__}'
        # Consider adding initialization scheme info if needed for debugging
        # s += f', init={getattr(self, "_init_scheme", "default")}' # Requires storing scheme if needed
        return s

# Example Usage (can be removed or kept for illustration/testing)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    in_dimension = 128
    out_dimension = 256
    batch = 4
    seq = 10

    dummy_input = torch.randn(batch, seq, in_dimension, device=device, dtype=dtype)

    print("--- Basic Linear Projection (Kaiming Init) ---")
    proj_basic = LinearProjection(in_dimension, out_dimension, device=device, dtype=dtype)
    print(proj_basic)
    out_basic = proj_basic(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", out_basic.shape)
    assert out_basic.shape == (batch, seq, out_dimension)

    print("\n--- Projection with GELU Activation (Xavier Init) ---")
    proj_act = LinearProjection(in_dimension, out_dimension, activation_cls=nn.GELU, initialize='xavier_uniform', device=device, dtype=dtype)
    print(proj_act)
    out_act = proj_act(dummy_input)
    print("Output shape:", out_act.shape)
    assert out_act.shape == (batch, seq, out_dimension)

    print("\n--- Projection with LayerNorm (No Activation, Default Init) ---")
    proj_norm = LinearProjection(in_dimension, out_dimension, use_norm=True, norm_eps=1e-6, initialize=False, device=device, dtype=dtype)
    print(proj_norm)
    out_norm = proj_norm(dummy_input)
    print("Output shape:", out_norm.shape)
    assert out_norm.shape == (batch, seq, out_dimension)
    print("Output Mean (approx 0):", out_norm.mean(dim=-1).mean().item())
    print("Output Std (approx 1):", out_norm.std(dim=-1).mean().item())

    print("\n--- Projection with LayerNorm and GELU Activation ---")
    proj_norm_act = LinearProjection(
        in_dimension, out_dimension,
        activation_cls=nn.GELU,
        use_norm=True,
        initialize='kaiming_uniform',
        device=device, dtype=dtype
    )
    print(proj_norm_act)
    out_norm_act = proj_norm_act(dummy_input)
    print("Output shape:", out_norm_act.shape)
    assert out_norm_act.shape == (batch, seq, out_dimension)

    # Test JIT scripting (likely works for this simple module)
    try:
        scripted_proj = torch.jit.script(proj_norm_act)
        print("\nJIT scripting successful.")
        jit_out = scripted_proj(dummy_input)
        assert torch.allclose(jit_out, out_norm_act)
        print("JIT output matches eager output.")
    except Exception as e:
        print("\nJIT scripting failed:", e)


