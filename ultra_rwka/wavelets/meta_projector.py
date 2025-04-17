import torch
import torch.nn as nn
import math
import warnings
from typing import Optional, Type, Union

# Imports from within the library
from ..projections import LinearProjection
# Correct relative path for backend interface
from ...backend.interface import wavelet_projection
# Import utility from the revised bases.py
from .bases import normalize_filters

# MetaNetW class remains unchanged
class MetaNetW(nn.Module):
    """
    Meta-Network (e.g., MLP) that generates wavelet parameters (theta_W)
    based on context. In the 'direct_fir' case, theta_W represents the
    filter coefficients directly.
    """
    def __init__(self,
                 context_dim: int,
                 parameter_dim: int, # d_thetaW = num_filters * filter_length
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2, # Typically needs some capacity
                 activation_cls: Type[nn.Module] = nn.GELU,
                 use_bias: bool = True,
                 initialize_weights: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.context_dim = context_dim
        self.parameter_dim = parameter_dim

        if num_layers < 1:
            raise ValueError("MetaNetW num_layers must be at least 1.")

        # Heuristic for hidden dim if not provided
        _hidden_dim = hidden_dim if hidden_dim is not None else max(context_dim, parameter_dim)

        layers = []
        current_dim = context_dim
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            output_dim = parameter_dim if is_last_layer else _hidden_dim
            linear_layer = nn.Linear(current_dim, output_dim, bias=use_bias, **self.factory_kwargs)

            if initialize_weights:
                # Use Xavier for potentially non-linear mapping
                nn.init.xavier_uniform_(linear_layer.weight)
                if use_bias and linear_layer.bias is not None:
                    nn.init.zeros_(linear_layer.bias)

            layers.append(linear_layer)
            if not is_last_layer:
                layers.append(activation_cls())
                current_dim = _hidden_dim
            else:
                # No activation after the final layer (outputs parameters/coefficients)
                pass

        self.network = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """ Computes wavelet parameters theta_W. """
        # Ensure input type matches network parameters
        expected_dtype = next(self.parameters()).dtype
        if context.dtype != expected_dtype:
            context = context.to(dtype=expected_dtype)
        return self.network(context)

class MetaWaveletProjector(nn.Module):
    """
    Performs adaptive multi-resolution analysis using meta-learned wavelet projections.

    Generates time-varying wavelet parameters (theta_W) using MetaNetW based on context,
    derives filters from these parameters (including normalization), and then projects
    the input signal using these dynamic filters via a backend call.

    Currently implements 'direct_fir' parameterization where MetaNetW directly
    outputs the filter coefficients, with filters organized as lo_d/hi_d pairs.
    """
    def __init__(self,
                 input_dim: int, # Dimension of input signal x_t
                 output_dim: int, # Dimension of output wavelet coefficient vector W_t
                 context_dim: int, # Dimension of context input for MetaNetW
                 parameterization_type: str = 'direct_fir',
                 num_filters: int = 32, # Number of dynamic filters (must be even for lo_d/hi_d pairs)
                 filter_length: int = 16, # Length of each FIR filter
                 meta_net_hidden_dim: Optional[int] = None,
                 meta_net_layers: int = 2,
                 proj_dropout_p: float = 0.0, # Dropout on input x before projection
                 device=None,
                 dtype=None):
        """
        Args:
            input_dim (int): Dimension of the input signal features.
            output_dim (int): Desired dimension of the output wavelet coefficient vector.
                               This must match the number of coefficients produced by the backend.
            context_dim (int): Input dimension for the MetaNetW.
            parameterization_type (str): Method for interpreting MetaNetW output.
                                         Currently only 'direct_fir' is implemented.
                                         Defaults to 'direct_fir'.
            num_filters (int): Number of FIR filters to generate if using 'direct_fir'.
                               Must be even to form lo_d/hi_d pairs.
            filter_length (int): Length of each FIR filter if using 'direct_fir'.
            meta_net_hidden_dim (Optional[int]): Hidden dimension for MetaNetW MLP.
            meta_net_layers (int): Number of layers for MetaNetW MLP. Defaults to 2.
            proj_dropout_p (float): Dropout probability applied to the input `x`. Defaults to 0.0.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.parameterization_type = parameterization_type.lower()
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # Enforce even num_filters for lo_d/hi_d pairing
        if self.num_filters % 2 != 0:
            raise ValueError(f"num_filters ({self.num_filters}) must be even to form lo_d/hi_d pairs.")

        if self.parameterization_type != 'direct_fir':
            # TODO: Implement other parameterizations if needed (e.g., 'analytical_morlet_params')
            raise NotImplementedError(f"parameterization_type '{self.parameterization_type}' not implemented.")

        # Calculate the total dimension of parameters needed from MetaNetW
        self.parameter_dim = num_filters * filter_length # d_thetaW

        # --- Layers ---
        self.dropout = nn.Dropout(p=proj_dropout_p) if proj_dropout_p > 0 else nn.Identity()

        # Meta-Network to generate wavelet parameters (filter coefficients)
        self.meta_net_w = MetaNetW(
            context_dim=context_dim,
            parameter_dim=self.parameter_dim,
            hidden_dim=meta_net_hidden_dim,
            num_layers=meta_net_layers,
            **self.factory_kwargs
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Generate dynamic filters, normalize them as lo_d/hi_d pairs, and project the input signal.

        Args:
            x (torch.Tensor): Input signal tensor of shape (Batch, SeqLen, input_dim).
            context (torch.Tensor): Context tensor for MetaNetW. Shape should be compatible,
                                     e.g., (Batch, SeqLen, context_dim) or (Batch, context_dim).

        Returns:
            torch.Tensor: Output wavelet coefficient tensor W_t of shape (Batch, SeqLen, output_dim).
        """
        if not x.is_contiguous(): x = x.contiguous()
        if not context.is_contiguous(): context = context.contiguous()

        B, T, D_in = x.shape
        Ctx_dim = context.shape[-1]

        if Ctx_dim != self.context_dim:
             raise ValueError(f"Context last dim ({Ctx_dim}) doesn't match expected context_dim ({self.context_dim})")

        if context.ndim == 2 and context.shape[0] == B:
             context = context.unsqueeze(1).expand(-1, T, -1)
        elif context.shape[:-1] != x.shape[:-1]:
             raise ValueError(f"Context shape {context.shape} is not compatible with input shape {x.shape}")

        x = self.dropout(x)

        # 1. Generate Wavelet Parameters (theta_W)
        theta_W = self.meta_net_w(context) # Shape: (B, T, parameter_dim)

        # 2. Derive and Normalize Filters
        if self.parameterization_type == 'direct_fir':
            try:
                # Reshape raw coefficients: (B, T, NumF*FiltL) -> (B*T, NumF, FiltL)
                raw_filters = theta_W.view(B * T, self.num_filters, self.filter_length)
                # Split into lo_d and hi_d filters (first half lo_d, second half hi_d)
                num_pairs = self.num_filters // 2
                lo_d_filters = raw_filters[:, :num_pairs, :] # Shape: (B*T, num_pairs, filter_length)
                hi_d_filters = raw_filters[:, num_pairs:, :] # Shape: (B*T, num_pairs, filter_length)
            except RuntimeError as e:
                 print(f"Error reshaping theta_W {theta_W.shape} to filters "
                       f"({B*T}, {self.num_filters}, {self.filter_length}). Parameter dim mismatch?")
                 raise e

            # Normalize filters, assuming normalize_filters handles lo_d/hi_d pairing
            # Note: normalize_filters must ensure lo_d/hi_d satisfy DWT conditions (e.g., quadrature mirror)
            lo_d_normalized = normalize_filters(lo_d_filters)
            hi_d_normalized = normalize_filters(hi_d_filters)
            # Concatenate normalized filters back in order: [lo_d, hi_d]
            filters = torch.cat([lo_d_normalized, hi_d_normalized], dim=1) # Shape: (B*T, num_filters, filter_length)

        else:
             raise NotImplementedError(f"Filter derivation for '{self.parameterization_type}' not implemented.")

        # 3. Prepare Input for Backend Projection
        x_reshaped = x.reshape(B * T, D_in) # Shape: (B*T, D_in)

        # 4. Call Backend Wavelet Projection
        try:
            x_reshaped = x_reshaped.contiguous()
            filters = filters.contiguous() # Use normalized filters

            # Backend takes normalized filters
            wavelet_coeffs = wavelet_projection(x_reshaped, filters) # Expected output: (B*T, output_dim)

        except ImportError:
             raise ImportError("Wavelet backend not found. Ensure 'ultra_rwka_backend' is compiled and installed.")
        except RuntimeError as e:
             print(f"RuntimeError during wavelet_projection backend call: {e}")
             print("Check backend kernel implementation and input shapes/types.")
             raise e
        except NotImplementedError as e:
             print(f"Backend function wavelet_projection reported as not implemented: {e}")
             raise e

        # 5. Validate and Reshape Output
        if wavelet_coeffs.shape != (B * T, self.output_dim):
             warnings.warn(f"Backend output shape {wavelet_coeffs.shape} does not match expected "
                           f"({B*T}, {self.output_dim}). Check backend implementation and output_dim config.")
             try:
                 output = wavelet_coeffs.view(B, T, -1)
                 if output.shape[-1] != self.output_dim:
                      raise ValueError("Backend output dimension mismatch after reshaping.")
             except RuntimeError as e:
                  print("Failed to reshape backend output.")
                  raise e
        else:
             output = wavelet_coeffs.view(B, T, self.output_dim)

        return output

    def extra_repr(self) -> str:
        s = f"input_dim={self.input_dim}, output_dim={self.output_dim}, context_dim={self.context_dim}\n"
        s += f"  parameterization={self.parameterization_type}, "
        if self.parameterization_type == 'direct_fir':
             s += (f"num_filter_pairs={self.num_filters // 2}, num_filters={self.num_filters}, "
                   f"filter_length={self.filter_length}, parameter_dim={self.parameter_dim}\n")
        s += f"  (meta_net_w): {self.meta_net_w}"
        if isinstance(self.dropout, nn.Dropout) and self.dropout.p > 0:
             s += f"\n  (dropout): Dropout(p={self.dropout.p})"
        return s.strip()

# Example Usage (unchanged, num_filters=8 is already even)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Config
    B, T, D_in = 4, 50, 32  # Batch, Time, Input Dim
    Ctx_dim = 64           # Context Dim
    Out_dim = 128          # Output coefficient Dim
    Num_Filt = 8           # Num filters for 'direct_fir' (even, forms 4 lo_d/hi_d pairs)
    Filt_Len = 16          # Filter length for 'direct_fir' -> Param Dim = 8*16=128

    # Create the projector
    meta_proj = MetaWaveletProjector(
        input_dim=D_in,
        output_dim=Out_dim,
        context_dim=Ctx_dim,
        parameterization_type='direct_fir',
        num_filters=Num_Filt,
        filter_length=Filt_Len,
        meta_net_layers=2,
        proj_dropout_p=0.1,
        device=device,
        dtype=dtype
    )
    print("--- MetaWaveletProjector (Updated) ---")
    print(meta_proj)

    # Create dummy inputs
    x_in = torch.randn(B, T, D_in, device=device, dtype=dtype)
    # Time-varying context
    context_in_tv = torch.randn(B, T, Ctx_dim, device=device, dtype=dtype)
    # Time-invariant context (per batch item)
    context_in_ti = torch.randn(B, Ctx_dim, device=device, dtype=dtype)

    # --- Forward pass ---
    print("\nRunning forward pass (Time-Varying Context)...")
    try:
        output_tv = meta_proj(x_in, context_in_tv)
        print("Input x shape:", x_in.shape)
        print("Context shape:", context_in_tv.shape)
        print("Output shape:", output_tv.shape) # Should be (B, T, Out_dim)
        assert output_tv.shape == (B, T, Out_dim)
        print("Output sample:", output_tv.view(-1)[:8].detach().cpu().numpy())
    except (ImportError, RuntimeError, NotImplementedError, ValueError) as e:
        print(f"\nCaught expected error (backend/config issue): {e}")

    print("\nRunning forward pass (Time-Invariant Context)...")
    try:
        output_ti = meta_proj(x_in, context_in_ti)
        print("Input x shape:", x_in.shape)
        print("Context shape:", context_in_ti.shape)
        print("Output shape:", output_ti.shape) # Should be (B, T, Out_dim)
        assert output_ti.shape == (B, T, Out_dim)
    except (ImportError, RuntimeError, NotImplementedError, ValueError) as e:
        print(f"\nCaught expected error (backend/config issue): {e}")
