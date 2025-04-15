# ultra_rwka/components/attention/tm_kla.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Union, Type, List, Sequence

# Imports from within the library
# Use relative imports assuming standard package structure
from ..kernels.feature_maps import FeatureMap
from ..kernels.mixer import KernelFeatureMixer
from ..projections import LinearProjection
# Correct relative path for backend interface
from ...backend.interface import fa_kla_attention

class TemperatureFunction(nn.Module):
    """
    Learnable function f_tau to compute relevance temperatures T_i >= 0.
    Typically an MLP followed by a non-negative activation function.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int, # Usually num_heads
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 1,
                 activation_cls: Type[nn.Module] = nn.GELU,
                 final_activation: str = 'softplus', # Ensures non-negative output
                 use_bias: bool = True,
                 dropout_p: float = 0.0,
                 initialize_weights: bool = True,
                 device=None,
                 dtype=None):
        """
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension (should match num_heads).
            hidden_dim (Optional[int]): Hidden dimension for MLP if num_layers > 1.
            num_layers (int): Number of layers in the MLP (>= 1).
            activation_cls (Type[nn.Module]): Activation class for hidden layers.
            final_activation (str): Name of the final non-negative activation ('softplus', 'relu', 'sigmoid', 'abs').
            use_bias (bool): Whether linear layers use bias.
            dropout_p (float): Dropout probability for hidden layers.
            initialize_weights (bool): Whether to apply Xavier initialization.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_dim = input_dim
        self.output_dim = output_dim # Should match num_heads
        self.final_activation_type = final_activation.lower()

        if num_layers < 1:
            raise ValueError("TemperatureFunction num_layers must be at least 1.")
        supported_final_activations = ['softplus', 'relu', 'sigmoid', 'abs']
        if self.final_activation_type not in supported_final_activations:
             raise ValueError(f"final_activation must be one of {supported_final_activations}")

        _hidden_dim = hidden_dim if hidden_dim is not None else max(input_dim // 2, output_dim)

        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            layer_output_dim = output_dim if is_last_layer else _hidden_dim
            linear_layer = nn.Linear(current_dim, layer_output_dim, bias=use_bias, **self.factory_kwargs)

            if initialize_weights:
                nn.init.xavier_uniform_(linear_layer.weight)
                if use_bias and linear_layer.bias is not None:
                    nn.init.zeros_(linear_layer.bias)

            layers.append(linear_layer)

            if not is_last_layer:
                layers.append(activation_cls())
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(p=dropout_p))
                current_dim = _hidden_dim
            # No activation/dropout after the final layer (logits)

        self.network = nn.Sequential(*layers)

        # Select final activation function mapping
        if self.final_activation_type == 'softplus':
            self.final_activation_fn = F.softplus
        elif self.final_activation_type == 'relu':
            self.final_activation_fn = F.relu
        elif self.final_activation_type == 'sigmoid': # Outputs [0, 1]
            self.final_activation_fn = torch.sigmoid
        elif self.final_activation_type == 'abs':
             self.final_activation_fn = torch.abs
        else:
            # This case should be prevented by the check in __init__
            raise NotImplementedError(f"Final activation {self.final_activation_type} not implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes non-negative temperatures. """
        # Ensure input type matches network parameters
        expected_dtype = next(self.parameters()).dtype
        if x.dtype != expected_dtype:
            x = x.to(dtype=expected_dtype)

        logits = self.network(x)
        temperatures = self.final_activation_fn(logits)
        return temperatures


class TMLinKernelAttn(nn.Module):
    """
    Temperature-Modulated Linear Kernel Attention (TM-KLA).

    Implements the core attention mechanism described in Sec 3.8[cite: 79, 80] of the paper,
    using kernel feature maps (§3.2)[cite: 59] (potentially mixed via KernelFeatureMixer)
    and learnable temperature modulation (§3.8)[cite: 30]. Relies on an efficient backend
    implementation (FA-KLA via `interface.py`) for the final computation (§3.7)[cite: 78].

    High-level Formula:
        Q = Proj_Q(x); K = Proj_K(x); V = Proj_V(x)
        # phi_q = FeatureMap(Q); phi_k = FeatureMap(K) # FeatureMap includes optional mixing
        # Note: Assumes FeatureMap takes projected Q/K as input
        phi_q = self.feature_mapper(Q); phi_k = self.feature_mapper(K)
        T = TempFunc(temp_input) # Calculate temperature based on precomputed input
        output = FA_KLA_Backend(phi_q, phi_k, V, T) # Shapes need alignment
    """
    def __init__(self,
                 embed_dim: int,
                 feature_mapper: Union[FeatureMap, KernelFeatureMixer],
                 num_heads: int = 8,
                 qk_dim: Optional[int] = None, # Dimension for Q/K projections per head
                 v_dim: Optional[int] = None,  # Dimension for V projection per head
                 temp_input_dim: int = 0, # Dimension for TemperatureFunction input
                 temp_hidden_dim: Optional[int] = None,
                 temp_layers: int = 1,
                 temp_final_activation: str = 'softplus',
                 proj_bias: bool = True,
                 proj_init: str = 'kaiming_uniform',
                 dropout_p: float = 0.0, # Dropout on input x before projections
                 device=None,
                 dtype=None):
        """
        Args:
            embed_dim (int): Input embedding dimension.
            feature_mapper (Union[FeatureMap, KernelFeatureMixer]): Instantiated feature map module
                (e.g., RFF, LearnablePositiveMap) or a KernelFeatureMixer instance.
                Its output dimension determines `feature_dim`. Its input dimension must match
                the output dimension of Q/K projections (total_qk_dim).
            num_heads (int): Number of attention heads. Defaults to 8.
            qk_dim (Optional[int]): Dimension per head for Query and Key projections.
                                    Defaults to embed_dim // num_heads.
            v_dim (Optional[int]): Dimension per head for Value projection.
                                   Defaults to embed_dim // num_heads.
            temp_input_dim (int): Input dimension required by the TemperatureFunction.
                                  Must be pre-calculated and passed in `forward`. Defaults to 0 (no temp).
            temp_hidden_dim (Optional[int]): Hidden dimension for TemperatureFunction MLP.
            temp_layers (int): Number of layers for TemperatureFunction MLP. Defaults to 1.
            temp_final_activation (str): Final non-negative activation for TemperatureFunction. Defaults to 'softplus'.
            proj_bias (bool): Whether Q/K/V projection layers use bias. Defaults to True.
            proj_init (str): Initialization scheme for Q/K/V projections. Defaults to 'kaiming_uniform'.
            dropout_p (float): Dropout probability applied to the input `x` before projections. Defaults to 0.0.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Get feature_dim (output dim of mapper) and verify input dim of mapper
        self.feature_dim = feature_mapper.feature_dim
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # Head dimensions
        self.qk_head_dim = qk_dim if qk_dim is not None else embed_dim // num_heads
        self.v_head_dim = v_dim if v_dim is not None else embed_dim // num_heads
        if embed_dim % num_heads != 0 and (qk_dim is None or v_dim is None):
             warnings.warn(f"embed_dim ({embed_dim}) is not divisible by num_heads ({num_heads}). "
                           "Head dimensions might not be as expected if qk_dim/v_dim are not specified.")

        self.total_qk_dim = self.num_heads * self.qk_head_dim
        self.total_v_dim = self.num_heads * self.v_head_dim

        # --- Input Dimension Check for Feature Mapper ---
        # Check if the feature mapper's input dimension matches the total Q/K projection dimension
        expected_mapper_in_dim = self.total_qk_dim
        if isinstance(feature_mapper, KernelFeatureMixer):
            # Mixer's input dim is based on its *first* feature map's input dim
            actual_mapper_in_dim = feature_mapper.feature_maps[0].in_dim
            # Also check the GateNet input dim if context is used (checked in forward)
        elif isinstance(feature_mapper, FeatureMap):
            actual_mapper_in_dim = feature_mapper.in_dim
        else:
             raise TypeError("feature_mapper must be an instance of FeatureMap or KernelFeatureMixer")

        if actual_mapper_in_dim != expected_mapper_in_dim:
            raise ValueError(f"feature_mapper input dimension ({actual_mapper_in_dim}) does not match "
                             f"required Q/K projection dimension ({expected_mapper_in_dim})")

        # --- Layers ---
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # Q, K, V projections
        self.proj_q = LinearProjection(embed_dim, self.total_qk_dim, use_bias=proj_bias, initialize=proj_init, **self.factory_kwargs)
        self.proj_k = LinearProjection(embed_dim, self.total_qk_dim, use_bias=proj_bias, initialize=proj_init, **self.factory_kwargs)
        self.proj_v = LinearProjection(embed_dim, self.total_v_dim, use_bias=proj_bias, initialize=proj_init, **self.factory_kwargs)

        # Feature mapper (passed in and registered)
        self.feature_mapper = feature_mapper

        # Temperature function (optional, only if temp_input_dim > 0)
        self.temp_func: Optional[TemperatureFunction] = None
        if temp_input_dim > 0:
            self.temp_func = TemperatureFunction(
                input_dim=temp_input_dim,
                output_dim=self.num_heads, # One temperature per head
                hidden_dim=temp_hidden_dim,
                num_layers=temp_layers,
                final_activation=temp_final_activation,
                device=device,
                dtype=dtype
            )
        else:
             # No warning needed if explicitly set to 0
             pass
             # warnings.warn("temp_input_dim is 0, temperature modulation will be disabled (T=1).")

        # Output projection is typically handled outside this module in the main model block


    def forward(self,
                x: torch.Tensor,
                temp_input: Optional[torch.Tensor] = None,
                mixer_context: Optional[torch.Tensor] = None # Context for KernelFeatureMixer's GateNet
               ) -> torch.Tensor:
        """
        Apply Temperature-Modulated Linear Kernel Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, SeqLen, embed_dim).
            temp_input (Optional[torch.Tensor]): Input for the TemperatureFunction.
                Shape (Batch, SeqLen, temp_input_dim). Required if temp_input_dim > 0.
                Must be pre-calculated based on repr(x_i), state, context etc.
            mixer_context (Optional[torch.Tensor]): Context for the KernelFeatureMixer's GateNet,
                                                    only used if feature_mapper is a KernelFeatureMixer.
                                                    Shape should be compatible with mixer's gate_input_dim logic.

        Returns:
            torch.Tensor: Output context vector of shape (Batch, SeqLen, total_v_dim).
        """
        if not x.is_contiguous(): x = x.contiguous()
        B, T, E = x.shape
        H = self.num_heads
        # Dqk = self.qk_head_dim # QK head dim
        Dv = self.v_head_dim   # V head dim
        # Df = self.feature_dim # Total feature map output dim

        # Apply dropout to input
        x = self.dropout(x)

        # 1. Project Q, K, V
        q_proj = self.proj_q(x) # (B, T, H*Dqk)
        k_proj = self.proj_k(x) # (B, T, H*Dqk)
        v      = self.proj_v(x) # (B, T, H*Dv)

        # 2. Apply Feature Map(s) to Q and K
        # The feature_mapper takes the *projected* Q/K as input.
        # Its input dim must match total_qk_dim.
        # Its output dim is self.feature_dim.
        if isinstance(self.feature_mapper, KernelFeatureMixer):
             # Pass mixer_context only to the mixer
             q_mapped_flat = self.feature_mapper(q_proj, mixer_context) # (B, T, Df)
             k_mapped_flat = self.feature_mapper(k_proj, mixer_context) # (B, T, Df)
        elif isinstance(self.feature_mapper, FeatureMap):
             q_mapped_flat = self.feature_mapper(q_proj) # (B, T, Df)
             k_mapped_flat = self.feature_mapper(k_proj) # (B, T, Df)
        else:
             # This path should ideally not be reached due to __init__ checks
             raise TypeError("self.feature_mapper is not a valid FeatureMap or KernelFeatureMixer")

        # Reshape Q_mapped, K_mapped, V for multi-head attention calculation
        # Check if feature_dim is divisible by num_heads for per-head features
        if self.feature_dim % H != 0:
            raise ValueError(f"Feature dim ({self.feature_dim}) must be divisible by num_heads ({H}) for head reshaping.")
        Df_head = self.feature_dim // H

        q_mapped = q_mapped_flat.view(B, T, H, Df_head) # (B, T, H, Df_head)
        k_mapped = k_mapped_flat.view(B, T, H, Df_head) # (B, T, H, Df_head)
        v_reshaped = v.view(B, T, H, Dv)                # (B, T, H, Dv)

        # 3. Compute Temperature T
        if self.temp_func is not None:
            if temp_input is None:
                raise ValueError("temp_input is required for TemperatureFunction but was not provided.")
            # Validate shape dynamically
            expected_temp_shape = (B, T, self.temp_func.input_dim)
            if temp_input.shape != expected_temp_shape:
                 raise ValueError(f"Expected temp_input shape {expected_temp_shape}, got {temp_input.shape}")

            # temp shape: (B, T, H) - one temperature per head per token
            temp = self.temp_func(temp_input) # Already ensures non-negative output
            # Ensure temp dtype matches q_mapped for backend compatibility
            if temp.dtype != q_mapped.dtype:
                temp = temp.to(dtype=q_mapped.dtype)
        else:
            # If no temp func, use default temperature of 1.0
            # Shape (B, T, H)
            temp = torch.ones(B, T, H, device=x.device, dtype=q_mapped.dtype) # Match mapped Q/K dtype

        # 4. Call FA-KLA Backend
        # Expected shapes: q_mapped=(B,T,H,Df_head), k_mapped=(B,T,H,Df_head), v=(B,T,H,Dv), temp=(B,T,H)
        # Backend function needs to handle these shapes.
        try:
            # Ensure contiguity for backend call if necessary (often handled by autograd function wrapper)
            q_mapped = q_mapped.contiguous()
            k_mapped = k_mapped.contiguous()
            v_reshaped = v_reshaped.contiguous()
            temp = temp.contiguous()

            # Call the function imported from the backend interface
            output = fa_kla_attention(q_mapped, k_mapped, v_reshaped, temp) # Output: (B, T, H, Dv)
        except ImportError:
             # This error means the setup.py build likely failed or wasn't run
             raise ImportError("FA-KLA backend not found. Ensure 'ultra_rwka_backend' is compiled and installed.")
        except RuntimeError as e:
             # This error likely means the backend C++/CUDA code itself has an issue
             print(f"RuntimeError during fa_kla_attention backend call: {e}")
             print("Check backend kernel implementation and input shapes/types.")
             raise e

        # 5. Reshape output
        # Concatenate heads: (B, T, H, Dv) -> (B, T, H*Dv)
        # Use reshape for efficiency, view might require contiguous call first
        output = output.reshape(B, T, self.total_v_dim)

        # 6. Final projection (proj_out) is typically applied *after* this module
        # in the main model architecture (e.g., within a Transformer block).

        return output

    def extra_repr(self) -> str:
        """ Provides a detailed string representation of the module configuration. """
        s = f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, '
        s += f'qk_head_dim={self.qk_head_dim}, v_head_dim={self.v_head_dim}, '
        s += f'feature_dim={self.feature_dim}\n'
        # Use the feature_mapper's own repr for details
        s += f'  (feature_mapper): {self.feature_mapper}\n'
        if self.temp_func:
             s += f'  (temp_func): {self.temp_func}\n'
        else:
             s += '  (temp_func): None (T=1)\n'
        # Use LinearProjection's repr
        s += f'  (proj_q): {self.proj_q}\n'
        s += f'  (proj_k): {self.proj_k}\n'
        s += f'  (proj_v): {self.proj_v}'
        # Add dropout info if active
        if isinstance(self.dropout, nn.Dropout) and self.dropout.p > 0:
             s += f'\n  (dropout): Dropout(p={self.dropout.p})'
        return s


# Example Usage (Remains largely the same, ensure configuration matches new checks)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Config
    B, T, E = 4, 10, 128 # Batch, Time, Embed Dim
    H = 8              # Heads
    Dqk, Dv = 16, 16     # Head dims for QK and V
    TotalQkDim = H * Dqk # 128
    Df = 64             # Feature map output dim (must be divisible by H)
    Temp_In_Dim = E + 32 # Example: Temp func sees concat(x, some_state)

    # --- Create Feature Mapper (using Mixer) ---
    # Feature maps must take TotalQkDim as input and output Df
    fm1 = RandomFourierFeatures(in_dim=TotalQkDim, feature_dim=Df, device=device, dtype=dtype)
    fm2 = LearnablePositiveMap(in_dim=TotalQkDim, feature_dim=Df, device=device, dtype=dtype)

    # Mixer gate input dim depends on what drives the mixing weights.
    # Let's assume it takes the projected Q (TotalQkDim) and some context (e.g., 16 dims)
    mixer_gate_in_dim = TotalQkDim + 16
    feature_mixer_proj = KernelFeatureMixer(
         feature_maps=[fm1, fm2],
         gate_input_dim=mixer_gate_in_dim, # Gate sees projected Q + context
         device=device, dtype=dtype
    )

    # --- Create TM-KLA module ---
    tm_kla_attn = TMLinKernelAttn(
        embed_dim=E,
        feature_mapper=feature_mixer_proj, # Pass the mixer instance
        num_heads=H,
        qk_dim=Dqk,
        v_dim=Dv,
        temp_input_dim=Temp_In_Dim, # Specify temp input dim
        temp_hidden_dim=32,
        temp_layers=2,
        dropout_p=0.1,
        device=device,
        dtype=dtype
    )
    print("--- TM-KLA Module ---")
    print(tm_kla_attn)

    # --- Create dummy inputs ---
    x_in = torch.randn(B, T, E, device=device, dtype=dtype)
    temp_in = torch.randn(B, T, Temp_In_Dim, device=device, dtype=dtype)
    # Dummy context for the mixer's gate
    mixer_ctx = torch.randn(B, T, 16, device=device, dtype=dtype) # Example context

    # --- Forward pass ---
    print("\nRunning forward pass...")
    try:
        # Pass mixer context if feature_mapper is KernelFeatureMixer
        output = tm_kla_attn(x_in, temp_input=temp_in, mixer_context=mixer_ctx)

        print("Input x shape:", x_in.shape)
        print("Temp input shape:", temp_in.shape)
        print("Mixer context shape:", mixer_ctx.shape)
        print("Output shape:", output.shape) # Should be (B, T, H*Dv) = (4, 10, 128)
        assert output.shape == (B, T, H * Dv)
        print("Output sample:", output.view(-1)[:8].detach().cpu().numpy()) # Use detach().cpu() for numpy
    except ImportError as e:
        print(f"\nCaught expected error (backend likely not compiled): {e}")
    except RuntimeError as e:
         print(f"\nCaught expected error (backend kernel likely not implemented or shape mismatch): {e}")
    except ValueError as e:
         print(f"\nCaught configuration error: {e}") # Catch config errors too

    # --- Test without temperature ---
    print("\n--- Testing without Temperature ---")
    tm_kla_no_temp = TMLinKernelAttn(
        embed_dim=E,
        feature_mapper=feature_mixer_proj, # Use same mixer
        num_heads=H,
        qk_dim=Dqk,
        v_dim=Dv,
        temp_input_dim=0, # Disable temperature
        device=device,
        dtype=dtype
    )
    print(tm_kla_no_temp)
    try:
        # Mixer still needs context if defined that way
        output_no_temp = tm_kla_no_temp(x_in, temp_input=None, mixer_context=mixer_ctx)
        print("Output shape (no temp):", output_no_temp.shape)
        assert output_no_temp.shape == (B, T, H * Dv)
    except ImportError as e:
        print(f"\nCaught expected error (backend likely not compiled): {e}")
    except RuntimeError as e:
         print(f"\nCaught expected error (backend kernel likely not implemented or shape mismatch): {e}")
    except ValueError as e:
         print(f"\nCaught configuration error: {e}")


