# ultra_rwka/components/moe/branches.py

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Type, Any

# Imports from within the library
from ..projections import LinearProjection
# Import other components if needed for more complex branches
# from ..attention.tm_kla import TMLinKernelAttn
# from ..wavelets.meta_projector import MetaWaveletProjector
# from ..memory.hierarchical import HierarchicalMemoryStack # For state processing?

class MLPBranch(nn.Module):
    """
    A simple Specialist Branch implemented as a Multi-Layer Perceptron (MLP).

    It expects specific keys in the input dictionary, concatenates the corresponding
    tensors along the feature dimension, and processes them through an MLP.
    """
    def __init__(self,
                 expected_inputs: Dict[str, int], # Dict mapping input key -> input dimension
                 output_dim: int, # d_h_branch
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2, # Default to at least 2 layers for some capacity
                 activation_cls: Type[nn.Module] = nn.GELU,
                 use_bias: bool = True,
                 initialize_weights: bool = True,
                 device=None,
                 dtype=None):
        """
        Args:
            expected_inputs (Dict[str, int]): Dictionary specifying the keys this branch
                expects in the input dictionary passed to forward(), and their
                corresponding feature dimensions. E.g., {'hidden_state': 128, 'kernel_features': 64}.
            output_dim (int): The output dimension of this branch.
            hidden_dim (Optional[int]): Hidden dimension for the MLP layers. Defaults to output_dim.
            num_layers (int): Number of layers in the MLP (>= 1). Defaults to 2.
            activation_cls (Type[nn.Module]): Activation class for hidden layers. Defaults to nn.GELU.
            use_bias (bool): Whether linear layers use bias. Defaults to True.
            initialize_weights (bool): Whether to apply standard initialization. Defaults to True.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        if not expected_inputs:
            raise ValueError("expected_inputs dictionary cannot be empty.")

        self.expected_inputs = expected_inputs
        self.output_dim = output_dim
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # Calculate the total input dimension after concatenation
        self.total_input_dim = sum(expected_inputs.values())

        _hidden_dim = hidden_dim if hidden_dim is not None else output_dim

        # --- Build the MLP Network ---
        layers = []
        current_dim = self.total_input_dim
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            layer_output_dim = output_dim if is_last_layer else _hidden_dim
            linear_layer = nn.Linear(current_dim, layer_output_dim, bias=use_bias, **self.factory_kwargs)

            if initialize_weights:
                # Standard init like Kaiming or Xavier often works well here
                nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
                if use_bias and linear_layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(linear_layer.bias, -bound, bound)

            layers.append(linear_layer)
            if not is_last_layer:
                layers.append(activation_cls())
                current_dim = _hidden_dim
            # No activation after final layer

        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process the selected inputs through the MLP.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing various potential
                inputs from the main model (e.g., 'hidden_state', 'kernel_features',
                'wavelet_coeffs', 'raw_input'). Tensors are expected to have shape
                (Batch, SeqLen, FeatureDim).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, SeqLen, output_dim).
        """
        selected_tensors = []
        expected_shape_prefix = None

        # 1. Select and validate expected inputs
        for key, expected_dim in self.expected_inputs.items():
            if key not in inputs:
                raise ValueError(f"MLPBranch expected input key '{key}' but it was not found in the input dictionary.")

            tensor = inputs[key]
            # Basic shape check (at least 3 dims: B, T, D)
            if tensor.ndim < 3:
                 raise ValueError(f"Input tensor '{key}' must have at least 3 dimensions (B, T, D), got {tensor.shape}")
            if tensor.shape[-1] != expected_dim:
                raise ValueError(f"Input tensor '{key}' has dimension {tensor.shape[-1]}, "
                                 f"but expected {expected_dim}.")

            # Check for consistent Batch and SeqLen dimensions across inputs
            current_shape_prefix = tensor.shape[:-1] # (B, T) or potentially just (B,)
            if expected_shape_prefix is None:
                expected_shape_prefix = current_shape_prefix
            elif current_shape_prefix != expected_shape_prefix:
                 raise ValueError(f"Input tensors have inconsistent batch/sequence dimensions. "
                                  f"Expected {expected_shape_prefix}, but got {current_shape_prefix} for key '{key}'.")

            selected_tensors.append(tensor)

        # 2. Concatenate selected inputs along the feature dimension
        if len(selected_tensors) == 1:
            combined_input = selected_tensors[0]
        else:
            combined_input = torch.cat(selected_tensors, dim=-1)

        # Verify concatenated dimension
        if combined_input.shape[-1] != self.total_input_dim:
             # This should ideally not happen if expected_inputs is correct
             raise RuntimeError(f"Internal error: Concatenated input dimension {combined_input.shape[-1]} "
                                f"does not match expected total input dimension {self.total_input_dim}.")

        # 3. Pass through MLP
        output = self.mlp(combined_input)

        return output

    def extra_repr(self) -> str:
        input_keys = list(self.expected_inputs.keys())
        return f"input_keys={input_keys}, total_input_dim={self.total_input_dim}, output_dim={self.output_dim}\n  (mlp): {self.mlp}"


# --- Placeholder/Example for more complex branches ---

# class AttentionBranch(nn.Module):
#     """ Example: A branch that internally uses TM-KLA """
#     def __init__(self, embed_dim, feature_mapper, num_heads, **kwargs):
#         super().__init__()
#         self.attention = TMLinKernelAttn(embed_dim, feature_mapper, num_heads, **kwargs)
#         # Potentially add pre/post processing layers specific to this branch
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
#         # Extract necessary inputs for attention, e.g., hidden state
#         x = inputs.get('hidden_state')
#         temp_input = inputs.get('temperature_input') # Needs careful construction
#         if x is None:
#             raise ValueError("AttentionBranch requires 'hidden_state' in inputs.")

#         # Assume attention output has same dim as input for residual connection
#         attn_output = self.attention(x, temp_input=temp_input)
#         output = self.norm(x + attn_output) # Example: Residual connection + Norm
#         return output

# class WaveletBranch(nn.Module): ...
# class HierarchicalStateBranch(nn.Module): ...


# Example Usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    B, T = 4, 10
    D_state = 128
    D_kappa = 64
    D_out_branch = 128

    # Define expected inputs and their dimensions
    expected = {'hidden_state': D_state, 'kernel_features': D_kappa}

    # Create the MLP branch
    mlp_branch = MLPBranch(
        expected_inputs=expected,
        output_dim=D_out_branch,
        hidden_dim=256,
        num_layers=2,
        device=device,
        dtype=dtype
    )
    print("--- MLPBranch ---")
    print(mlp_branch)

    # Create dummy input dictionary
    dummy_inputs = {
        'hidden_state': torch.randn(B, T, D_state, device=device, dtype=dtype),
        'kernel_features': torch.randn(B, T, D_kappa, device=device, dtype=dtype),
        'wavelet_coeffs': torch.randn(B, T, 32, device=device, dtype=dtype) # Extra input ignored by this branch
    }

    # Forward pass
    print("\nRunning forward pass...")
    output = mlp_branch(dummy_inputs)

    print("\nInput shapes:")
    for key, val in dummy_inputs.items():
        print(f"  '{key}': {val.shape}")

    print("\nOutput shape:", output.shape)
    assert output.shape == (B, T, D_out_branch)
    print("Output sample:", output[0, 0, :8].detach().cpu().numpy())

    # Test error handling
    print("\nTesting missing input error...")
    bad_inputs = {'hidden_state': torch.randn(B, T, D_state, device=device, dtype=dtype)}
    try:
        mlp_branch(bad_inputs)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nTesting wrong dimension error...")
    bad_inputs_dim = {
        'hidden_state': torch.randn(B, T, D_state, device=device, dtype=dtype),
        'kernel_features': torch.randn(B, T, D_kappa + 1, device=device, dtype=dtype) # Wrong dim
    }
    try:
        mlp_branch(bad_inputs_dim)
    except ValueError as e:
        print(f"Caught expected error: {e}")


