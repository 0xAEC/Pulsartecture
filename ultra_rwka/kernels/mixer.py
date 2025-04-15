# ultra_rwka/components/kernels/mixer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Type # Added Type for activation class

# Import the base class and potentially specific types for type hinting/checks
from .feature_maps import FeatureMap # Assuming FeatureMap lives here

class GateNet(nn.Module):
    """
    MLP gating network to compute mixture weights (logits).
    Takes concatenated input features and optional context. Includes optional dropout.
    """
    def __init__(self,
                 input_dim: int,
                 num_outputs: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 1,
                 use_bias: bool = True,
                 activation_cls: Type[nn.Module] = nn.GELU, # Pass activation class
                 dropout_p: float = 0.0, # Add dropout probability
                 initialize_weights: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.dropout_p = dropout_p

        if num_layers < 1:
            raise ValueError("GateNet num_layers must be at least 1.")
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be between 0.0 and 1.0, got {dropout_p}")

        _hidden_dim = hidden_dim if hidden_dim is not None else max(input_dim // 2, num_outputs) # Heuristic

        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            output_dim = num_outputs if is_last_layer else _hidden_dim
            linear_layer = nn.Linear(current_dim, output_dim, bias=use_bias, **self.factory_kwargs)

            if initialize_weights:
                nn.init.xavier_uniform_(linear_layer.weight)
                if use_bias and linear_layer.bias is not None:
                    nn.init.zeros_(linear_layer.bias)

            layers.append(linear_layer)

            if not is_last_layer:
                layers.append(activation_cls())
                # Add dropout after activation and before next linear layer (common practice)
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(p=dropout_p))
                current_dim = _hidden_dim
            # No activation or dropout after the final output layer (logits)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes gate logits. """
        return self.network(x)

# Try JIT scripting the mixer's forward pass
# Note: Success depends heavily on PyTorch version and specifics of FeatureMap implementations.
# Iteration over ModuleList can sometimes be problematic for JIT.
@torch.jit.script
def _kernel_mixer_forward_script(
    x: torch.Tensor,
    feature_maps_list: List[FeatureMap], # JIT needs list of specific base class
    gate_net: GateNet,
    context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    """
    JIT-scriptable version of KernelFeatureMixer forward pass.
    """
    # 1. Prepare GateNet input
    if context is not None:
        if x.shape[:-1] != context.shape[:-1]:
            if context.ndim == x.ndim - 1 and x.ndim > 1:
                # Auto-expand context: (B, C) -> (B, T, C) to match x=(B, T, D)
                context = context.unsqueeze(1).expand(-1, x.size(1), -1)
            # Consider adding elif context.ndim == x.ndim here if needed
            elif context.ndim != x.ndim:
                 raise ValueError(f"Shape mismatch for concat: x {x.shape}, context {context.shape}")

        # Check again after potential expansion
        if x.shape[:-1] == context.shape[:-1]:
            gate_input = torch.cat([x, context], dim=-1)
            # Minimal runtime check (static check done in __init__)
            # assert gate_input.shape[-1] == gate_net.input_dim, "Runtime gate input dim mismatch"
        else:
            raise ValueError(f"Cannot automatically concatenate x ({x.shape}) and context ({context.shape}) after expansion")
    else:
        gate_input = x
        # Minimal runtime check
        # assert gate_input.shape[-1] == gate_net.input_dim, "Runtime gate input dim mismatch (no context)"


    # 2. Compute mixture weights
    # Ensure dtype matches the gate_net parameters implicitly via gate_net call
    # Use float() for softmax intermediate calculations for stability if using autocast/AMP
    gate_logits = gate_net(gate_input.to(dtype=next(gate_net.parameters()).dtype))
    mixture_weights = torch.softmax(gate_logits.float(), dim=-1).to(dtype=gate_logits.dtype) # Shape: (*, num_feature_maps)

    # 3. Compute base feature maps and apply weights
    # Pre-allocate tensor for stacked features for JIT compatibility (avoid list append)
    num_feature_maps = len(feature_maps_list)
    # Assume feature_dim is consistent (checked in __init__)
    # Get feature_dim and other properties from the first map
    feature_dim = feature_maps_list[0].feature_dim
    output_dtype = feature_maps_list[0].forward(x[0:1]).dtype # Infer dtype from first map output
    device = x.device

    # Use torch.empty for pre-allocation
    # Shape: (*batch_dims, num_feature_maps, feature_dim)
    batch_shape = x.shape[:-1]
    stacked_features = torch.empty(batch_shape + (num_feature_maps, feature_dim), dtype=output_dtype, device=device)

    # This loop over ModuleList might still cause JIT issues.
    for i in range(num_feature_maps): # Use range loop for JIT
        feature_map_module = feature_maps_list[i]
        phi_i = feature_map_module(x) # (*, feature_dim)
        stacked_features[..., i, :] = phi_i

    # 4. Apply weights and sum using torch.einsum for clarity/potential optimization
    # Einsum: Multiply stacked features (*, n, f) by weights (*, n) -> sum over n -> (*, f)
    # Need to ensure broadcasting works correctly if mixture_weights doesn't have all batch dims
    # Let's match dims explicitly first for robustness with einsum
    weight_unsqueeze_count = stacked_features.ndim - mixture_weights.ndim - 1 # num_feature_maps dim = -2, feature_dim = -1
    unsqueezed_weights = mixture_weights
    for _ in range(weight_unsqueeze_count):
        unsqueezed_weights = unsqueezed_weights.unsqueeze(0)
    # Add feature dim broadcast
    unsqueezed_weights = unsqueezed_weights.unsqueeze(-1) # (*batch_dims, num_feature_maps, 1)

    # Element-wise multiplication, then sum
    # weighted_features = stacked_features * unsqueezed_weights # (*, n, f)
    # mixed_features = torch.sum(weighted_features, dim=-2) # Sum over n -> (*, f)

    # Alternative using einsum:
    # '...nf,...n->...f' - Sum product over the 'n' dimension (num_feature_maps)
    mixed_features = torch.einsum('...nf,...n->...f', stacked_features, mixture_weights)


    return mixed_features # kappa_t


class KernelFeatureMixer(nn.Module):
    """
    Dynamically mixes multiple kernel feature map approximations based on input and context.
    Enhanced with GateNet dropout and torch.einsum summation. Attempts JIT scripting.

    Implements Equation from Sec 3.2[cite: 28]:
        kappa_t = sum_{i=1}^{I} pi_t^{(i)} * FeatureMap_i(x_t)
    where pi_t = Softmax(GateNet(x_t, h_{t-1})). [cite: 27]
    """
    def __init__(self,
                 feature_maps: Sequence[FeatureMap],
                 gate_input_dim: int,
                 gate_hidden_dim: Optional[int] = None,
                 gate_layers: int = 1,
                 gate_activation_cls: Type[nn.Module] = nn.GELU,
                 gate_dropout_p: float = 0.0, # Added dropout
                 device=None,
                 dtype=None):
        super().__init__()
        if not feature_maps:
            raise ValueError("feature_maps list/tuple cannot be empty.")

        # Use nn.ModuleList to ensure modules are properly registered
        self.feature_maps = nn.ModuleList(feature_maps)
        self.num_feature_maps = len(feature_maps)

        # Verify all feature maps have the same output dimension
        self.feature_dim = feature_maps[0].feature_dim
        for i, fm in enumerate(self.feature_maps):
            if not isinstance(fm, FeatureMap):
                 warnings.warn(f"Item at index {i} is not an instance of FeatureMap base class.")
            if fm.feature_dim != self.feature_dim:
                raise ValueError(f"Feature map at index {i} has dimension {fm.feature_dim}, "
                                 f"but expected {self.feature_dim} based on the first map.")

        # Initialize the gating network with dropout option
        self.gate_net = GateNet(
            input_dim=gate_input_dim,
            num_outputs=self.num_feature_maps,
            hidden_dim=gate_hidden_dim,
            num_layers=gate_layers,
            activation_cls=gate_activation_cls,
            dropout_p=gate_dropout_p, # Pass dropout prob
            device=device,
            dtype=dtype
        )
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # Try to script the forward function
        # self.forward = torch.jit.script(self._forward_impl) # --> Needs _forward_impl method
        # Or script the helper function and call it

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the dynamically mixed feature map representation kappa_t.
        Calls the (potentially JIT-scripted) helper function.

        Args:
            x (torch.Tensor): Input tensor (*, feature_map_input_dim).
            context (Optional[torch.Tensor]): Optional context tensor (*, context_dim).

        Returns:
            torch.Tensor: Mixed feature tensor kappa_t (*, feature_dim).
        """
        # Note: JIT scripting the helper works better than scripting this method directly
        # if the helper correctly handles iterating ModuleList (which might still fail).
        # Pass the ModuleList as a List[FeatureMap] for JIT type compatibility.
        feature_maps_list = [fm for fm in self.feature_maps]
        try:
            return _kernel_mixer_forward_script(x, feature_maps_list, self.gate_net, context)
        except Exception as e:
            # Fallback to eager execution if JIT fails (common with ModuleList iteration)
            warnings.warn(f"JIT scripting for KernelFeatureMixer failed: {e}. Running in eager mode.")
            return self._forward_eager(x, context) # Provide an eager fallback

    def _forward_eager(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Eager mode implementation matching the scripted version """
         # 1. Prepare GateNet input (Copied from script version for consistency)
        if context is not None:
            if x.shape[:-1] != context.shape[:-1]:
                if context.ndim == x.ndim - 1 and x.ndim > 1:
                    context = context.unsqueeze(1).expand(-1, x.size(1), -1)
                elif context.ndim != x.ndim:
                    raise ValueError(f"Shape mismatch for concat: x {x.shape}, context {context.shape}")

            if (x.shape[:-1] == context.shape[:-1]):
                gate_input = torch.cat([x, context], dim=-1)
                if gate_input.shape[-1] != self.gate_net.input_dim:
                     raise ValueError(f"Concatenated input dim ({gate_input.shape[-1]}) does not match GateNet expected input dim ({self.gate_net.input_dim}).")
            else:
                 raise ValueError(f"Cannot automatically concatenate x ({x.shape}) and context ({context.shape})")
        else:
            gate_input = x
            if gate_input.shape[-1] != self.gate_net.input_dim:
                 raise ValueError(f"Input dim ({gate_input.shape[-1]}) does not match GateNet expected input dim ({self.gate_net.input_dim}) when context is None.")

        # 2. Compute mixture weights
        gate_logits = self.gate_net(gate_input.to(dtype=next(self.gate_net.parameters()).dtype))
        mixture_weights = torch.softmax(gate_logits.float(), dim=-1).to(dtype=gate_logits.dtype)

        # 3. Compute base features
        # Use list comprehension again in eager mode
        base_features = [fm(x) for fm in self.feature_maps]
        stacked_features = torch.stack(base_features, dim=-2) # (*, num_feature_maps, feature_dim)

        # 4. Apply weights and sum using torch.einsum
        mixed_features = torch.einsum('...nf,...n->...f', stacked_features, mixture_weights)

        return mixed_features

    def extra_repr(self) -> str:
        s = f'num_feature_maps={self.num_feature_maps}, output_feature_dim={self.feature_dim}\n'
        s += f'  (gate_net): {self.gate_net}\n'
        # Use ModuleList repr which is clean
        s += f'  (feature_maps): {self.feature_maps}'
        # for i, fm in enumerate(self.feature_maps):
        #     s += f'  (feature_map_{i}): {fm}\n' # Redundant if using ModuleList repr
        return s.strip()

# Example Usage (minor changes)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 # Use float32 for broader compatibility

    in_d = 32       # Dimension of raw input x for FeatureMap
    context_d = 16  # Dimension of context h
    gate_in_d = in_d + context_d # Example: GateNet sees concat(x, h)
    feat_d = 64     # Output dimension d_k of each feature map
    num_maps = 2    # Number of base feature maps to mix

    batch = 4
    seq = 10

    # Create dummy input and context
    x_input = torch.randn(batch, seq, in_d, device=device, dtype=dtype)
    context_input = torch.randn(batch, seq, context_d, device=device, dtype=dtype)

    # Create base feature maps
    # Ensure they accept `in_d` and output `feat_d`
    fm1 = RandomFourierFeatures(in_dim=in_d, feature_dim=feat_d, gamma=1.0, device=device, dtype=dtype)
    fm2 = LearnablePositiveMap(in_dim=in_d, feature_dim=feat_d, activation='elu+1', device=device, dtype=dtype)
    feature_map_list = [fm1, fm2]

    # Create the mixer with dropout
    mixer = KernelFeatureMixer(
        feature_maps=feature_map_list,
        gate_input_dim=gate_in_d,
        gate_hidden_dim=32,
        gate_layers=2,
        gate_dropout_p=0.1, # Add dropout
        device=device,
        dtype=dtype
    )
    print("--- Mixer with Context ---")
    print(mixer)

    # Set to eval mode to disable dropout for inference checks if needed
    mixer.eval()

    # Forward pass
    # Try with JIT first (might warn/fallback)
    print("\nAttempting forward pass (might use JIT or fallback)...")
    mixed_kappa = mixer(x_input, context_input)

    print("\nInput x shape:", x_input.shape)
    print("Context shape:", context_input.shape)
    print("Mixed kappa shape:", mixed_kappa.shape) # Should be (batch, seq, feat_d)
    assert mixed_kappa.shape == (batch, seq, feat_d)
    print("Mixed kappa sample (eval mode):", mixed_kappa.view(-1)[:8])

    # Test without context
    mixer_no_context = KernelFeatureMixer(
        feature_maps=feature_map_list,
        gate_input_dim=in_d, # Gate only sees x
        gate_dropout_p=0.1,
        device=device,
        dtype=dtype
    )
    print("\n--- Mixer (No Context) ---")
    print(mixer_no_context)
    mixer_no_context.eval()
    mixed_kappa_no_ctx = mixer_no_context(x_input) # Pass only x
    print("\nMixed kappa (no context) shape:", mixed_kappa_no_ctx.shape)
    assert mixed_kappa_no_ctx.shape == (batch, seq, feat_d)
