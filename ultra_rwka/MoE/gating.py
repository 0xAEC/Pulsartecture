# ultra_rwka/components/moe/gating.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Type, Literal

# Imports from within the library
from ..projections import LinearProjection

# Helper for stable division
def _safe_divide(num, den, eps=1e-8):
    return num / (den.clamp(min=eps))

class GatingNetwork(nn.Module):
    """
    Implements the gating network (GateNet_g) for the Mixture-of-Experts layer (§3.5).

    Takes routing inputs (e.g., hidden states) and outputs probabilities or routing
    decisions for selecting/weighting expert branches. Supports standard Softmax
    gating and sparse Top-K gating with optional noise for load balancing.
    """
    _supported_gating_types = {'softmax', 'top_k'}

    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 1,
                 gating_type: Literal['softmax', 'top_k'] = 'softmax',
                 k: int = 1, # For Top-K gating
                 add_noise: bool = True, # For Top-K load balancing (Switch Transformer style)
                 noise_epsilon: float = 1e-2, # Epsilon for noise calculation stability
                 activation_cls: Type[nn.Module] = nn.GELU,
                 use_bias: bool = True,
                 initialize_weights: bool = True,
                 device=None,
                 dtype=None):
        """
        Args:
            input_dim (int): Dimension of the input tensor used for gating decisions.
            num_experts (int): The number of expert branches (B).
            hidden_dim (Optional[int]): Hidden dimension for the gating MLP. Defaults to input_dim.
            num_layers (int): Number of layers in the gating MLP (>= 1). Defaults to 1.
            gating_type (Literal['softmax', 'top_k']): Type of gating mechanism. Defaults to 'softmax'.
            k (int): Number of experts to select for Top-K gating. Defaults to 1.
            add_noise (bool): If True and gating_type is 'top_k', add noise to logits during training
                              for load balancing. Defaults to True.
            noise_epsilon (float): Small value for numerical stability in noise calculation. Defaults to 1e-2.
            activation_cls (Type[nn.Module]): Activation class for hidden layers in MLP. Defaults to nn.GELU.
            use_bias (bool): Whether linear layers in MLP use bias. Defaults to True.
            initialize_weights (bool): Whether to apply standard initialization. Defaults to True.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        if gating_type not in self._supported_gating_types:
            raise ValueError(f"Unsupported gating_type: {gating_type}. Choose from {self._supported_gating_types}")
        if gating_type == 'top_k' and not (0 < k <= num_experts):
            raise ValueError(f"For top_k gating, k ({k}) must be > 0 and <= num_experts ({num_experts})")
        if num_experts <= 0:
             raise ValueError("num_experts must be positive.")

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.gating_type = gating_type
        self.k = k
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # --- Gating MLP ---
        _hidden_dim = hidden_dim if hidden_dim is not None else input_dim # Default hidden to input
        mlp_layers = []
        current_dim = input_dim
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            output_dim = num_experts if is_last_layer else _hidden_dim
            linear_layer = nn.Linear(current_dim, output_dim, bias=use_bias, **self.factory_kwargs)

            if initialize_weights:
                # Typically smaller init for gating networks
                nn.init.normal_(linear_layer.weight, std=0.02)
                if use_bias and linear_layer.bias is not None:
                    nn.init.zeros_(linear_layer.bias)

            mlp_layers.append(linear_layer)
            if not is_last_layer:
                mlp_layers.append(activation_cls())
                current_dim = _hidden_dim
            # No activation after final layer (outputs logits)

        self.gating_mlp = nn.Sequential(*mlp_layers)

        # --- Noise Projection (for Top-K load balancing) ---
        self.noise_projection: Optional[nn.Linear] = None
        if self.gating_type == 'top_k' and self.add_noise:
            self.noise_projection = nn.Linear(input_dim, num_experts, bias=False, **self.factory_kwargs)
            if initialize_weights:
                 nn.init.normal_(self.noise_projection.weight, std=1.0 / math.sqrt(input_dim))


    def _compute_softmax_routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes dense softmax weights and zero auxiliary loss. """
        # Ensure calculations are done in float32 for stability, then cast back
        original_dtype = logits.dtype
        weights = F.softmax(logits.float(), dim=-1).to(original_dtype) # (..., num_experts)
        aux_loss = torch.tensor(0.0, device=logits.device, dtype=original_dtype) # No aux loss for softmax
        return weights, aux_loss

    def _compute_top_k_routing(self, logits: torch.Tensor, routing_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes sparse top-k weights and auxiliary load balancing loss. """
        # Input logits shape: (..., num_experts)
        # routing_input shape: (..., input_dim) - needed for noise projection

        if self.k == self.num_experts:
            # If k equals num_experts, it's equivalent to softmax gating
            warnings.warn("Top-K gating with k == num_experts is equivalent to softmax gating.")
            return self._compute_softmax_routing(logits)

        # --- Optional Noise Addition ---
        if self.add_noise and self.training and self.noise_projection is not None:
            # Switch Transformer noise: noise_stddev = softplus(W_noise(x)) / num_experts
            # noise = N(0, 1) * noise_stddev
            # noisy_logits = logits + noise
            noise_logits = self.noise_projection(routing_input.to(self.noise_projection.weight.dtype)) # (..., num_experts)
            noise_stddev = F.softplus(noise_logits) + self.noise_epsilon # Ensure positive stddev
            # Sample standard Gaussian noise
            raw_noise = torch.randn_like(logits)
            noise = raw_noise * noise_stddev
            logits = logits + noise

        # --- Top-K Selection ---
        # Select top-k values and their indices along the expert dimension (-1)
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1) # (..., k), (..., k)

        # --- Create Sparse Weights ---
        # Create a mask based on top-k indices
        sparse_weights = torch.zeros_like(logits) # (..., num_experts)
        # Compute softmax probabilities only for the selected top-k experts
        top_k_probs = F.softmax(top_k_logits.float(), dim=-1).to(logits.dtype) # (..., k)
        # Scatter these probabilities back into the sparse weights tensor at the correct indices
        sparse_weights.scatter_(-1, top_k_indices, top_k_probs) # (..., num_experts)

        # --- Auxiliary Load Balancing Loss (Switch Transformer style) ---
        aux_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if self.training:
            # Calculate f_i: fraction of tokens routed to expert i
            # Calculate P_i: average router probability for expert i
            # L_aux = alpha * sum_i(f_i * P_i)
            # Need to handle potential batch/sequence dimensions flattened

            original_shape = logits.shape
            num_tokens = math.prod(original_shape[:-1]) # Total number of tokens (B*T*...)
            num_experts = self.num_experts

            # Flatten logits and weights for easier calculation
            flat_logits = logits.view(num_tokens, num_experts) # (N, E)
            flat_weights = sparse_weights.view(num_tokens, num_experts) # (N, E)

            # f_i: Count how many tokens selected expert i (non-zero weight) / total tokens
            # Indicator if expert i was selected for token n (approx by checking non-zero weight)
            indicator = (flat_weights > 0.0).float() # (N, E)
            tokens_per_expert = indicator.sum(dim=0) # (E,)
            fraction_routed_f = tokens_per_expert / num_tokens # (E,)

            # P_i: Average probability assigned to expert i across all tokens
            # Use the original (potentially noisy) logits passed through softmax
            # This represents the router's confidence in assigning to expert i
            router_probs_p = F.softmax(flat_logits.float(), dim=-1).to(flat_logits.dtype) # (N, E)
            avg_prob_p = router_probs_p.mean(dim=0) # (E,)

            # Aux loss = sum (f_i * P_i) * num_experts (scale factor often used)
            # Coefficient alpha is typically tuned (e.g., 0.01) - handle outside this module
            aux_loss = torch.sum(fraction_routed_f * avg_prob_p) * num_experts


        return sparse_weights, aux_loss


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes gating weights and auxiliary loss.

        Args:
            x (torch.Tensor): Input tensor for gating decision, shape (..., input_dim).
                              Commonly (Batch, SeqLen, input_dim) or (Batch * SeqLen, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - weights: Routing weights, shape (..., num_experts). Dense for softmax, sparse for top_k.
                - aux_loss: Auxiliary load balancing loss (scalar tensor). Zero for softmax gating.
        """
        # Ensure input type matches network parameters
        expected_dtype = next(self.gating_mlp.parameters()).dtype
        if x.dtype != expected_dtype:
            x = x.to(dtype=expected_dtype)

        # Compute logits
        logits = self.gating_mlp(x) # (..., num_experts)

        # Apply gating strategy
        if self.gating_type == 'softmax':
            weights, aux_loss = self._compute_softmax_routing(logits)
        elif self.gating_type == 'top_k':
            # Pass original input x for noise calculation if needed
            weights, aux_loss = self._compute_top_k_routing(logits, x)
        else:
            # Should be caught by __init__
            raise RuntimeError(f"Internal error: Invalid gating_type '{self.gating_type}'")

        return weights, aux_loss

    def extra_repr(self) -> str:
        s = f"input_dim={self.input_dim}, num_experts={self.num_experts}, gating_type='{self.gating_type}'"
        if self.gating_type == 'top_k':
            s += f", k={self.k}, add_noise={self.add_noise}"
        s += f"\n  (gating_mlp): {self.gating_mlp}"
        if self.noise_projection is not None:
             s += f"\n  (noise_projection): {self.noise_projection}"
        return s

# Example Usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    B, T, D = 4, 10, 128 # Batch, Time, Input Dim
    E = 8 # Num Experts
    K = 2 # Top-K

    dummy_input = torch.randn(B, T, D, device=device, dtype=dtype)
    dummy_input_flat = dummy_input.view(B * T, D)

    print("--- Softmax Gating ---")
    softmax_gate = GatingNetwork(
        input_dim=D,
        num_experts=E,
        gating_type='softmax',
        device=device, dtype=dtype
    )
    print(softmax_gate)
    softmax_gate.train() # Aux loss is 0 anyway, but good practice
    weights_sm, loss_sm = softmax_gate(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Weights shape:", weights_sm.shape)
    print("Aux Loss:", loss_sm.item())
    assert weights_sm.shape == (B, T, E)
    assert torch.allclose(weights_sm.sum(dim=-1), torch.ones(B, T, device=device, dtype=dtype))

    print("\n--- Top-K Gating (k=2, with noise) ---")
    topk_gate = GatingNetwork(
        input_dim=D,
        num_experts=E,
        gating_type='top_k',
        k=K,
        add_noise=True,
        device=device, dtype=dtype
    )
    print(topk_gate)
    topk_gate.train() # Enable noise and aux loss calculation
    weights_tk, loss_tk = topk_gate(dummy_input_flat) # Test with flattened input
    print("Input shape (flat):", dummy_input_flat.shape)
    print("Weights shape (flat):", weights_tk.shape)
    print("Aux Loss:", loss_tk.item())
    assert weights_tk.shape == (B * T, E)
    # Check sparsity: each token should have exactly K non-zero weights
    num_non_zero = torch.count_nonzero(weights_tk, dim=-1)
    print("Num non-zero weights per token (should be k=2):", num_non_zero.float().mean().item())
    assert torch.all(num_non_zero == K)
    # Check sum is 1 for selected experts
    print("Sum of weights per token (should be 1):", weights_tk.sum(dim=-1).mean().item())
    assert torch.allclose(weights_tk.sum(dim=-1), torch.ones(B * T, device=device, dtype=dtype))

    # Test Top-K eval mode (no noise, no aux loss)
    topk_gate.eval()
    weights_tk_eval, loss_tk_eval = topk_gate(dummy_input_flat)
    print("\nAux Loss (eval mode, should be 0):", loss_tk_eval.item())
    assert loss_tk_eval.item() == 0.0
    num_non_zero_eval = torch.count_nonzero(weights_tk_eval, dim=-1)
    assert torch.all(num_non_zero_eval == K) # Still selects K experts


