# ultra_rwka/components/moe/moe_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Tuple, Type, Literal, List, Dict, Sequence

# Imports from within the library
from .gating import GatingNetwork
# Import base class or specific types from branches - use Any for flexibility now
# from .branches import SpecialistBranch # Assuming a base class exists
from torch.nn import Module as SpecialistBranch # Use nn.Module as base if no specific base defined

class MoELayer(nn.Module):
    """
    Top-Level Mixture-of-Experts (MoE) Layer.

    Combines a gating network with multiple expert branches. It routes the input
    to selected experts (dense or sparse) and combines their outputs.
    Based on §3.5 of the Ultra-RWKA paper.
    """
    def __init__(self,
                 experts: Sequence[SpecialistBranch],
                 gating_network: GatingNetwork,
                 output_dim: int, # Expected output dim from experts & the layer
                 device=None, # Optional device/dtype propagation
                 dtype=None):
        """
        Args:
            experts (Sequence[SpecialistBranch]): A sequence (e.g., list or nn.ModuleList)
                of instantiated expert branch modules. All experts must output tensors
                of shape (..., output_dim).
            gating_network (GatingNetwork): An instantiated GatingNetwork module.
                Its `num_experts` must match len(experts).
            output_dim (int): The expected output dimension from each expert and this layer.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        if not experts:
            raise ValueError("Expert list cannot be empty.")
        if not isinstance(gating_network, GatingNetwork):
             raise TypeError("gating_network must be an instance of GatingNetwork.")

        self.num_experts = len(experts)
        if gating_network.num_experts != self.num_experts:
            raise ValueError(f"Number of experts in gating_network ({gating_network.num_experts}) "
                             f"must match the number of provided expert modules ({self.num_experts}).")

        self.experts = nn.ModuleList(experts) # Ensure experts are properly registered
        self.gating_network = gating_network
        self.output_dim = output_dim
        self.gating_type = gating_network.gating_type # Store for routing logic
        self.k = gating_network.k if hasattr(gating_network, 'k') else 1 # Get k if top-k

        # TODO: Verify expert output dims dynamically? Hard without running forward.
        # Rely on user providing correct output_dim for now.

    def forward(self,
                gating_input: torch.Tensor,
                branch_inputs: Dict[str, torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MoE layer.

        Args:
            gating_input (torch.Tensor): Input tensor for the GatingNetwork,
                shape (..., input_dim_gate).
            branch_inputs (Dict[str, torch.Tensor]): Dictionary of inputs required by
                the expert branches. Keys should match those expected by the branches.
                Values are tensors, typically shape (Batch, SeqLen, feature_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - final_output: Combined output from experts, shape (..., output_dim).
                - aux_loss: Auxiliary load balancing loss from gating (scalar tensor).
        """
        # 1. Compute Gating Weights and Auxiliary Loss
        # Ensure gating input has correct dtype
        expected_gate_dtype = next(self.gating_network.parameters()).dtype
        if gating_input.dtype != expected_gate_dtype:
            gating_input = gating_input.to(dtype=expected_gate_dtype)

        gate_weights, aux_loss = self.gating_network(gating_input)
        # gate_weights shape: (..., num_experts) - can be dense or sparse

        # --- Prepare for Expert Execution ---
        # Assume all branch inputs have the same batch/sequence dimensions
        # Use shape from one of the branch inputs (e.g., the first one listed by an expert)
        # This assumes branch_inputs dictionary is not empty and has valid tensors
        first_branch_input_key = next(iter(self.experts[0].expected_inputs.keys())) # Requires experts expose expected_inputs
        # Or get shape from gating_input if appropriate
        batch_seq_shape = gating_input.shape[:-1] # Shape like (B, T) or (B*T,)
        original_ndim = gating_input.ndim # To reshape output later

        # Flatten batch/sequence dimensions for potentially easier sparse routing later
        # Although the simple implementation below doesn't strictly need it yet.
        # num_tokens = math.prod(batch_seq_shape)
        # flat_gate_weights = gate_weights.view(num_tokens, self.num_experts)

        # --- Execute Experts and Combine ---
        # We need to handle dense (softmax) and sparse (top-k) routing differently
        # for efficiency, but will start with a functionally correct (but potentially
        # inefficient for sparse) implementation first.

        # Compute outputs for ALL experts first
        # Requires experts to handle the input dictionary correctly
        expert_outputs_list = []
        for i, expert in enumerate(self.experts):
            try:
                # Ensure branch inputs have correct dtype for this expert
                # (Assuming all experts use same dtype for simplicity here)
                # expected_expert_dtype = next(expert.parameters()).dtype
                # processed_branch_inputs = {
                #     k: v.to(dtype=expected_expert_dtype) if v.dtype != expected_expert_dtype else v
                #     for k, v in branch_inputs.items()
                # }
                # Use inputs directly, assuming dtypes are handled or consistent
                expert_out = expert(branch_inputs) # Shape (..., output_dim)
                expert_outputs_list.append(expert_out)
            except Exception as e:
                 print(f"Error during forward pass of expert {i} ({type(expert).__name__}): {e}")
                 raise e

        # Stack expert outputs along a new dimension
        # stacked_outputs shape: (..., num_experts, output_dim)
        stacked_outputs = torch.stack(expert_outputs_list, dim=-2)

        # Combine outputs using gate weights
        # Unsqueeze weights to match stacked_outputs: (..., num_experts, 1)
        gate_weights_expanded = gate_weights.unsqueeze(-1)

        # Weighted sum: Multiply outputs by weights and sum along expert dimension
        # Element-wise product broadcasts weights: (..., num_experts, output_dim)
        # Sum along num_experts dim (-2): (..., output_dim)
        # This works for both dense (softmax) and sparse (top-k, inefficiently) weights
        final_output = torch.sum(stacked_outputs * gate_weights_expanded, dim=-2)

        # --- Inefficient Sparse Implementation Warning ---
        if self.gating_type == 'top_k' and self.training: # Only warn during training maybe
             warnings.warn("Using inefficient implementation for sparse Top-K MoE routing. "
                           "All experts are computed. For performance, implement token dispatching.")

        # Ensure output shape matches expected output_dim
        if final_output.shape[-1] != self.output_dim:
             # This indicates a mismatch between expert output and layer config
             raise RuntimeError(f"MoELayer output dimension ({final_output.shape[-1]}) "
                                f"does not match expected output_dim ({self.output_dim}). "
                                "Check expert branch implementations.")

        return final_output, aux_loss

    def extra_repr(self) -> str:
        s = f"num_experts={self.num_experts}, output_dim={self.output_dim}\n"
        s += f"  (gating_network): {self.gating_network}\n"
        # Show expert types concisely
        expert_types = [type(exp).__name__ for exp in self.experts]
        s += f"  (experts): ModuleList containing types: {expert_types}"
        # Optionally list all experts if not too many
        # for i, exp in enumerate(self.experts):
        #     s += f"\n    ({i}): {exp}"
        return s

# Example Usage
if __name__ == '__main__':
    from .branches import MLPBranch # Import MLPBranch for example

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Config
    B, T = 4, 10
    D_gate_in = 128 # Input dim for gating network (e.g., hidden state dim)
    D_branch_out = 128 # Output dim for experts and MoE layer
    D_branch_in_h = 128 # Dim of 'hidden_state' input for MLPBranch
    D_branch_in_k = 64  # Dim of 'kernel_features' input for MLPBranch

    Num_Experts = 4
    K_top = 2

    # --- Create Experts ---
    # Example: All experts are the same MLP type
    expert_input_config = {'hidden_state': D_branch_in_h, 'kernel_features': D_branch_in_k}
    experts_list = [
        MLPBranch(
            expected_inputs=expert_input_config,
            output_dim=D_branch_out,
            hidden_dim=256,
            num_layers=2,
            device=device, dtype=dtype
        ) for _ in range(Num_Experts)
    ]

    # --- Create Gating Network (Top-K) ---
    gating_net = GatingNetwork(
        input_dim=D_gate_in,
        num_experts=Num_Experts,
        gating_type='top_k',
        k=K_top,
        add_noise=True,
        device=device, dtype=dtype
    )

    # --- Create MoE Layer ---
    moe_layer = MoELayer(
        experts=experts_list,
        gating_network=gating_net,
        output_dim=D_branch_out, # Must match expert output dim
        device=device, dtype=dtype
    )
    print("--- MoELayer (Top-K) ---")
    print(moe_layer)

    # --- Create Dummy Inputs ---
    gating_in = torch.randn(B, T, D_gate_in, device=device, dtype=dtype)
    branch_in_dict = {
        'hidden_state': torch.randn(B, T, D_branch_in_h, device=device, dtype=dtype),
        'kernel_features': torch.randn(B, T, D_branch_in_k, device=device, dtype=dtype),
        'other_input': torch.randn(B, T, 32, device=device, dtype=dtype) # Ignored by MLPBranch
    }

    # --- Forward Pass (Training) ---
    print("\nRunning forward pass (training mode)...")
    moe_layer.train()
    output, aux_loss = moe_layer(gating_in, branch_in_dict)

    print("Gating input shape:", gating_in.shape)
    print("Branch inputs keys:", list(branch_in_dict.keys()))
    print("Output shape:", output.shape)
    print("Auxiliary Loss:", aux_loss.item())
    assert output.shape == (B, T, D_branch_out)
    assert aux_loss.ndim == 0 # Should be scalar

    # --- Forward Pass (Eval) ---
    print("\nRunning forward pass (eval mode)...")
    moe_layer.eval()
    output_eval, aux_loss_eval = moe_layer(gating_in, branch_in_dict)
    print("Output shape (eval):", output_eval.shape)
    print("Auxiliary Loss (eval):", aux_loss_eval.item()) # Should be 0 in eval
    assert output_eval.shape == (B, T, D_branch_out)
    assert aux_loss_eval.item() == 0.0

    # --- Test with Softmax Gating ---
    softmax_gating_net = GatingNetwork(
        input_dim=D_gate_in,
        num_experts=Num_Experts,
        gating_type='softmax',
        device=device, dtype=dtype
    )
    moe_layer_softmax = MoELayer(
        experts=experts_list, # Reuse same experts
        gating_network=softmax_gating_net,
        output_dim=D_branch_out,
        device=device, dtype=dtype
    )
    print("\n--- MoELayer (Softmax) ---")
    print(moe_layer_softmax)
    output_sm, aux_loss_sm = moe_layer_softmax(gating_in, branch_in_dict)
    print("\nOutput shape (softmax):", output_sm.shape)
    print("Auxiliary Loss (softmax):", aux_loss_sm.item()) # Should be 0
    assert output_sm.shape == (B, T, D_branch_out)
    assert aux_loss_sm.item() == 0.0


