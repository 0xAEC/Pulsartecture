# ultra_rwka/components/memory/associative.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple, Union, Dict, Type

# Imports from within the library
from ..projections import LinearProjection

# Helper for stable division
def _safe_divide(num, den, eps=1e-8):
    """ Performs safe division: num / (den + eps), ensuring den is positive. """
    return num / (den.clamp(min=eps)) # Clamp denominator instead of adding eps

class ImplicitDAM(nn.Module):
    """
    Implicit Differentiable Augmented Memory (i-DAM).

    An O(1) complexity associative memory based on kernel regression over a fixed
    number of memory bins/prototypes. Inspired by §3.4 of the Ultra-RWKA paper[cite: 6, 39, 70].

    Workflow:
    1. Generate Query Key (k_t) from input/context using Key Network.
    2. Compute Addressing Weights (a_j) using Kernel Softmax between k_t and Prototypes (phi_j).
    3. Read/Retrieve Value (r_t) by weighted sum of Buffer contents (B_j) using addressing weights.
    4. Generate Content Projection (m_t) from input/context using Content Network.
    5. Update Buffer (B_j) using an EMA-like rule based on addressing weights and projected content.
    """
    _supported_distance_metrics = {'euclidean', 'cosine'}

    def __init__(self,
                 input_dim: int,
                 context_dim: int,
                 key_dim: int,
                 value_dim: int,
                 num_bins: int,
                 key_net_config: Optional[Dict] = None, # Config for MLP (layers, hidden_dim, activation_cls)
                 content_net_config: Optional[Dict] = None, # Config for MLP
                 learnable_prototypes: bool = True,
                 prototype_init: str = 'randn', # 'randn', 'uniform'
                 normalize_prototypes: bool = False, # Normalize prototypes before distance calc
                 normalize_keys: bool = False, # Normalize query keys before distance calc
                 distance_metric: str = 'euclidean',
                 temperature: Union[float, str] = 1.0, # float or 'learnable'
                 learning_rate: Union[float, str] = 0.1, # float or 'learnable'
                 temperature_init: float = 1.0, # Initial value if learnable
                 learning_rate_init: float = 0.1, # Initial value if learnable
                 constrain_lambda: bool = True, # Apply sigmoid if lambda is learnable
                 prototype_reg_coeff: float = 0.0, # Coefficient for prototype diversity regularization
                 device=None,
                 dtype=None):
        """
        Args:
            input_dim (int): Dimension of the primary input x_t.
            context_dim (int): Dimension of the context input h_{t-1}.
            key_dim (int): Dimension of query keys (k_t) and prototypes (phi_j).
            value_dim (int): Dimension of memory buffer content (B_j) and projected content (m_t).
            num_bins (int): Number of memory bins / prototypes (N_b).
            key_net_config (Optional[Dict]): Configuration for the Key Network MLP (K).
                Example: {'num_layers': 2, 'hidden_dim': 128, 'activation_cls': nn.GELU}. Defaults to single layer linear.
            content_net_config (Optional[Dict]): Configuration for the Content Network MLP (M_DAM).
                Example: {'num_layers': 1}. Defaults to single layer linear.
            learnable_prototypes (bool): If True, prototypes (phi_j) are learnable Parameters. Defaults to True.
            prototype_init (str): Initialization method for prototypes ('randn', 'uniform'). Defaults to 'randn'.
            normalize_prototypes (bool): If True, L2 normalize prototypes before distance calculation. Defaults to False.
            normalize_keys (bool): If True, L2 normalize query keys before distance calculation. Defaults to False.
            distance_metric (str): Distance metric for addressing ('euclidean', 'cosine'). Defaults to 'euclidean'.
            temperature (Union[float, str]): Temperature (tau) for kernel softmax. Fixed float or 'learnable'. Defaults to 1.0.
            learning_rate (Union[float, str]): Learning rate (lambda) for buffer EMA update. Fixed float or 'learnable'. Defaults to 0.1.
            temperature_init (float): Initial value if temperature is learnable. Defaults to 1.0.
            learning_rate_init (float): Initial value if learning_rate is learnable. Defaults to 0.1.
            constrain_lambda (bool): If True and lambda is learnable, applies sigmoid to keep it in (0, 1). Defaults to True.
            prototype_reg_coeff (float): Coefficient for regularization encouraging prototype diversity. Defaults to 0.0.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_bins = num_bins
        self.learnable_prototypes = learnable_prototypes
        self.normalize_prototypes = normalize_prototypes
        self.normalize_keys = normalize_keys
        self.distance_metric = distance_metric.lower()
        self.constrain_lambda = constrain_lambda
        self.prototype_reg_coeff = prototype_reg_coeff
        self._is_temp_learnable = isinstance(temperature, str) and temperature.lower() == 'learnable'
        self._is_lambda_learnable = isinstance(learning_rate, str) and learning_rate.lower() == 'learnable'
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        if self.distance_metric not in self._supported_distance_metrics:
            raise ValueError(f"Unsupported distance_metric: {distance_metric}. Choose from {self._supported_distance_metrics}")
        if self.normalize_keys and self.distance_metric == 'euclidean':
            warnings.warn("Normalizing keys with Euclidean distance might not be standard practice.")
        if self.normalize_prototypes and self.distance_metric == 'euclidean':
            warnings.warn("Normalizing prototypes with Euclidean distance might not be standard practice.")

        combined_input_dim = input_dim + context_dim

        # --- Networks ---
        key_config = key_net_config or {}
        self.key_net = self._create_mlp(combined_input_dim, key_dim, **key_config)

        content_config = content_net_config or {}
        self.content_net = self._create_mlp(combined_input_dim, value_dim, **content_config)

        # --- Prototypes (phi_j) ---
        prototype_tensor = self._initialize_prototypes(prototype_init)
        if learnable_prototypes:
            self.prototypes = nn.Parameter(prototype_tensor) # Shape: (N_b, key_dim)
        else:
            self.register_buffer('prototypes', prototype_tensor)

        # --- Memory Buffer (B_j) ---
        buffer_tensor = torch.zeros(num_bins, value_dim, **self.factory_kwargs)
        self.register_buffer('buffer', buffer_tensor) # Shape: (N_b, value_dim)

        # --- Temperature (tau) ---
        if self._is_temp_learnable:
            init_val = math.log(math.exp(max(temperature_init, 1e-6)) - 1) # Inverse softplus
            self.temperature_param = nn.Parameter(torch.tensor(init_val, **self.factory_kwargs))
        else:
            if not isinstance(temperature, (float, int)) or temperature <= 0:
                 raise ValueError("Fixed temperature must be a positive float.")
            self.register_buffer('temperature_val', torch.tensor(float(temperature), **self.factory_kwargs))

        # --- Learning Rate (lambda) ---
        if self._is_lambda_learnable:
             init_val = math.log(learning_rate_init / (1.0 - learning_rate_init)) if 0 < learning_rate_init < 1 else 0.0 # Inverse sigmoid
             self.learning_rate_param = nn.Parameter(torch.tensor(init_val, **self.factory_kwargs))
        else:
             if not isinstance(learning_rate, (float, int)) or not (0.0 <= learning_rate <= 1.0):
                 raise ValueError("Fixed learning_rate must be a float in [0, 1]")
             self.register_buffer('learning_rate_val', torch.tensor(float(learning_rate), **self.factory_kwargs))

    def get_temperature(self) -> torch.Tensor:
        """ Returns the current temperature value (scalar tensor). """
        if self._is_temp_learnable:
            # Add epsilon to prevent temperature from becoming exactly zero
            return F.softplus(self.temperature_param) + 1e-6
        else:
            return self.temperature_val

    def get_learning_rate(self) -> torch.Tensor:
        """ Returns the current learning rate value (scalar tensor). """
        if self._is_lambda_learnable:
            if self.constrain_lambda:
                # Sigmoid keeps lambda in (0, 1)
                return torch.sigmoid(self.learning_rate_param)
            else:
                # Return raw parameter if constraint is disabled
                return self.learning_rate_param
        else:
            return self.learning_rate_val

    def _create_mlp(self, in_dim: int, out_dim: int, num_layers: int = 1, hidden_dim: Optional[int] = None, activation_cls: Type[nn.Module] = nn.GELU, **kwargs) -> nn.Sequential:
        """ Helper to create simple MLPs using LinearProjection """
        _hidden_dim = hidden_dim if hidden_dim is not None else max(in_dim // 2, out_dim)
        layers = []
        current_dim = in_dim
        # Pass relevant kwargs from config dict
        proj_kwargs = {k: v for k, v in kwargs.items() if k in ['use_bias', 'initialize']}
        proj_kwargs.update(self.factory_kwargs)

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            layer_out_dim = out_dim if is_last else _hidden_dim
            layers.append(LinearProjection(
                current_dim, layer_out_dim,
                activation_cls=None if is_last else activation_cls,
                initialize=proj_kwargs.get('initialize', 'xavier_uniform'), # Default init
                use_bias=proj_kwargs.get('use_bias', True),
                device=proj_kwargs.get('device'),
                dtype=proj_kwargs.get('dtype')
            ))
            current_dim = layer_out_dim
        return nn.Sequential(*layers)

    def _initialize_prototypes(self, method: str) -> torch.Tensor:
        """ Initialize prototype vectors. """
        if method == 'randn':
            tensor = torch.randn(self.num_bins, self.key_dim, **self.factory_kwargs)
            nn.init.normal_(tensor, mean=0.0, std=0.02) # Small std deviation
        elif method == 'uniform':
            bound = 1.0 / math.sqrt(self.key_dim)
            tensor = torch.rand(self.num_bins, self.key_dim, **self.factory_kwargs) * 2 * bound - bound
        else:
            raise ValueError(f"Unknown prototype_init method: {method}. Choose 'randn' or 'uniform'.")
        return tensor

    def _compute_distances(self, query_key: torch.Tensor) -> torch.Tensor:
        """ Computes distances between query keys and prototypes. """
        # query_key: (B, key_dim)
        # prototypes: (N_b, key_dim) -> Could be Parameter or Buffer

        # Normalize if requested
        q = F.normalize(query_key, p=2, dim=-1) if self.normalize_keys else query_key
        p = F.normalize(self.prototypes, p=2, dim=-1) if self.normalize_prototypes else self.prototypes

        # Expand dims for broadcasting: query (B, 1, D), prototypes (1, N, D) -> (N, D) for cdist
        q_exp = q.unsqueeze(1) # (B, 1, D)
        p_for_cdist = p # (N, D)

        if self.distance_metric == 'euclidean':
            # Use torch.cdist for potentially better efficiency/stability
            # cdist computes pairwise distances: output (B, N)
            # p=2 gives Euclidean distance. Square it for squared Euclidean.
            distances = torch.cdist(q, p_for_cdist, p=2).pow(2) # (B, N)
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            # Ensure inputs are normalized if using cosine distance for meaningful results
            if not (self.normalize_keys and self.normalize_prototypes):
                 warnings.warn("Using cosine distance without normalizing keys and prototypes.")
            # Compute similarity: (B, D) @ (D, N) -> (B, N)
            sim = torch.matmul(q, p_for_cdist.t())
            distances = 1.0 - sim # Range approx [0, 2]
        else:
            raise NotImplementedError(f"Distance metric {self.distance_metric} not implemented.")

        # Clamp minimum distance only for Euclidean to avoid issues with exp(-dist/temp) if dist is tiny negative due to float errors
        if self.distance_metric == 'euclidean':
             distances = torch.clamp(distances, min=0.0)

        return distances # Shape: (B, N)

    def compute_addressing(self, query_key: torch.Tensor) -> torch.Tensor:
        """ Computes addressing weights using Kernel Softmax. """
        distances = self._compute_distances(query_key) # (B, N)
        temperature = self.get_temperature() # Scalar tensor

        # Kernel Softmax: exp(-distance / tau) / sum(exp(-distance / tau))
        scaled_neg_distances = -distances / temperature # (B, N)

        # Use log-softmax for numerical stability, then exponentiate
        log_probs = F.log_softmax(scaled_neg_distances, dim=-1) # (B, N)
        attention_weights = torch.exp(log_probs) # (B, N)

        # # Alternative stable softmax (manual)
        # max_val = torch.max(scaled_neg_distances, dim=-1, keepdim=True)[0]
        # exp_val = torch.exp(scaled_neg_distances - max_val.detach()) # Detach max_val for stability?
        # sum_exp_val = torch.sum(exp_val, dim=-1, keepdim=True)
        # attention_weights = _safe_divide(exp_val, sum_exp_val)

        return attention_weights # a_jmem

    def read(self, query_key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Reads from the memory buffer using the query key. """
        attention_weights = self.compute_addressing(query_key) # (B, N)
        buffer_state = self.buffer # (N, V)
        # einsum: 'bn,nv->bv' (batch, bins) @ (bins, value_dim) -> (batch, value_dim)
        retrieved_value = torch.einsum('bn,nv->bv', attention_weights, buffer_state) # (B, V)
        return retrieved_value, attention_weights

    def write(self, attention_weights: torch.Tensor, content_projection: torch.Tensor):
        """
        Updates the memory buffer using a batch-averaged EMA-like rule.

        Interpretation of paper's formula B_j,t = (1 - lambda*a_bj)*B_j,t-1 + lambda*a_bj*m_t
        for batch processing: Update buffer B_j based on the average contribution
        from batch items, weighted by their attention a_bj to bin j.
        B_j_new = B_j_old * (1 - avg_j[lambda*a_bj]) + avg_j[lambda*a_bj*m_b]
        """
        B, N = attention_weights.shape
        _N, V = self.buffer.shape
        _B, _V = content_projection.shape

        assert N == _N and B == _B and V == _V, "Shape mismatch during write operation."

        effective_lambda = self.get_learning_rate() # Scalar tensor

        # Calculate batch-averaged update components
        with torch.no_grad(): # Avoid tracking gradients for buffer update calculations if buffer isn't param
            # Average write gate per bin (how much each bin is updated on average)
            avg_write_gate_j = (effective_lambda * attention_weights).mean(dim=0) # Shape (N,)

            # Average content weighted by write gate per bin
            write_signal = effective_lambda * attention_weights # (B, N)
            # einsum 'bn,bv->nv': computes Sum_b[ write_signal_bj * content_b ] for each bin j
            sum_weighted_content_j = torch.einsum('bn,bv->nv', write_signal, content_projection) # (N, V)
            # Average over batch size B
            avg_weighted_content_j = sum_weighted_content_j / B # (N, V)

            # Apply EMA Update
            current_buffer = self.buffer # (N, V)
            forget_mult = (1.0 - avg_write_gate_j).unsqueeze(-1) # (N, 1)
            # Clamp forget multiplier to avoid negative values if lambda or attention is unstable
            forget_mult = torch.clamp(forget_mult, min=0.0)

            new_buffer = forget_mult * current_buffer + avg_weighted_content_j # (N, V)

            # Update buffer state in-place using copy_
            self.buffer.data.copy_(new_buffer)

    def calculate_prototype_regularization(self) -> Optional[torch.Tensor]:
        """ Calculates the prototype diversity regularization loss. """
        if self.learnable_prototypes and self.prototype_reg_coeff > 0.0 and self.training:
            if self.num_bins <= 1: return None

            # Use cosine similarity for regularization penalty
            protos_norm = F.normalize(self.prototypes, p=2, dim=-1) # (N, Dk)
            sim_matrix = torch.matmul(protos_norm, protos_norm.t()) # (N, N)
            # Penalize squared similarity in upper triangle (excluding diagonal)
            upper_tri_sim_sq = torch.triu(sim_matrix, diagonal=1).pow(2)
            num_pairs = self.num_bins * (self.num_bins - 1) / 2
            reg_loss = torch.sum(upper_tri_sim_sq) / max(num_pairs, 1.0) # Avoid div by zero
            return reg_loss * self.prototype_reg_coeff
        return None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a full read-then-write cycle of the i-DAM and returns the
        retrieved value along with any regularization loss (if applicable).

        Args:
            x (torch.Tensor): Input tensor x_t, shape (Batch, input_dim).
            context (torch.Tensor): Context tensor h_{t-1}, shape (Batch, context_dim).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - retrieved_value (r_t): Shape (Batch, value_dim).
                - prototype_reg_loss: Scalar tensor if computed, else None.
        """
        # 1. Prepare combined input for networks
        if x.shape[0] != context.shape[0]:
             raise ValueError(f"Batch size mismatch: x {x.shape[0]}, context {context.shape[0]}")
        # Ensure types match for concatenation
        expected_dtype = self.key_net[0].linear.weight.dtype # Get dtype from first layer
        if x.dtype != expected_dtype: x = x.to(dtype=expected_dtype)
        if context.dtype != expected_dtype: context = context.to(dtype=expected_dtype)
        combined_input = torch.cat([x, context], dim=-1) # (B, input_dim + context_dim)

        # 2. Generate Query Key
        query_key = self.key_net(combined_input) # (B, key_dim)

        # 3. Read from Memory
        retrieved_value, attention_weights = self.read(query_key) # r_t (B, V), a_j (B, N)

        # 4. Generate Content Projection
        content_projection = self.content_net(combined_input) # m_t (B, V)

        # 5. Write to Memory (update buffer state)
        # Only update during training or if explicitly needed during inference (usually not)
        if self.training:
            self.write(attention_weights, content_projection)

        # 6. Calculate optional prototype regularization loss
        reg_loss = self.calculate_prototype_regularization()

        # 7. Return the retrieved value and regularization loss
        return retrieved_value, reg_loss # r_t

    def extra_repr(self) -> str:
        s = (f"key_dim={self.key_dim}, value_dim={self.value_dim}, num_bins={self.num_bins}, "
             f"distance_metric='{self.distance_metric}'\n")
        s += f"  normalize_keys={self.normalize_keys}, normalize_prototypes={self.normalize_prototypes}\n"
        s += f"  learnable_prototypes={self.learnable_prototypes}, prototype_init='{self.prototype_init}'\n"
        # Safely get tensor value for display if not learnable
        temp_val_tensor = self.get_temperature()
        temp_val = f"{temp_val_tensor.item():.4f}" if temp_val_tensor.numel() == 1 else "Tensor"
        temp_status = "(learnable)" if self._is_temp_learnable else "(fixed)"
        lambda_val_tensor = self.get_learning_rate()
        lambda_val = f"{lambda_val_tensor.item():.4f}" if lambda_val_tensor.numel() == 1 else "Tensor"
        lambda_status = "(learnable, constrained)" if self._is_lambda_learnable and self.constrain_lambda else \
                        "(learnable, unconstrained)" if self._is_lambda_learnable else "(fixed)"
        s += f"  temperature(tau)={temp_val} {temp_status}, learning_rate(lambda)={lambda_val} {lambda_status}\n"
        if self.prototype_reg_coeff > 0:
             s += f"  prototype_reg_coeff={self.prototype_reg_coeff}\n"
        s += f"  (key_net): {self.key_net}\n"
        s += f"  (content_net): {self.content_net}"
        return s

# Example Usage (remains the same)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Config
    B = 4          # Batch size
    D_in = 32      # Input dim x_t
    D_ctx = 64     # Context dim h_{t-1}
    D_key = 48     # Key dim
    D_val = 96     # Value dim
    N_bins = 128   # Number of bins

    # Create i-DAM module
    idam_memory = ImplicitDAM(
        input_dim=D_in,
        context_dim=D_ctx,
        key_dim=D_key,
        value_dim=D_val,
        num_bins=N_bins,
        key_net_config={'num_layers': 2, 'hidden_dim': 64},
        content_net_config={'num_layers': 1},
        learnable_prototypes=True,
        prototype_init='randn',
        normalize_keys=True, # Test normalization
        normalize_prototypes=True,
        distance_metric='cosine', # Test cosine
        temperature='learnable', # Learnable temperature
        learning_rate=0.05,    # Fixed learning rate
        prototype_reg_coeff=0.001, # Add some regularization
        device=device,
        dtype=dtype
    )
    print("--- ImplicitDAM Module ---")
    print(idam_memory)

    # Create dummy inputs
    x_t = torch.randn(B, D_in, device=device, dtype=dtype)
    h_prev = torch.randn(B, D_ctx, device=device, dtype=dtype)

    # --- Forward pass (Read-then-Write) ---
    print("\nRunning forward pass (training mode)...")
    # Set to train mode to compute regularization and perform write
    idam_memory.train()
    retrieved_r_t, reg_loss = idam_memory(x_t, h_prev)

    print("Input x shape:", x_t.shape)
    print("Context h shape:", h_prev.shape)
    print("Retrieved r_t shape:", retrieved_r_t.shape)
    assert retrieved_r_t.shape == (B, D_val)
    print("Retrieved sample:", retrieved_r_t[0, :8].detach().cpu().numpy())
    if reg_loss is not None:
        print(f"Prototype Reg Loss: {reg_loss.item():.6f}")
    else:
        print("Prototype Reg Loss: None")


    # Inspect buffer state after write
    print("Buffer shape:", idam_memory.buffer.shape)
    print("Buffer sample (after write):", idam_memory.buffer[0, :8].detach().cpu().numpy())
    buffer_after_train_write = idam_memory.buffer.clone()

    # --- Forward pass (eval mode) ---
    print("\nRunning forward pass (eval mode)...")
    idam_memory.eval()
    retrieved_r_t_eval, reg_loss_eval = idam_memory(x_t, h_prev) # Write should not happen
    print("Retrieved r_t shape (eval):", retrieved_r_t_eval.shape)
    print("Prototype Reg Loss (eval):", reg_loss_eval) # Should be None in eval mode
    assert reg_loss_eval is None
    # Check buffer didn't change
    buffer_after_eval = idam_memory.buffer
    print("Buffer changed after eval forward:", not torch.allclose(buffer_after_train_write, buffer_after_eval))
    assert torch.allclose(buffer_after_train_write, buffer_after_eval)


    # --- Test Read Separately ---
    print("\nTesting read operation separately (eval mode)...")
    idam_memory.eval() # Switch to eval mode
    key_net_input = torch.cat([x_t, h_prev], dim=-1)
    test_key = idam_memory.key_net(key_net_input)
    r_read, a_read = idam_memory.read(test_key)
    print("Test Key shape:", test_key.shape)
    print("Read Retrieved Value shape:", r_read.shape)
    print("Read Attention Weights shape:", a_read.shape)
    assert r_read.shape == (B, D_val)
    assert a_read.shape == (B, N_bins)
    print("Read Attention sample (sum should be 1):", a_read[0, :8].detach().cpu().numpy())
    print("Attention sum check:", a_read.sum(dim=-1).mean().item()) # Should be close to 1


