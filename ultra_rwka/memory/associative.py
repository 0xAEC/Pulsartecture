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
    """ Performs safe division: num / (den + eps), ensuring den is non-negative. """
    # Clamp denominator to be slightly positive to avoid NaN gradients with den=0
    return num / den.clamp(min=eps)

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
    5. Update Buffer (B_j) using an EMA-like rule based on addressing weights and projected content (only during training).
    6. Optionally compute prototype diversity regularization loss.
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

        # --- Input Validation ---
        if self.distance_metric not in self._supported_distance_metrics:
            raise ValueError(f"Unsupported distance_metric: {distance_metric}. Choose from {self._supported_distance_metrics}")
        if self.normalize_keys and self.distance_metric == 'euclidean':
            warnings.warn("Normalizing keys with Euclidean distance might have unintended effects on distance scaling.")
        if self.normalize_prototypes and self.distance_metric == 'euclidean':
            warnings.warn("Normalizing prototypes with Euclidean distance might have unintended effects on distance scaling.")
        if num_bins <= 0:
            raise ValueError("num_bins must be positive.")

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
        # Register as buffer - its state is updated manually in `write`
        self.register_buffer('buffer', buffer_tensor) # Shape: (N_b, value_dim)

        # --- Temperature (tau) ---
        if self._is_temp_learnable:
            # Initialize parameter before softplus such that softplus(param) ~= temperature_init
            init_val = math.log(math.exp(max(temperature_init, 1e-6)) - 1.0)
            self.temperature_param = nn.Parameter(torch.tensor(init_val, **self.factory_kwargs))
        else:
            if not isinstance(temperature, (float, int)) or temperature <= 0:
                 raise ValueError("Fixed temperature must be a positive float.")
            # Store fixed value directly as buffer
            self.register_buffer('temperature_val', torch.tensor(float(temperature), **self.factory_kwargs))

        # --- Learning Rate (lambda) ---
        if self._is_lambda_learnable:
             # Initialize parameter before sigmoid such that sigmoid(param) ~= learning_rate_init
             clamped_init = max(min(learning_rate_init, 1.0 - 1e-6), 1e-6) # Clamp init to (eps, 1-eps)
             init_val = math.log(clamped_init / (1.0 - clamped_init))
             self.learning_rate_param = nn.Parameter(torch.tensor(init_val, **self.factory_kwargs))
        else:
             if not isinstance(learning_rate, (float, int)) or not (0.0 <= learning_rate <= 1.0):
                 raise ValueError("Fixed learning_rate must be a float in [0, 1]")
             # Store fixed value directly as buffer
             self.register_buffer('learning_rate_val', torch.tensor(float(learning_rate), **self.factory_kwargs))

    def get_temperature(self) -> torch.Tensor:
        """ Returns the current temperature value (positive scalar tensor). """
        if self._is_temp_learnable:
            # Softplus ensures positivity. Add epsilon for safety.
            return F.softplus(self.temperature_param) + 1e-6
        else:
            # Return buffer value directly
            return self.temperature_val

    def get_learning_rate(self) -> torch.Tensor:
        """ Returns the current learning rate value (scalar tensor in [0, 1] if constrained). """
        if self._is_lambda_learnable:
            if self.constrain_lambda:
                # Sigmoid keeps lambda in (0, 1)
                return torch.sigmoid(self.learning_rate_param)
            else:
                # Return raw parameter if constraint is disabled (user must ensure validity)
                return self.learning_rate_param
        else:
            # Return buffer value directly
            return self.learning_rate_val

    def _create_mlp(self, in_dim: int, out_dim: int, num_layers: int = 1, hidden_dim: Optional[int] = None, activation_cls: Type[nn.Module] = nn.GELU, **kwargs) -> nn.Sequential:
        """ Helper to create simple MLPs using LinearProjection. """
        _hidden_dim = hidden_dim if hidden_dim is not None else max(in_dim // 2, out_dim)
        layers = []
        current_dim = in_dim
        # Get relevant kwargs for LinearProjection from the config dict
        proj_kwargs = {k: v for k, v in kwargs.items() if k in ['use_bias', 'initialize']}
        proj_kwargs.update(self.factory_kwargs) # Add device/dtype

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            layer_out_dim = out_dim if is_last else _hidden_dim
            layers.append(LinearProjection(
                current_dim, layer_out_dim,
                activation_cls=None if is_last else activation_cls, # No activation on final layer
                initialize=proj_kwargs.get('initialize', 'xavier_uniform'), # Default init
                use_bias=proj_kwargs.get('use_bias', True),
                device=proj_kwargs.get('device'),
                dtype=proj_kwargs.get('dtype')
            ))
            current_dim = layer_out_dim
        return nn.Sequential(*layers)

    def _initialize_prototypes(self, method: str) -> torch.Tensor:
        """ Initialize prototype vectors phi_j. """
        if method == 'randn':
            tensor = torch.empty(self.num_bins, self.key_dim, **self.factory_kwargs)
            # Initialize with small std deviation, similar to nn.Linear default bias init range
            std = 1.0 / math.sqrt(self.key_dim)
            nn.init.normal_(tensor, mean=0.0, std=std)
        elif method == 'uniform':
            tensor = torch.empty(self.num_bins, self.key_dim, **self.factory_kwargs)
            bound = 1.0 / math.sqrt(self.key_dim)
            nn.init.uniform_(tensor, -bound, bound)
        else:
            raise ValueError(f"Unknown prototype_init method: {method}. Choose 'randn' or 'uniform'.")
        return tensor

    def _compute_distances(self, query_key: torch.Tensor) -> torch.Tensor:
        """ Computes distances between query keys and prototypes. """
        # Normalize if requested
        q = F.normalize(query_key, p=2, dim=-1) if self.normalize_keys else query_key
        # Detach prototypes during normalization if they are learnable but normalization shouldn't affect their gradients directly
        p_maybe_detached = self.prototypes.detach() if self.learnable_prototypes else self.prototypes
        p_normalized = F.normalize(p_maybe_detached, p=2, dim=-1)
        p = p_normalized if self.normalize_prototypes else self.prototypes # Use normalized or original based on flag

        if self.distance_metric == 'euclidean':
            # Squared Euclidean distance using torch.cdist
            distances = torch.cdist(q, p, p=2).pow(2) # (B, N)
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            # Ensure inputs are normalized if using cosine distance
            if not (self.normalize_keys and self.normalize_prototypes):
                 warnings.warn("Using cosine distance without normalizing keys and prototypes.")
            sim = torch.matmul(q, p.t()) # (B, Dk) @ (Dk, N) -> (B, N)
            distances = 1.0 - sim
        else:
            # Should be caught by __init__ but check defensively
            raise NotImplementedError(f"Distance metric {self.distance_metric} not implemented.")

        # Clamp minimum distance for stability, esp. for Euclidean
        distances = torch.clamp(distances, min=0.0)
        return distances # Shape: (B, N)

    def compute_addressing(self, query_key: torch.Tensor) -> torch.Tensor:
        """ Computes addressing weights using Kernel Softmax. """
        distances = self._compute_distances(query_key) # (B, N)
        temperature = self.get_temperature() # Scalar tensor

        # Kernel Softmax: exp(-distance / tau) / sum(exp(-distance / tau))
        scaled_neg_distances = -distances / temperature # (B, N)

        # Use log-softmax for numerical stability
        log_probs = F.log_softmax(scaled_neg_distances, dim=-1) # (B, N)
        attention_weights = torch.exp(log_probs) # (B, N)
        # Ensure weights sum to 1 (within float precision)
        # attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return attention_weights # a_jmem

    def read(self, query_key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Reads from the memory buffer using the query key. """
        attention_weights = self.compute_addressing(query_key) # (B, N)
        buffer_state = self.buffer # (N, V)
        # Ensure buffer dtype matches attention weights if needed (e.g., for AMP)
        if buffer_state.dtype != attention_weights.dtype:
             buffer_state = buffer_state.to(attention_weights.dtype)

        # einsum: 'bn,nv->bv' (batch, bins) @ (bins, value_dim) -> (batch, value_dim)
        retrieved_value = torch.einsum('bn,nv->bv', attention_weights, buffer_state) # (B, V)
        return retrieved_value, attention_weights

    def write(self, attention_weights: torch.Tensor, content_projection: torch.Tensor):
        """
        Updates the memory buffer using a batch-averaged EMA-like rule.
        This interprets the paper's formula for batch processing by averaging
        the update contribution across the batch for each memory bin.
        Update occurs in-place on self.buffer.
        """
        B, N = attention_weights.shape
        _N, V = self.buffer.shape
        _B, _V = content_projection.shape

        assert N == _N and B == _B and V == _V, "Shape mismatch during write operation."

        effective_lambda = self.get_learning_rate() # Scalar tensor

        # --- Batch-Averaged Update Calculation ---
        # This calculation should not be part of the autograd graph w.r.t the buffer state itself,
        # only w.r.t parameters used to compute attention_weights and content_projection.
        with torch.no_grad():
            # Ensure calculation happens on the correct device/dtype
            current_buffer = self.buffer.to(device=attention_weights.device, dtype=attention_weights.dtype)
            content = content_projection.to(dtype=current_buffer.dtype)
            attn = attention_weights.to(dtype=current_buffer.dtype)
            eff_lambda = effective_lambda.to(dtype=current_buffer.dtype)

            # Average write gate per bin
            avg_write_gate_j = (eff_lambda * attn).mean(dim=0) # Shape (N,)

            # Average content weighted by write gate per bin
            write_signal = eff_lambda * attn # (B, N)
            sum_weighted_content_j = torch.einsum('bn,bv->nv', write_signal, content) # (N, V)
            # Average over batch size B
            avg_weighted_content_j = sum_weighted_content_j / B # (N, V)

            # Apply EMA Update
            forget_mult = (1.0 - avg_write_gate_j).unsqueeze(-1).clamp_(min=0.0) # (N, 1), clamp for stability
            write_mult = avg_weighted_content_j # (N, V)

            new_buffer = forget_mult * current_buffer + write_mult # (N, V)

            # Update buffer state in-place using copy_
            self.buffer.data.copy_(new_buffer)

    def calculate_prototype_regularization(self) -> Optional[torch.Tensor]:
        """ Calculates the prototype diversity regularization loss. """
        # Only compute during training if enabled and prototypes are learnable
        if not (self.learnable_prototypes and self.prototype_reg_coeff > 0.0 and self.training):
            return None
        if self.num_bins <= 1:
            return None # No pairs to compare

        # Use cosine similarity for regularization penalty
        protos = self.prototypes # (N, Dk)
        protos_norm = F.normalize(protos, p=2, dim=-1)
        sim_matrix = torch.matmul(protos_norm, protos_norm.t()) # (N, N)

        # Penalize squared similarity in upper triangle (excluding diagonal)
        upper_tri_sim_sq = torch.triu(sim_matrix, diagonal=1).pow(2)
        num_pairs = self.num_bins * (self.num_bins - 1) / 2
        # Use safe divide in case num_bins=1 (though checked above)
        avg_sim_sq = torch.sum(upper_tri_sim_sq) / max(num_pairs, 1.0)

        reg_loss = avg_sim_sq * self.prototype_reg_coeff
        return reg_loss

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
        # Ensure types match for concatenation and networks
        expected_dtype = next(self.key_net.parameters()).dtype # Get dtype from key_net
        if x.dtype != expected_dtype: x = x.to(dtype=expected_dtype)
        if context.dtype != expected_dtype: context = context.to(dtype=expected_dtype)
        combined_input = torch.cat([x, context], dim=-1) # (B, input_dim + context_dim)

        # 2. Generate Query Key
        query_key = self.key_net(combined_input) # (B, key_dim)

        # 3. Read from Memory
        # Ensure query key dtype matches prototype dtype for distance calc
        if query_key.dtype != self.prototypes.dtype:
             query_key = query_key.to(dtype=self.prototypes.dtype)
        retrieved_value, attention_weights = self.read(query_key) # r_t (B, V), a_j (B, N)

        # 4. Generate Content Projection
        content_projection = self.content_net(combined_input) # m_t (B, V)

        # 5. Write to Memory (update buffer state)
        # Only update during training by default
        if self.training:
            # Ensure dtypes match for write operation
            if attention_weights.dtype != self.buffer.dtype:
                attention_weights = attention_weights.to(dtype=self.buffer.dtype)
            if content_projection.dtype != self.buffer.dtype:
                content_projection = content_projection.to(dtype=self.buffer.dtype)
            self.write(attention_weights, content_projection)

        # 6. Calculate optional prototype regularization loss
        reg_loss = self.calculate_prototype_regularization()

        # 7. Return the retrieved value (ensure original dtype) and regularization loss
        return retrieved_value.to(dtype=x.dtype), reg_loss # r_t

    def extra_repr(self) -> str:
        s = (f"key_dim={self.key_dim}, value_dim={self.value_dim}, num_bins={self.num_bins}, "
             f"distance_metric='{self.distance_metric}'\n")
        s += f"  normalize_keys={self.normalize_keys}, normalize_prototypes={self.normalize_prototypes}\n"
        s += f"  learnable_prototypes={self.learnable_prototypes}, prototype_init='{self.prototype_init}'\n"
        # Get scalar value for display if possible
        try:
            temp_val_tensor = self.get_temperature()
            temp_val = f"{temp_val_tensor.item():.4f}" if temp_val_tensor.numel() == 1 else "Tensor"
        except: temp_val = "Error" # Handle case where param might not be initialized yet
        temp_status = "(learnable)" if self._is_temp_learnable else "(fixed)"

        try:
            lambda_val_tensor = self.get_learning_rate()
            lambda_val = f"{lambda_val_tensor.item():.4f}" if lambda_val_tensor.numel() == 1 else "Tensor"
        except: lambda_val = "Error"
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
        key_net_config={'num_layers': 2, 'hidden_dim': 64, 'activation_cls': nn.GELU}, # Pass activation
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

    # --- Forward pass (training mode) ---
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
    print("Buffer changed after eval forward:", not torch.allclose(buffer_after_train_write, buffer_after_eval, atol=1e-6)) # Use tolerance for float comparison
    assert torch.allclose(buffer_after_train_write, buffer_after_eval, atol=1e-6)


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

