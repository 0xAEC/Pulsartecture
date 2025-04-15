# ultra_rwka/components/memory/hierarchical.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union, Type, Dict

# Imports from within the library
from ..projections import LinearProjection

# Helper function for outer product (batch-aware)
def batch_outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes outer product for batches of vectors. (B, M), (B, N) -> (B, M, N) """
    return torch.einsum('bi,bj->bij', x, y)

class HierarchicalMemoryStack(nn.Module):
    """
    Implements the Hierarchical Modular Memory Stack from Sec 3.3 of the paper.
    Manages L levels of hidden states (h_l) and memory matrices (M_l),
    evolving them with different timescales and inter-level communication.

    State Update (h): Gated update based on input, context, previous state, and memory.
    Memory Update (M): EMA-like update with outer product write and inter-level communication.
    """
    def __init__(self,
                 num_levels: int,
                 input_dim: int, # Dimension of primary input x_t
                 context_dim: int, # Dimension of context c_t (e.g., from TM-KLA)
                 state_dim: int, # Dimension of hidden state h_l (d_h)
                 memory_rows: int, # Dimension dm_rows for M_l
                 memory_cols: int, # Dimension dm_cols for M_l
                 transition_config: Optional[Dict] = None, # Config for phi_l (e.g., {'type': 'mlp', 'layers': 1, 'hidden': ...})
                 gate_config: Optional[Dict] = None, # Config for alpha/beta gate MLPs
                 write_vec_config: Optional[Dict] = None, # Config for u/v projection MLPs
                 inter_level_comm_config: Optional[Dict] = None, # Config for f (e.g., {'type': 'avg', 'proj_dim': ...})
                 learnable_gamma: bool = True, # Learnable decay/write rates gamma_l
                 learnable_eta: bool = True, # Learnable inter-level strengths eta_l
                 gamma_init: Union[float, List[float]] = 0.1, # Initial gamma value(s)
                 eta_init: Union[float, List[float]] = 0.01, # Initial eta value(s)
                 gamma_constraint: str = 'sigmoid', # 'sigmoid' or 'clamp'
                 eta_constraint: str = 'sigmoid', # 'sigmoid' or 'clamp'
                 share_params_across_levels: bool = False, # Share phi, gates, projections across levels?
                 device=None,
                 dtype=None):
        """
        Args:
            num_levels (int): Number of levels (L).
            input_dim (int): Dimension of input x_t.
            context_dim (int): Dimension of context c_t.
            state_dim (int): Dimension of hidden states h_l.
            memory_rows (int): Number of rows in memory matrices M_l.
            memory_cols (int): Number of columns in memory matrices M_l.
            transition_config (Optional[Dict]): Configuration for state transition function phi_l.
                                                 Defaults to a simple linear layer.
                                                 Example: {'type': 'mlp', 'num_layers': 1, 'hidden_dim': state_dim}
                                                          {'type': 'gru_cell'} (requires state_dim=hidden_dim)
            gate_config (Optional[Dict]): Configuration for gate MLPs (alpha, beta). Defaults to linear.
            write_vec_config (Optional[Dict]): Configuration for MLPs generating u_l, v_l. Defaults to linear.
            inter_level_comm_config (Optional[Dict]): Configuration for inter-level function f.
                                                      Defaults to simple averaging.
                                                      Example: {'type': 'mlp', 'num_layers': 1, 'hidden_dim': ...}
                                                               {'type': 'attention', 'num_heads': ...}
            learnable_gamma (bool): If True, gamma_l rates are learnable parameters. Defaults to True.
            learnable_eta (bool): If True, eta_l strengths are learnable parameters. Defaults to True.
            gamma_init (Union[float, List[float]]): Initial value(s) for gamma. If float, used for all levels.
                                                     If list, must have length num_levels. Defaults to 0.1.
            eta_init (Union[float, List[float]]): Initial value(s) for eta. Defaults to 0.01.
            gamma_constraint (str): How to constrain gamma to [0, 1] if learnable ('sigmoid', 'clamp'). Defaults to 'sigmoid'.
            eta_constraint (str): How to constrain eta to [0, 1] if learnable ('sigmoid', 'clamp'). Defaults to 'sigmoid'.
            share_params_across_levels (bool): If True, use the same nn.Module instance for phi, gates, etc.,
                                               across all levels. Defaults to False.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        if num_levels <= 0: raise ValueError("num_levels must be positive.")
        self.num_levels = num_levels
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.state_dim = state_dim
        self.memory_rows = memory_rows
        self.memory_cols = memory_cols
        self.share_params = share_params_across_levels
        self.gamma_constraint = gamma_constraint
        self.eta_constraint = eta_constraint
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # --- Determine Input Dimensions for Sub-Modules ---
        # Input to gates and transition function phi_l
        # Example: concat(x_t, c_t, h_{l-1}, pooled(M_{l-1}))
        # Let's simplify: Use concat(x_t, c_t, h_{l-1}) for now.
        # Pooling M adds complexity, can be added via config later.
        phi_gate_input_dim = input_dim + context_dim + state_dim

        # Input to u/v projections
        # Example: concat(h_l, x_t, c_t) - Use h_l (current computed state)
        write_vec_input_dim = state_dim + input_dim + context_dim

        # --- Instantiate Level-Specific or Shared Modules ---
        num_instances = 1 if share_params_across_levels else num_levels

        # State Transition Functions (phi_l)
        self.phi_transitions = nn.ModuleList([
            self._create_transition_module(phi_gate_input_dim, state_dim, transition_config)
            for _ in range(num_instances)
        ])

        # Gate Networks (alpha and beta gates)
        gate_cfg = gate_config or {}
        self.alpha_gates = nn.ModuleList([
             self._create_gate_mlp(phi_gate_input_dim, state_dim, **gate_cfg)
             for _ in range(num_instances)
        ])
        self.beta_gates = nn.ModuleList([
             self._create_gate_mlp(phi_gate_input_dim, state_dim, **gate_cfg)
             for _ in range(num_instances)
        ])

        # Write Vector Projections (u_l and v_l)
        write_cfg = write_vec_config or {}
        self.u_projs = nn.ModuleList([
            self._create_write_vec_mlp(write_vec_input_dim, memory_rows, **write_cfg)
            for _ in range(num_instances)
        ])
        self.v_projs = nn.ModuleList([
            self._create_write_vec_mlp(write_vec_input_dim, memory_cols, **write_cfg)
            for _ in range(num_instances)
        ])

        # Inter-Level Communication Function (f)
        # This function needs to combine information from adjacent memory levels.
        # Input might be M_{l-1}, M_{l+1}. Output should match M_l shape.
        self.inter_level_comm = self._create_inter_level_comm_module(
            memory_rows, memory_cols, inter_level_comm_config
        )

        # --- Learnable Rates/Strengths (gamma_l, eta_l) ---
        self.gammas = self._initialize_level_params(
            num_levels, learnable_gamma, gamma_init, 'gamma', gamma_constraint)
        self.etas = self._initialize_level_params(
            num_levels, learnable_eta, eta_init, 'eta', eta_constraint)

    # --- Helper methods for creating sub-modules ---
    def _create_transition_module(self, in_dim, out_dim, config: Optional[Dict]) -> nn.Module:
        cfg = config or {}
        m_type = cfg.get('type', 'mlp').lower()
        if m_type == 'mlp':
            return self._create_generic_mlp(in_dim, out_dim, cfg)
        elif m_type == 'gru_cell':
            if in_dim != out_dim: # GRUCell expects input_size == hidden_size typically
                warnings.warn(f"GRUCell transition expects input_dim==state_dim ({in_dim}!={out_dim}). Using MLP fallback.")
                return self._create_generic_mlp(in_dim, out_dim, cfg)
            return nn.GRUCell(in_dim, out_dim, **self.factory_kwargs)
        # Add LSTMCell etc. if needed
        else:
            raise ValueError(f"Unsupported transition type: {m_type}")

    def _create_gate_mlp(self, in_dim, out_dim, **config) -> nn.Sequential:
        # Gates typically use simple MLPs
        return self._create_generic_mlp(in_dim, out_dim, config, default_layers=1)

    def _create_write_vec_mlp(self, in_dim, out_dim, **config) -> nn.Sequential:
        # Projections for u, v are often simple linear layers
        return self._create_generic_mlp(in_dim, out_dim, config, default_layers=1)

    def _create_generic_mlp(self, in_dim, out_dim, config: Dict, default_layers=1) -> nn.Sequential:
        """ Creates an MLP based on config dict """
        num_layers = config.get('num_layers', default_layers)
        hidden_dim = config.get('hidden_dim') # Defaults handled in LinearProjection helper potentially
        activation_cls = config.get('activation_cls', nn.GELU)
        proj_kwargs = {k:v for k,v in config.items() if k in ['use_bias', 'initialize']}
        proj_kwargs.update(self.factory_kwargs)

        _hidden_dim = hidden_dim if hidden_dim is not None else max(in_dim // 2, out_dim)
        layers = []
        current_dim = in_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            layer_out_dim = out_dim if is_last else _hidden_dim
            layers.append(LinearProjection(
                current_dim, layer_out_dim,
                activation_cls=None if is_last else activation_cls,
                initialize=proj_kwargs.get('initialize', 'xavier_uniform'),
                use_bias=proj_kwargs.get('use_bias', True),
                device=proj_kwargs.get('device'),
                dtype=proj_kwargs.get('dtype')
            ))
            current_dim = layer_out_dim
        return nn.Sequential(*layers)

    def _create_inter_level_comm_module(self, rows, cols, config: Optional[Dict]) -> nn.Module:
        cfg = config or {}
        m_type = cfg.get('type', 'avg').lower()
        # Output must be shape (B, rows, cols)
        if m_type == 'avg':
            # Simple averaging needs no parameters
            return lambda m_prev, m_next: 0.5 * (m_prev + m_next)
        elif m_type == 'mlp':
            # MLP processes concatenated flattened memories? Or applies convolution?
            # Let's assume simple MLP on flattened average for now
            hidden_dim = cfg.get('hidden_dim', rows * cols)
            num_layers = cfg.get('num_layers', 1)
            # Input: concatenated flattened M_{l-1}, M_{l+1} -> 2*rows*cols ?
            # Or average first: rows*cols
            in_dim = rows * cols
            network = self._create_generic_mlp(in_dim, rows * cols, {'num_layers': num_layers, 'hidden_dim': hidden_dim})
            # Need a wrapper to handle averaging, flattening, reshaping
            class MLPComm(nn.Module):
                def __init__(self, net, r, c):
                    super().__init__()
                    self.net = net
                    self.r = r
                    self.c = c
                def forward(self, m_prev, m_next):
                    avg_mem = 0.5 * (m_prev + m_next) # (B, R, C)
                    B = avg_mem.shape[0]
                    flat_avg = avg_mem.view(B, -1) # (B, R*C)
                    processed_flat = self.net(flat_avg) # (B, R*C)
                    return processed_flat.view(B, self.r, self.c) # (B, R, C)
            return MLPComm(network, rows, cols)
        # Add 'attention' type later if needed
        else:
            raise ValueError(f"Unsupported inter_level_comm type: {m_type}")

    def _initialize_level_params(self, num_levels, learnable, init_val, name, constraint):
        """ Helper to initialize gamma or eta parameters/buffers. """
        if isinstance(init_val, list):
            if len(init_val) != num_levels:
                raise ValueError(f"Length of {name}_init list must match num_levels ({num_levels})")
            init_tensor = torch.tensor(init_val, **self.factory_kwargs).view(num_levels, 1) # Shape (L, 1) for broadcasting
        else:
            init_tensor = torch.full((num_levels, 1), float(init_val), **self.factory_kwargs)

        # Apply inverse transform if learnable and constrained
        if learnable:
            if constraint == 'sigmoid':
                 clamped_init = torch.clamp(init_tensor, min=1e-6, max=1.0 - 1e-6)
                 init_param_val = torch.log(clamped_init / (1.0 - clamped_init)) # Inverse sigmoid
            elif constraint == 'clamp':
                 init_param_val = init_tensor # Store raw value, clamp in forward
            else: # None or other
                 init_param_val = init_tensor
            return nn.Parameter(init_param_val) # Shape (L, 1)
        else:
            # Fixed values stored as buffers
            if constraint == 'sigmoid' or constraint == 'clamp':
                 # Ensure fixed values are valid
                 if not torch.all((init_tensor >= 0.0) & (init_tensor <= 1.0)):
                      raise ValueError(f"Fixed {name} values must be in [0, 1] if constraint is '{constraint}'")
            self.register_buffer(f"{name}_val", init_tensor)
            return None # Indicate it's stored in buffer

    def _get_level_params(self, level_idx, params_or_buffer_name, constraint):
        """ Helper to get constrained gamma or eta for a specific level. """
        if isinstance(params_or_buffer_name, nn.Parameter): # Learnable
            param = params_or_buffer_name[level_idx] # Get value for level l
            if constraint == 'sigmoid':
                return torch.sigmoid(param)
            elif constraint == 'clamp':
                return torch.clamp(param, min=0.0, max=1.0)
            else: # No constraint
                return param
        else: # Fixed (stored in buffer)
            buffer = getattr(self, params_or_buffer_name)
            return buffer[level_idx]

    def forward(self,
                x_t: torch.Tensor,
                c_t: torch.Tensor,
                prev_states: Union[List[torch.Tensor], torch.Tensor], # List[ (B, Dh) ] or (L, B, Dh)
                prev_memories: Union[List[torch.Tensor], torch.Tensor] # List[ (B, Dr, Dc) ] or (L, B, Dr, Dc)
               ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Performs one time step update for all levels.

        Args:
            x_t (torch.Tensor): Current primary input, shape (Batch, input_dim).
            c_t (torch.Tensor): Current context input, shape (Batch, context_dim).
            prev_states (Union[List[torch.Tensor], torch.Tensor]): Hidden states from previous step (h_{t-1}).
            prev_memories (Union[List[torch.Tensor], torch.Tensor]): Memory matrices from previous step (M_{t-1}).

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - new_states (List[torch.Tensor]): Updated hidden states h_t, list of L tensors (B, state_dim).
                - new_memories (List[torch.Tensor]): Updated memory matrices M_t, list of L tensors (B, memory_rows, memory_cols).
        """
        # --- Input Handling and Type Conversion ---
        # Ensure inputs match factory kwargs if possible
        expected_dtype = next(self.parameters()).dtype
        x_t = x_t.to(dtype=expected_dtype) if x_t.dtype != expected_dtype else x_t
        c_t = c_t.to(dtype=expected_dtype) if c_t.dtype != expected_dtype else c_t

        # Convert input lists/tensors to lists for easier iteration
        if torch.is_tensor(prev_states):
            # Input (L, B, D) -> List of L tensors (B, D)
            prev_states_list = [s.squeeze(0) for s in torch.split(prev_states, 1, dim=0)]
        elif isinstance(prev_states, list):
            prev_states_list = prev_states
        else:
            raise TypeError("prev_states must be a Tensor or List[Tensor]")

        if torch.is_tensor(prev_memories):
             # Input (L, B, R, C) -> List of L tensors (B, R, C)
             prev_memories_list = [m.squeeze(0) for m in torch.split(prev_memories, 1, dim=0)]
        elif isinstance(prev_memories, list):
             prev_memories_list = prev_memories
        else:
             raise TypeError("prev_memories must be a Tensor or List[Tensor]")

        if len(prev_states_list) != self.num_levels or len(prev_memories_list) != self.num_levels:
             raise ValueError("Length of prev_states/prev_memories must match num_levels")

        # --- Initialization for results ---
        new_states = [None] * self.num_levels
        new_memories = [None] * self.num_levels
        B = x_t.shape[0] # Batch size

        # --- Iterate through levels ---
        # We compute all new states h_t first, then all new memories M_t
        # This simplifies dependencies for inter-level communication if needed,
        # although current implementation uses M_{t-1} for communication.

        # 1. Compute New States h_t
        for l in range(self.num_levels):
            idx = 0 if self.share_params else l
            h_prev_l = prev_states_list[l].to(dtype=expected_dtype) # (B, Dh)
            # M_prev_l = prev_memories_list[l] # (B, Dr, Dc) - Not used in simplified phi input

            # Prepare input for gates and phi
            # Simplified: concat(x_t, c_t, h_{l-1})
            phi_gate_input = torch.cat([x_t, c_t, h_prev_l], dim=-1) # (B, Din+Dctx+Dh)

            # Compute gates
            alpha = torch.sigmoid(self.alpha_gates[idx](phi_gate_input)) # (B, Dh)
            beta = torch.sigmoid(self.beta_gates[idx](phi_gate_input))   # (B, Dh)

            # Compute candidate state update using phi_l
            phi_module = self.phi_transitions[idx]
            if isinstance(phi_module, nn.GRUCell):
                 # GRUCell expects (input, hidden)
                 h_candidate = phi_module(phi_gate_input, h_prev_l) # Assumes input_dim=state_dim
            elif isinstance(phi_module, nn.Sequential): # Assuming MLP
                 h_candidate = phi_module(phi_gate_input)
            else:
                 # Add other transition types if needed
                 raise TypeError(f"Unsupported transition module type: {type(phi_module)}")

            # Apply gated update: h_t = alpha * h_{t-1} + beta * h_candidate
            h_new_l = alpha * h_prev_l + beta * h_candidate
            new_states[l] = h_new_l

        # 2. Compute New Memories M_t
        for l in range(self.num_levels):
            idx = 0 if self.share_params else l
            h_new_l = new_states[l] # Use the *updated* state h_t for write vectors
            M_prev_l = prev_memories_list[l].to(dtype=expected_dtype) # (B, Dr, Dc)

            # Prepare input for u/v projections: concat(h_t, x_t, c_t)
            write_vec_input = torch.cat([h_new_l, x_t, c_t], dim=-1) # (B, Dh+Din+Dctx)

            # Compute write vectors u, v
            u_l = self.u_projs[idx](write_vec_input) # (B, Dr)
            v_l = self.v_projs[idx](write_vec_input) # (B, Dc)

            # Compute outer product write update
            write_update = batch_outer_product(u_l, v_l) # (B, Dr, Dc)

            # Compute inter-level communication term f(M_{t-1}^{l-1}, M_{t-1}^{l+1})
            # Handle boundary conditions
            M_prev_l_minus_1 = prev_memories_list[l-1] if l > 0 else torch.zeros_like(M_prev_l)
            M_prev_l_plus_1 = prev_memories_list[l+1] if l < self.num_levels - 1 else torch.zeros_like(M_prev_l)
            # Ensure dtype match for communication function
            M_prev_l_minus_1 = M_prev_l_minus_1.to(dtype=expected_dtype)
            M_prev_l_plus_1 = M_prev_l_plus_1.to(dtype=expected_dtype)
            comm_term = self.inter_level_comm(M_prev_l_minus_1, M_prev_l_plus_1) # (B, Dr, Dc)

            # Get constrained gamma_l and eta_l (scalar tensors)
            gamma_l = self._get_level_params(l, self.gammas if self.gammas is not None else 'gamma_val', self.gamma_constraint)
            eta_l = self._get_level_params(l, self.etas if self.etas is not None else 'eta_val', self.eta_constraint)

            # Ensure rates are correct dtype and shape for broadcasting (B, 1, 1)
            gamma_l = gamma_l.view(1, 1, 1).to(dtype=expected_dtype) # Expand for broadcasting
            eta_l = eta_l.view(1, 1, 1).to(dtype=expected_dtype)     # Expand for broadcasting

            # Apply memory evolution update rule
            # M_t = (1 - gamma) * M_{t-1} + gamma * outer(u,v) + eta * f(...)
            M_new_l = (1.0 - gamma_l) * M_prev_l + gamma_l * write_update + eta_l * comm_term
            new_memories[l] = M_new_l

        return new_states, new_memories

    def extra_repr(self) -> str:
        s = f"num_levels={self.num_levels}, state_dim={self.state_dim}, "
        s += f"memory_shape=({self.memory_rows}, {self.memory_cols})\n"
        s += f"  share_params_across_levels={self.share_params}\n"
        # Show first level's modules as representative if shared
        idx = 0
        s += f"  (phi_transition): {self.phi_transitions[idx]}\n"
        s += f"  (alpha_gate): {self.alpha_gates[idx]}\n"
        s += f"  (beta_gate): {self.beta_gates[idx]}\n"
        s += f"  (u_proj): {self.u_projs[idx]}\n"
        s += f"  (v_proj): {self.v_projs[idx]}\n"
        s += f"  (inter_level_comm): {self.inter_level_comm}"
        # Could add gamma/eta details too
        return s.strip()

# Example Usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Config
    B = 4
    L = 3 # Num levels
    D_in = 32
    D_ctx = 64
    D_h = 128 # State dim
    D_r, D_c = 16, 8 # Memory matrix dims

    # Create the stack
    hms = HierarchicalMemoryStack(
        num_levels=L,
        input_dim=D_in,
        context_dim=D_ctx,
        state_dim=D_h,
        memory_rows=D_r,
        memory_cols=D_c,
        transition_config={'type': 'mlp', 'num_layers': 1}, # Simple MLP transition
        inter_level_comm_config={'type': 'avg'}, # Simple averaging communication
        learnable_gamma=True,
        learnable_eta=True,
        gamma_init=[0.1, 0.05, 0.02], # Example: Slower decay at higher levels
        eta_init=0.01,
        device=device,
        dtype=dtype
    )
    print("--- HierarchicalMemoryStack ---")
    print(hms)

    # Create dummy inputs for one time step
    x_in = torch.randn(B, D_in, device=device, dtype=dtype)
    c_in = torch.randn(B, D_ctx, device=device, dtype=dtype)
    # Initial states/memories (usually zeros)
    prev_h = [torch.zeros(B, D_h, device=device, dtype=dtype) for _ in range(L)]
    prev_M = [torch.zeros(B, D_r, D_c, device=device, dtype=dtype) for _ in range(L)]

    # --- Forward pass for one step ---
    print("\nRunning forward pass (one step)...")
    hms.train() # Enable training mode
    new_h, new_M = hms(x_in, c_in, prev_h, prev_M)

    print("Input x shape:", x_in.shape)
    print("Input c shape:", c_in.shape)
    print("Num output states:", len(new_h))
    print("Output state shape (level 0):", new_h[0].shape)
    assert new_h[0].shape == (B, D_h)
    print("Num output memories:", len(new_M))
    print("Output memory shape (level 0):", new_M[0].shape)
    assert new_M[0].shape == (B, D_r, D_c)

    print("\nSample output state (level 0, item 0):", new_h[0][0, :8].detach().cpu().numpy())
    print("Sample output memory (level 0, item 0):", new_M[0][0, 0, :4].detach().cpu().numpy())

    # Check if state/memory changed from zeros
    print("State changed:", not torch.allclose(prev_h[0], new_h[0]))
    print("Memory changed:", not torch.allclose(prev_M[0], new_M[0]))


