# ultra_rwka/components/positional/learned_fourier.py

import torch
import torch.nn as nn
import math
from typing import Optional, Union

# Import LinearProjection if the final projection is used
from ..projections import LinearProjection

class LearnedSinFourier(nn.Module):
    """
    Generates positional encodings using a learnable Fourier synthesis approach.
    Based on the formula in Sec 3.6:
        p_t = Proj_P( sum_{k=1}^{K} rho_k * sin(omega_k * t + phi_k) )
    where rho (amplitude), omega (frequency), and phi (phase) are learnable parameters.

    Note: The implementation outputs the vector of K components before the optional
    final projection, consistent with how positional encodings are often used (concatenated
    or added dimension-wise). The 'sum' in the paper's formula might refer to the
    overall synthesis process rather than summing the K components into a scalar.
    """
    def __init__(self,
                 output_dim: int,
                 num_components: int, # K in the paper's formula
                 max_timescale: float = 10000.0, # Used for frequency initialization range
                 learnable_freq: bool = True,
                 learnable_amp_phase: bool = True,
                 use_projection: bool = True, # Whether to use the final Proj_P
                 proj_bias: bool = True,
                 proj_init: str = 'xavier_uniform',
                 device=None,
                 dtype=None):
        """
        Args:
            output_dim (int): The final output dimension (d_p) of the positional encoding.
            num_components (int): Number of sine wave components (K).
            max_timescale (float): Helps define the range for frequency initialization.
                                   Frequencies will range roughly from 1/max_timescale to 1.
                                   Defaults to 10000.0 (similar to Transformer PE).
            learnable_freq (bool): If True, frequencies (omega_k) are learnable parameters.
                                   If False, they are fixed buffers initialized logarithmically.
                                   Defaults to True.
            learnable_amp_phase (bool): If True, amplitudes (rho_k) and phases (phi_k) are
                                        learnable parameters. If False, they are fixed buffers.
                                        Defaults to True.
            use_projection (bool): If True, applies a final LinearProjection (Proj_P) to map
                                   the K components to output_dim. If False, num_components
                                   must equal output_dim. Defaults to True.
            proj_bias (bool): Whether the final projection layer uses bias. Defaults to True.
            proj_init (str): Initialization scheme for the final projection. Defaults to 'xavier_uniform'.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_components = num_components # K
        self.learnable_freq = learnable_freq
        self.learnable_amp_phase = learnable_amp_phase
        self.use_projection = use_projection
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        if not use_projection and num_components != output_dim:
            raise ValueError(f"If use_projection is False, num_components ({num_components}) "
                             f"must equal output_dim ({output_dim}).")

        # --- Initialize Parameters or Buffers ---

        # Frequencies (omega_k)
        log_min_freq = math.log(1.0 / max_timescale)
        log_max_freq = math.log(1.0) # Frequency = 1/period
        # Initialize frequencies logarithmically spaced in the log domain
        log_freq_init = torch.linspace(log_max_freq, log_min_freq, num_components, **self.factory_kwargs)
        freq_init = torch.exp(log_freq_init)
        if learnable_freq:
            self.omega = nn.Parameter(freq_init) # Shape: (K,)
        else:
            self.register_buffer('omega', freq_init)

        # Amplitudes (rho_k)
        # Initialize amplitudes, e.g., to 1 or small random values
        amp_init = torch.ones(num_components, **self.factory_kwargs)
        # amp_init = torch.randn(num_components, **self.factory_kwargs) * 0.1 + 1.0
        if learnable_amp_phase:
            self.rho = nn.Parameter(amp_init) # Shape: (K,)
        else:
            self.register_buffer('rho', amp_init)

        # Phases (phi_k)
        # Initialize phases, e.g., to zeros or random values in [0, 2*pi]
        phase_init = torch.zeros(num_components, **self.factory_kwargs)
        # phase_init = torch.rand(num_components, **self.factory_kwargs) * 2 * math.pi
        if learnable_amp_phase:
            self.phi = nn.Parameter(phase_init) # Shape: (K,)
        else:
            self.register_buffer('phi', phase_init)

        # --- Optional Final Projection (Proj_P) ---
        self.projection: Optional[LinearProjection] = None
        if use_projection:
            self.projection = LinearProjection(
                in_dim=num_components,
                out_dim=output_dim,
                use_bias=proj_bias,
                initialize=proj_init,
                **self.factory_kwargs
            )

    def forward(self, seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Generate the learned sinusoidal positional encoding for a given sequence length.

        Args:
            seq_len (int): The length of the sequence (T).
            device (Optional[torch.device]): Device to create the encoding on. If None, uses parameter device.
            dtype (Optional[torch.dtype]): Dtype for the encoding. If None, uses parameter dtype.

        Returns:
            torch.Tensor: Positional encoding tensor of shape (seq_len, output_dim).
        """
        # Determine device and dtype from parameters/buffers if not provided
        ref_param = self.omega # Use omega as reference for device/dtype
        _device = device if device is not None else ref_param.device
        _dtype = dtype if dtype is not None else ref_param.dtype

        # Create time steps tensor t = [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=_device, dtype=_dtype) # Shape: (T,)

        # Prepare shapes for broadcasting:
        # t: (T, 1)
        # omega, phi, rho: (1, K)
        t_reshaped = t.unsqueeze(-1)        # (T, 1)
        omega = self.omega.unsqueeze(0)     # (1, K)
        phi = self.phi.unsqueeze(0)         # (1, K)
        rho = self.rho.unsqueeze(0)         # (1, K)

        # Calculate weighted sine waves: rho * sin(omega * t + phi)
        # omega * t -> (T, K)
        # + phi -> (T, K)
        # sin(...) -> (T, K)
        # rho * ... -> (T, K)
        weighted_sines = rho * torch.sin(omega * t_reshaped + phi) # Shape: (T, K)

        # Apply optional final projection
        if self.projection is not None:
            # Ensure dtype matches projection layer
            if weighted_sines.dtype != next(self.projection.parameters()).dtype:
                 weighted_sines = weighted_sines.to(dtype=next(self.projection.parameters()).dtype)
            p_t = self.projection(weighted_sines) # Shape: (T, output_dim)
        else:
            # If no projection, output_dim must equal num_components
            p_t = weighted_sines # Shape: (T, K=output_dim)

        return p_t

    def extra_repr(self) -> str:
        s = f"output_dim={self.output_dim}, num_components={self.num_components}"
        s += f", learnable_freq={self.learnable_freq}, learnable_amp_phase={self.learnable_amp_phase}"
        s += f", use_projection={self.use_projection}"
        if self.projection is not None:
             # Leverage LinearProjection's repr
             s += f"\n  (projection): {self.projection}"
        return s

# Example Usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    out_d = 128 # Final dimension d_p
    K = 64      # Number of Fourier components
    seq_length = 50

    print("--- Learnable Fourier PE (with Projection) ---")
    pos_enc_learnable = LearnedSinFourier(
        output_dim=out_d,
        num_components=K,
        learnable_freq=True,
        learnable_amp_phase=True,
        use_projection=True,
        device=device,
        dtype=dtype
    )
    print(pos_enc_learnable)
    p_t_learnable = pos_enc_learnable(seq_length)
    print("Output shape:", p_t_learnable.shape)
    assert p_t_learnable.shape == (seq_length, out_d)
    print("Sample output (first timestep):\n", p_t_learnable[0, :16])
    print("Sample output (last timestep):\n", p_t_learnable[-1, :16])

    print("\n--- Fixed Fourier PE (No Projection) ---")
    # Requires num_components == output_dim
    pos_enc_fixed = LearnedSinFourier(
        output_dim=K, # output_dim must match K
        num_components=K,
        learnable_freq=False,
        learnable_amp_phase=False, # Fixed amps/phases too
        use_projection=False,
        device=device,
        dtype=dtype
    )
    print(pos_enc_fixed)
    p_t_fixed = pos_enc_fixed(seq_length)
    print("Output shape:", p_t_fixed.shape)
    assert p_t_fixed.shape == (seq_length, K)
    print("Sample output (first timestep):\n", p_t_fixed[0, :16])
    print("Sample output (last timestep):\n", p_t_fixed[-1, :16])

    # Check parameter counts
    print("\nLearnable parameters:")
    for name, param in pos_enc_learnable.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

    print("\nFixed parameters (should have no learnable params):")
    has_learnable = False
    for name, param in pos_enc_fixed.named_parameters():
         if param.requires_grad:
             print(f"{name}: {param.shape}")
             has_learnable = True
    if not has_learnable:
        print("(None)")


