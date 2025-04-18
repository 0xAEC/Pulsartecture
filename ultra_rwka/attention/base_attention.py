# ultra_rwka/components/attention/base_attention.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class BaseAttention(nn.Module, ABC):
    """
    Abstract Base Class for Attention Mechanisms in the Ultra-RWKA framework.

    This class defines the common interface that all attention modules should adhere to,
    promoting consistency, modularity, and extensibility. Concrete attention
    implementations (like TMLinKernelAttn) should inherit from this class and
    implement the abstract `forward` method.
    """

    def __init__(self,
                 embed_dim: int,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):
        """
        Initializes the base attention module.

        Args:
            embed_dim (int): The primary embedding dimension of the input tensor `x`
                             that the attention mechanism operates on. Subclasses typically
                             use this for setting up internal projections (Q, K, V, etc.).
            device (Optional[torch.device]): The target device for the module's parameters
                                             and buffers. Defaults to None (uses default device).
            dtype (Optional[torch.dtype]): The target data type for the module's parameters
                                           and buffers. Defaults to None (uses default dtype).
            **kwargs: Catches any additional keyword arguments passed during instantiation
                      for flexibility, though not explicitly used by the base class itself.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # Store device and dtype, useful for subclasses creating parameters/buffers
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # Placeholder for potential future common attributes if needed
        # e.g., self.output_dim = embed_dim # Or allow subclasses to define

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Abstract forward method for applying the attention mechanism.

        This method must be implemented by all concrete subclasses. It defines
        how the attention mechanism processes the input and produces an output.

        Args:
            x (torch.Tensor): The primary input tensor, typically representing a sequence
                              of embeddings. Expected Shape: (Batch, SequenceLength, embed_dim).
            **kwargs (Any): Optional keyword arguments specific to the attention implementation.
                            This allows for flexibility in passing masks, context vectors,
                            previous states, or other necessary information without rigidly
                            defining them in the base class signature. Examples include:
                            - `mask: Optional[torch.Tensor]` for causal or padding masks.
                            - `temp_input: Optional[torch.Tensor]` for TM-KLA temperature.
                            - `mixer_context: Optional[torch.Tensor]` for TM-KLA kernel mixing.
                            - `memory: Optional[torch.Tensor]` for memory-augmented attention.
                            - `state: Optional[Any]` for recurrent attention variants.

        Returns:
            torch.Tensor: The output tensor after applying attention. The shape is
                          typically (Batch, SequenceLength, output_dim), where output_dim
                          might be the same as embed_dim or different (e.g., sum of head dims).
        """
        pass # Subclasses must implement this

    # Optional: Define common utility methods if applicable to many attention types

    def extra_repr(self) -> str:
        """
        Provides a base representation string including the embedding dimension.
        Subclasses should extend this to include their specific important parameters.

        Example Usage in Subclass:
            def extra_repr(self) -> str:
                base_repr = super().extra_repr()
                return f'{base_repr}, num_heads={self.num_heads}, dropout={self.dropout_p}'
        """
        return f'embed_dim={self.embed_dim}'

    # Make the module callable like a function
    def __call__(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # This ensures that when you call `attn_layer(x, ...)`, it routes to forward
        # while also handling PyTorch hooks, etc.
        return super().__call__(x, **kwargs)
