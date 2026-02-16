"""
Activation hooks for extracting FFN inputs and outputs from dense models.

Supports both HuggingFace transformers and custom architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field


@dataclass
class FFNActivation:
    """Stores input and output activations for a single FFN layer."""
    layer_idx: int
    input: torch.Tensor   # (batch, seq, d_model)
    output: torch.Tensor  # (batch, seq, d_model)


class FFNActivationCache:
    """
    Hook-based cache for extracting FFN activations during forward pass.

    Automatically detects FFN layers in common architectures:
    - HuggingFace: LlamaMLP, GemmaMLP, MistralMLP, etc.
    - Custom: Any module with 'ffn' or 'mlp' in name

    Usage:
        cache = FFNActivationCache(model)
        cache.register_hooks()

        with torch.no_grad():
            model(input_ids)

        activations = cache.get_activations()
        cache.clear()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    ):
        """
        Args:
            model: The dense model to extract activations from
            layer_filter: Optional function (name, module) -> bool to filter FFN layers
        """
        self.model = model
        self.layer_filter = layer_filter or self._default_filter

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[int, FFNActivation] = {}
        self._ffn_modules: Dict[int, nn.Module] = {}

        # Discover FFN layers
        self._discover_ffn_layers()

    def _default_filter(self, name: str, module: nn.Module) -> bool:
        """Default filter to identify FFN/MLP layers."""
        name_lower = name.lower()

        # Common FFN layer names
        ffn_indicators = ['mlp', 'ffn', 'feed_forward', 'feedforward']

        # Check if this is likely an FFN layer
        is_ffn_name = any(ind in name_lower for ind in ffn_indicators)

        # Also check module type names
        type_name = type(module).__name__.lower()
        is_ffn_type = any(ind in type_name for ind in ffn_indicators)

        # Must have parameters and be a container (not just a linear layer)
        has_children = len(list(module.children())) > 0

        return (is_ffn_name or is_ffn_type) and has_children

    def _discover_ffn_layers(self):
        """Find all FFN layers in the model."""
        self._ffn_modules = {}

        for name, module in self.model.named_modules():
            if self.layer_filter(name, module):
                # Extract layer index from name
                layer_idx = self._extract_layer_idx(name)
                if layer_idx is not None and layer_idx not in self._ffn_modules:
                    self._ffn_modules[layer_idx] = module

        # Sort by layer index
        self._ffn_modules = dict(sorted(self._ffn_modules.items()))

        if not self._ffn_modules:
            raise ValueError(
                "No FFN layers found. Try providing a custom layer_filter function."
            )

    def _extract_layer_idx(self, name: str) -> Optional[int]:
        """Extract layer index from module name like 'model.layers.5.mlp'."""
        import re
        # Find all numbers in the name
        numbers = re.findall(r'\.(\d+)\.', name)
        if numbers:
            # Usually the first number is the layer index
            return int(numbers[0])
        return None

    @property
    def n_layers(self) -> int:
        """Number of FFN layers discovered."""
        return len(self._ffn_modules)

    @property
    def layer_indices(self) -> List[int]:
        """List of layer indices."""
        return list(self._ffn_modules.keys())

    def register_hooks(self):
        """Register forward hooks on all FFN layers."""
        self.remove_hooks()  # Clear any existing hooks

        for layer_idx, module in self._ffn_modules.items():
            # Create hook that captures both input and output
            hook = self._create_hook(layer_idx)
            handle = module.register_forward_hook(hook)
            self._hooks.append(handle)

    def _create_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""
        def hook(module: nn.Module, inputs: Tuple, output: torch.Tensor):
            # inputs is a tuple, first element is the hidden states
            input_tensor = inputs[0] if isinstance(inputs, tuple) else inputs

            # Detach and clone to avoid memory issues
            self._activations[layer_idx] = FFNActivation(
                layer_idx=layer_idx,
                input=input_tensor.detach(),
                output=output.detach(),
            )
        return hook

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def clear(self):
        """Clear cached activations."""
        self._activations = {}

    def get_activations(self) -> Dict[int, FFNActivation]:
        """Get all cached activations."""
        return self._activations

    def get_layer_activation(self, layer_idx: int) -> Optional[FFNActivation]:
        """Get activation for a specific layer."""
        return self._activations.get(layer_idx)

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, *args):
        """Context manager exit - remove hooks and clear cache."""
        self.remove_hooks()
        self.clear()


class StreamingFFNExtractor:
    """
    Memory-efficient streaming extractor that yields activations batch-by-batch.

    Instead of caching all activations, this yields them as the model processes
    each batch, allowing for immediate use in MoE training.

    Usage:
        extractor = StreamingFFNExtractor(dense_model)

        for batch in dataloader:
            activations = extractor.extract(batch['input_ids'])
            # activations is Dict[layer_idx, FFNActivation]

            for layer_idx, act in activations.items():
                moe_output = moe_layers[layer_idx](act.input)
                loss = F.mse_loss(moe_output, act.output)
    """

    def __init__(self, model: nn.Module, **kwargs):
        """
        Args:
            model: Dense model to extract from
            **kwargs: Passed to FFNActivationCache
        """
        self.model = model
        self.cache = FFNActivationCache(model, **kwargs)
        self.cache.register_hooks()

        # Put model in eval mode for consistent outputs
        self.model.eval()

    @torch.no_grad()
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[int, FFNActivation]:
        """
        Run forward pass and extract all FFN activations.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional args passed to model forward

        Returns:
            Dict mapping layer_idx to FFNActivation
        """
        self.cache.clear()

        # Forward pass through dense model
        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        return self.cache.get_activations()

    @property
    def n_layers(self) -> int:
        return self.cache.n_layers

    @property
    def layer_indices(self) -> List[int]:
        return self.cache.layer_indices

    def __del__(self):
        """Cleanup hooks on deletion."""
        if hasattr(self, 'cache'):
            self.cache.remove_hooks()
