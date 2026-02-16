"""
Model Conversion: Replace dense FFN layers with trained MoE layers.

This script takes a pretrained dense model and the trained MoE layers,
and creates a new sparse model with MoE FFN layers.
"""

import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import copy

# Increase recursion limit for very deep models (140+ layers)
sys.setrecursionlimit(5000)


def replace_ffn_with_moe(
    model: nn.Module,
    moe_layers: nn.ModuleDict,
    layer_indices: List[int],
    ffn_name_pattern: str = "mlp",
    inplace: bool = False,
) -> nn.Module:
    """
    Replace FFN layers in a model with trained MoE layers.

    Args:
        model: Original dense model
        moe_layers: Trained MoE layers from distillation
        layer_indices: Which layer indices to replace
        ffn_name_pattern: Pattern to match FFN module names
        inplace: If True, modify model in place. If False, create a copy.

    Returns:
        Model with MoE layers replacing FFN layers
    """
    if not inplace:
        model = copy.deepcopy(model)

    layer_idx_set = set(layer_indices)
    replaced = 0

    def replace_module(parent: nn.Module, name: str, new_module: nn.Module):
        """Replace a child module."""
        setattr(parent, name, new_module)

    import re

    # Walk through model and find FFN layers
    for full_name, module in model.named_modules():
        # Check if this is a top-level FFN we should replace (ends with pattern)
        # e.g., "model.layers.0.mlp" but NOT "model.layers.0.mlp.gate_proj"
        if full_name.lower().endswith(ffn_name_pattern.lower()):
            # Extract layer index from name
            numbers = re.findall(r'\.(\d+)\.', full_name)
            if numbers:
                layer_idx = int(numbers[0])

                if layer_idx in layer_idx_set and str(layer_idx) in moe_layers:
                    # Find parent module
                    parts = full_name.rsplit('.', 1)
                    if len(parts) == 2:
                        parent_name, child_name = parts
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                        child_name = full_name

                    # Replace with MoE
                    moe_layer = moe_layers[str(layer_idx)]
                    replace_module(parent, child_name, moe_layer)
                    replaced += 1
                    print(f"Replaced {full_name} with MoE layer")

    print(f"\nReplaced {replaced} FFN layers with MoE layers")
    return model


class SparseModelWrapper(nn.Module):
    """
    Wrapper that provides a clean interface for the sparse model.

    Handles the different forward signatures between dense and MoE models.
    """

    def __init__(
        self,
        base_model: nn.Module,
        moe_layers: nn.ModuleDict,
        layer_indices: List[int],
    ):
        super().__init__()
        self.base_model = base_model
        self.moe_layers = moe_layers
        self.layer_indices = layer_indices

        # Try to extract model config
        self.config = getattr(base_model, 'config', None)

    def forward(self, *args, **kwargs):
        """Forward pass through the sparse model."""
        return self.base_model(*args, **kwargs)

    @property
    def n_experts(self) -> int:
        """Number of experts per layer."""
        first_moe = list(self.moe_layers.values())[0]
        return first_moe.n_experts

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in different components."""
        moe_params = sum(p.numel() for p in self.moe_layers.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        other_params = total_params - moe_params

        return {
            'total': total_params,
            'moe': moe_params,
            'other': other_params,
            'moe_percentage': 100 * moe_params / total_params,
        }

    def get_active_parameters(self, top_k: int = 2) -> int:
        """
        Estimate active parameters during inference.

        For MoE layers, only top_k experts are active per token.
        """
        # Get parameter IDs from MoE layers
        moe_param_ids = {id(p) for p in self.moe_layers.parameters()}
        n_experts = self.n_experts

        active = 0
        for param in self.parameters():
            if id(param) in moe_param_ids:
                # MoE parameter - only fraction is active
                active += param.numel() * top_k / n_experts
            else:
                # Non-MoE parameter - always active
                active += param.numel()

        return int(active)


def _infer_layer_dimensions_from_state_dict(state_dict: Dict, layer_indices: List[int]) -> Dict[int, Dict[str, int]]:
    """
    Infer n_experts and d_ff for each layer from the checkpoint state_dict.

    Expert weights have shape:
    - w1: (n_experts, d_model, d_ff)
    - w2: (n_experts, d_model, d_ff)
    - w3: (n_experts, d_ff, d_model)
    """
    layer_dimensions = {}

    for layer_idx in layer_indices:
        # Look for expert weights to infer dimensions
        w1_key = f"{layer_idx}.experts.w1"

        if w1_key in state_dict:
            w1_shape = state_dict[w1_key].shape
            n_experts = w1_shape[0]
            d_ff = w1_shape[2]  # w1 is (n_experts, d_model, d_ff)

            layer_dimensions[layer_idx] = {
                'n_experts': n_experts,
                'd_ff': d_ff,
            }

    return layer_dimensions


def convert_to_sparse(
    dense_model: nn.Module,
    distillation_checkpoint: str,
    device: str = "cuda",
    layer_config_path: Optional[str] = None,
    random_init: bool = False,
) -> SparseModelWrapper:
    """
    High-level function to convert a dense model to sparse using a distillation checkpoint.

    Args:
        dense_model: Original pretrained dense model
        distillation_checkpoint: Path to saved distillation checkpoint
        device: Device to load model on
        layer_config_path: Optional path to layer_config.json (only used if checkpoint
                          doesn't have layer_dimensions and you want to override)
        random_init: If True, skip loading weights (control trial for measuring distillation value)

    Returns:
        SparseModelWrapper containing the converted model
    """
    import json

    # Load distillation checkpoint
    # weights_only=False needed because checkpoint contains DistillationConfig dataclass
    checkpoint = torch.load(distillation_checkpoint, map_location=device, weights_only=False)

    # Extract info
    layer_indices = checkpoint['layer_indices']
    d_model = checkpoint['d_model']
    d_ff = checkpoint['d_ff']  # Dense FFN dimension
    d_ff_expert = checkpoint.get('d_ff_expert', d_ff)  # Default expert dimension
    config = checkpoint['config']

    # Parse dtype from checkpoint
    dtype_str = checkpoint.get('dtype', 'torch.float32')
    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float16': torch.float16,
        'torch.bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # ALWAYS infer dimensions from the actual weights in the checkpoint
    # This is the most reliable method - the weights have the ground truth
    moe_state_dict = checkpoint['moe_layers']
    layer_dimensions = _infer_layer_dimensions_from_state_dict(moe_state_dict, layer_indices)

    if layer_dimensions:
        print(f"Inferred layer dimensions from checkpoint weights:")
        n_experts_list = [layer_dimensions[i]['n_experts'] for i in layer_indices if i in layer_dimensions]
        d_ff_list = [layer_dimensions[i]['d_ff'] for i in layer_indices if i in layer_dimensions]
        print(f"  n_experts range: {min(n_experts_list)} - {max(n_experts_list)}")
        print(f"  d_ff range: {min(d_ff_list)} - {max(d_ff_list)}")
    else:
        # Fallback to checkpoint metadata or config
        layer_dimensions = checkpoint.get('layer_dimensions', None)

        # Try external layer_config_path as last resort
        if layer_dimensions is None and layer_config_path:
            with open(layer_config_path, 'r') as f:
                layer_config = json.load(f)
            print(f"Loaded layer config from: {layer_config_path}")

            layer_dimensions = {}
            for layer_idx in layer_indices:
                if str(layer_idx) in layer_config['layers']:
                    layer_cfg = layer_config['layers'][str(layer_idx)]
                    layer_dimensions[layer_idx] = {
                        'n_experts': layer_cfg['n_experts'],
                        'd_ff': layer_cfg['d_ff_expert'],
                    }

    # Recreate MoE layers with correct dimensions
    from .distill import MoEDistillationLayer

    moe_layers = nn.ModuleDict()
    for layer_idx in layer_indices:
        # Get per-layer dimensions or use defaults
        if layer_dimensions and layer_idx in layer_dimensions:
            n_experts = layer_dimensions[layer_idx]['n_experts']
            layer_d_ff = layer_dimensions[layer_idx]['d_ff']
        elif layer_dimensions and str(layer_idx) in layer_dimensions:
            # Handle string keys (from JSON)
            n_experts = layer_dimensions[str(layer_idx)]['n_experts']
            layer_d_ff = layer_dimensions[str(layer_idx)]['d_ff']
        else:
            n_experts = config.n_experts
            layer_d_ff = d_ff_expert

        moe_layer = MoEDistillationLayer(
            d_model=d_model,
            d_ff=layer_d_ff,
            n_experts=n_experts,
            capacity_factor=config.moe_capacity_factor,
            routing_type=config.moe_routing,
            # Router architecture (use getattr for backwards compatibility)
            mlp_router=getattr(config, 'router_mlp', False),
            router_hidden_dim=getattr(config, 'router_hidden_dim', 256),
            router_n_layers=getattr(config, 'router_n_layers', 2),
            use_expert_bias=getattr(config, 'use_expert_bias', True),
        )
        moe_layers[str(layer_idx)] = moe_layer

    # Load weights and move to device with correct dtype
    if random_init:
        print("\n*** RANDOM INIT MODE: Skipping weight loading (control trial) ***")
        # Keep random initialization - don't load checkpoint weights
    else:
        moe_layers.load_state_dict(checkpoint['moe_layers'])

    moe_layers = moe_layers.to(device=device, dtype=dtype)

    # Replace FFN layers in dense model
    # Note: dense_model is already on device from caller, MoE layers already moved above
    sparse_model = replace_ffn_with_moe(
        dense_model,
        moe_layers,
        layer_indices,
        inplace=False,
    )
    # Skip .to(device) - model already on device, and deep models hit recursion limit

    # Wrap in clean interface
    wrapper = SparseModelWrapper(
        sparse_model,
        moe_layers,
        layer_indices,
    )

    # Print summary
    params = wrapper.count_parameters()
    print(f"\nSparse model created:")
    if random_init:
        print(f"  *** CONTROL TRIAL: Random MoE weights ***")
    print(f"  Total params: {params['total'] / 1e9:.2f}B")
    print(f"  MoE params: {params['moe'] / 1e9:.2f}B ({params['moe_percentage']:.1f}%)")
    print(f"  Active params (top-2): {wrapper.get_active_parameters() / 1e9:.2f}B")

    return wrapper
