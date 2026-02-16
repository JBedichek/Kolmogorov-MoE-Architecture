"""
Sparse Distillation Trainer: Train MoE layers to approximate dense FFN layers.

This module implements the core training loop for distilling dense FFN layers
into MoE layers using a streaming approach (no disk caching).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from numbers import Number
from dataclasses import dataclass
from tqdm import tqdm
import math
import copy
import numpy as np

from .hooks import StreamingFFNExtractor, FFNActivation

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from moe_arch.model.moe import GroupedExpertsExpertChoice, ExpertChoiceRouter
from moe_arch.model.config import AdvancedMoEConfig

# Dynamic expert reallocation (optional)
try:
    from .dynamic_experts import (
        DynamicExpertLayer,
        DynamicExpertTracker,
        DynamicExpertConfig,
        reallocate_expert_layer,
        interpolate_expert_weights,
        print_active_params_report,
    )
    DYNAMIC_EXPERTS_AVAILABLE = True
except ImportError:
    DYNAMIC_EXPERTS_AVAILABLE = False


def print_layer_routing_stats(moe_layer, layer_idx: int, prefix: str = ""):
    """
    Print routing statistics for a single MoE layer.

    Args:
        moe_layer: The MoE layer to analyze
        layer_idx: Layer index for display
        prefix: Optional prefix for output lines
    """
    n_experts = moe_layer.n_experts

    # Get expert counts from last forward pass
    if not hasattr(moe_layer, '_last_expert_counts') or moe_layer._last_expert_counts is None:
        print(f"{prefix}Layer {layer_idx}: No routing data available")
        return

    expert_counts = moe_layer._last_expert_counts.float()
    total_tokens = expert_counts.sum()

    if total_tokens == 0:
        print(f"{prefix}Layer {layer_idx}: No tokens routed")
        return

    # Compute statistics
    fracs = expert_counts / total_tokens

    # Entropy (higher = more balanced)
    probs = fracs + 1e-10
    entropy = -(probs * torch.log(probs)).sum().item()
    max_entropy = math.log(n_experts)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Usage stats
    min_frac = fracs.min().item()
    max_frac = fracs.max().item()
    active_experts = (expert_counts > 0).sum().item()

    # Detect collapse
    uniform_share = 1.0 / n_experts
    meaningful_threshold = 0.1 * uniform_share
    meaningful_experts = (fracs > meaningful_threshold).sum().item()
    collapsed = meaningful_experts < n_experts * 0.5

    # Print summary
    status = "COLLAPSED!" if collapsed else ("WARNING" if normalized_entropy < 0.7 else "OK")

    print(f"{prefix}Layer {layer_idx} Routing Stats:")
    print(f"{prefix}  Experts: {n_experts}, Active: {int(active_experts)}, Meaningful: {int(meaningful_experts)}")
    print(f"{prefix}  Normalized entropy: {normalized_entropy:.3f} (1.0 = perfect balance)")
    print(f"{prefix}  Usage range: {min_frac*100:.2f}% - {max_frac*100:.1f}% (imbalance: {max_frac/(min_frac+1e-10):.1f}x)")
    print(f"{prefix}  Status: {status}")

    # Show top/bottom experts if imbalanced
    if normalized_entropy < 0.9:
        sorted_indices = torch.argsort(expert_counts, descending=True)
        top_3 = sorted_indices[:3].tolist()
        bottom_3 = sorted_indices[-3:].tolist()
        top_3_pcts = [fracs[i].item() * 100 for i in top_3]
        bottom_3_pcts = [fracs[i].item() * 100 for i in bottom_3]
        print(f"{prefix}  Top 3 experts: {top_3} ({top_3_pcts[0]:.1f}%, {top_3_pcts[1]:.1f}%, {top_3_pcts[2]:.1f}%)")
        print(f"{prefix}  Bottom 3 experts: {bottom_3} ({bottom_3_pcts[0]:.2f}%, {bottom_3_pcts[1]:.2f}%, {bottom_3_pcts[2]:.2f}%)")


@dataclass
class DistillationConfig:
    """Configuration for sparse distillation."""
    # MoE architecture
    n_experts: int = 16
    moe_top_k: int = 2  # For token-choice: experts per token
    moe_capacity_factor: float = 1.25  # For expert-choice: tokens_per_expert = seq_len * cap_factor / n_experts
    # NOTE: token_choice recommended for distillation - guarantees every token is processed
    # expert_choice can leave tokens with near-zero output, causing poor distillation
    moe_routing: str = "token_choice"

    # Expert dimensions
    # None = match dense FFN dimension (d_ff)
    # Float (0-1) = fraction of dense d_ff (e.g., 0.5 = half size per expert)
    # Int > 1 = absolute dimension
    d_ff_expert: Optional[Union[int, float]] = None

    # Per-layer config (overrides n_experts and d_ff_expert per layer)
    # Path to layer_config.json from generate_config.py
    layer_config_path: Optional[str] = None

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Loss
    use_cosine_loss: bool = True  # Add cosine similarity loss
    cosine_loss_weight: float = 0.1

    # Auxiliary losses for routing
    use_load_balancing: bool = True  # Encourage even expert usage
    load_balancing_weight: float = 0.01
    use_z_loss: bool = True  # Prevent router logits from exploding
    z_loss_weight: float = 0.001

    # Expert initialization
    init_from_dense: bool = True  # Initialize experts from dense FFN weights

    # Memory optimization
    sequential_layers: bool = False  # Train one layer at a time (much lower memory)

    # Dynamic expert reallocation
    dynamic_experts: bool = False  # Enable dynamic expert sizing based on loss
    reallocate_every_n_steps: int = 500  # How often to reallocate
    growth_factor: float = 1.25  # Grow high-loss experts by this factor
    shrink_factor: float = 0.8   # Shrink low-loss experts by this factor
    min_d_ff: int = 256  # Minimum expert dimension
    max_d_ff: int = 8192  # Maximum expert dimension
    top_k_grow: int = 2  # Number of experts to grow
    top_k_shrink: int = 2  # Number of experts to shrink

    # Logging
    log_interval: int = 100

    # Data-free distillation (use random inputs instead of real data)
    data_free: bool = True  # Recommended: faster and often better coverage
    data_free_batch_size: int = 4
    data_free_seq_len: int = 512
    data_free_steps: int = 1000  # Steps per layer

    # Performance
    use_compile: bool = True  # Use torch.compile for faster training

    # Supervised routing (cluster-based)
    supervised_routing: bool = True  # Use clustering to supervise router
    cluster_samples: int = 10000  # Samples to collect for clustering
    cluster_batch_size: int = 32  # Batch size for clustering data collection
    router_train_steps: int = 2000  # Steps to train router per layer

    # Empirical routing (train router based on actual expert performance)
    # This runs all experts and trains router to select lowest-loss experts
    empirical_routing: bool = False  # Use empirical routing instead of clustering
    empirical_eval_interval: int = 10  # Recompute routing targets every N steps
    empirical_temperature: float = 0.1  # Temperature for softmax over -losses
    empirical_entropy_weight: float = 0.01  # Entropy bonus to prevent collapse

    # Curriculum balanced routing (prevents collapse by starting uniform, gradually specializing)
    curriculum_routing: bool = True  # Enable curriculum balanced routing
    curriculum_warmup_steps: int = 5000  # Steps to transition from uniform to empirical targets
    curriculum_initial_balance: float = 0.9  # Initial balance weight (0=empirical, 1=uniform)
    curriculum_final_balance: float = 0.3  # Final balance weight after warmup (keep some balance always)
    # Target smoothing to prevent collapse
    target_max_prob: float = 0.25  # Max probability for any single expert (clips and renormalizes)
    target_temperature_schedule: bool = True  # Anneal temperature from high to low
    target_initial_temperature: float = 1.0  # Start with high temperature (flat targets)
    target_final_temperature: float = 0.1  # End with low temperature (peaked targets)
    # Router training isolation
    detach_router_from_expert_loss: bool = True  # Stop gradient from expert loss to router
    router_loss_weight: float = 1.0  # Weight for router KL loss (increase to prioritize balance)

    # Per-expert bias (DeepSeek-style load balancing)
    use_expert_bias: bool = True  # Learnable bias per expert to help underused experts

    # Gradient-based routing (route to experts that would learn most)
    gradient_routing: bool = True  # Use gradient norms instead of losses for routing targets
    gradient_routing_fast: bool = True  # Use fast approximation (error magnitude) vs exact gradients

    # Router architecture
    router_mlp: bool = False  # Use MLP router instead of linear
    router_hidden_dim: int = 256  # Hidden dimension for MLP router
    router_n_layers: int = 2  # Number of layers in MLP router


class MLPRouter(nn.Module):
    """MLP-based router with non-linearities for better expressivity."""

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts

        layers = []
        in_dim = d_model
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_experts))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        Returns:
            logits: (batch, seq, n_experts)
        """
        return self.mlp(x)


class MoEDistillationLayer(nn.Module):
    """
    A single MoE layer being trained to match a dense FFN.

    Combines router and experts, optimized for distillation.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        capacity_factor: float = 1.25,
        routing_type: str = "expert_choice",
        mlp_router: bool = False,
        router_hidden_dim: int = 256,
        router_n_layers: int = 2,
        use_expert_bias: bool = True,  # DeepSeek-style per-expert bias for load balancing
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.routing_type = routing_type
        self.mlp_router = mlp_router
        self.use_expert_bias = use_expert_bias

        # Per-expert learnable bias (DeepSeek-style)
        # Initialized to zero, will learn to boost underused experts
        if use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(n_experts))
        else:
            self.register_buffer('expert_bias', None)

        # Create a minimal config for the router
        config = AdvancedMoEConfig(
            d_model=d_model,
            n_experts=n_experts,
            moe_capacity_factor=capacity_factor,
            d_ff_expert=d_ff,
        )

        if routing_type == "expert_choice":
            if mlp_router:
                self.router = MLPRouter(d_model, n_experts, router_hidden_dim, router_n_layers)
                self._base_router = ExpertChoiceRouter(config)  # For routing logic
            else:
                self.router = ExpertChoiceRouter(config)
            self.experts = GroupedExpertsExpertChoice(
                n_experts=n_experts,
                d_model=d_model,
                d_ff=d_ff,
                activation="swiglu",
            )
        else:
            # Token-choice routing
            from moe_arch.model.moe import Router, GroupedExperts
            if mlp_router:
                self.router = MLPRouter(d_model, n_experts, router_hidden_dim, router_n_layers)
                self._base_router = Router(config)  # For routing logic
            else:
                self.router = Router(config)
            self.experts = GroupedExperts(
                n_experts=n_experts,
                d_model=d_model,
                d_ff=d_ff,
                activation="swiglu",
            )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Forward pass through MoE layer.

        Args:
            x: (batch, seq_len, d_model)
            return_aux: If True, return (output, aux_info) for loss computation

        Returns:
            output: (batch, seq_len, d_model)
            aux_info: dict with router_logits, expert_counts, etc. (if return_aux=True)

        Note: Always caches _last_router_logits and _last_expert_counts for auxiliary loss computation.
        """
        batch_size, seq_len, d_model = x.shape

        if self.mlp_router:
            # MLP router: get logits directly
            router_logits = self.router(x)  # (batch, seq, n_experts)

            # Add per-expert bias (DeepSeek-style load balancing)
            if self.use_expert_bias and self.expert_bias is not None:
                router_logits = router_logits + self.expert_bias

            if self.routing_type == "expert_choice":
                # Expert-choice: each expert picks top-k tokens
                # Transpose to (batch, n_experts, seq) for expert selection
                logits_t = router_logits.transpose(1, 2)  # (batch, n_experts, seq)
                capacity = int(seq_len * self.capacity_factor / self.n_experts)
                capacity = max(1, capacity)

                # Each expert selects top-capacity tokens
                expert_weights, token_indices = torch.topk(logits_t, k=capacity, dim=-1)
                expert_weights = F.softmax(expert_weights, dim=-1)

                output = self.experts(x, expert_weights, token_indices)

                # Cache for auxiliary loss computation
                expert_counts = torch.ones(self.n_experts, device=x.device) * capacity * batch_size
                self._last_router_logits = router_logits.detach()
                self._last_expert_counts = expert_counts.detach()

                if return_aux:
                    aux_info = {
                        'router_logits': router_logits,
                        'expert_counts': expert_counts,
                        'expert_weights': expert_weights,
                    }
                    return output, aux_info
            else:
                # Token-choice: each token picks top-k experts
                from moe_arch.model.moe import Router
                top_k = getattr(self._base_router, 'top_k', 2)

                # Select top-k experts per token
                routing_weights, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)
                routing_weights = F.softmax(routing_weights, dim=-1)

                x_flat = x.view(-1, d_model)
                routing_weights_flat = routing_weights.view(-1, top_k)
                selected_experts_flat = selected_experts.view(-1, top_k)
                output_flat = self.experts(x_flat, selected_experts_flat, routing_weights_flat)
                output = output_flat.view(batch_size, seq_len, d_model)

                # Cache for auxiliary loss computation
                expert_counts = torch.zeros(self.n_experts, device=x.device)
                for i in range(self.n_experts):
                    expert_counts[i] = (selected_experts == i).sum().float()
                self._last_router_logits = router_logits.detach()
                self._last_expert_counts = expert_counts.detach()

                if return_aux:
                    aux_info = {
                        'router_logits': router_logits,
                        'expert_counts': expert_counts,
                        'routing_weights': routing_weights,
                        'selected_experts': selected_experts,
                    }
                    return output, aux_info
        else:
            # Standard router
            if self.routing_type == "expert_choice":
                expert_weights, token_indices, router_logits, capacity = self.router(x)
                output = self.experts(x, expert_weights, token_indices)

                # Cache for auxiliary loss computation
                batch_size_local, n_experts, cap = token_indices.shape
                expert_counts = torch.ones(n_experts, device=x.device) * cap * batch_size_local
                self._last_router_logits = router_logits.detach()
                self._last_expert_counts = expert_counts.detach()

                if return_aux:
                    aux_info = {
                        'router_logits': router_logits,
                        'expert_counts': expert_counts,
                        'expert_weights': expert_weights,
                    }
                    return output, aux_info
            else:
                routing_weights, selected_experts, router_logits = self.router(x)
                x_flat = x.view(-1, d_model)
                routing_weights_flat = routing_weights.view(-1, routing_weights.shape[-1])
                selected_experts_flat = selected_experts.view(-1, selected_experts.shape[-1])
                output_flat = self.experts(x_flat, selected_experts_flat, routing_weights_flat)
                output = output_flat.view(batch_size, seq_len, d_model)

                # Cache for auxiliary loss computation
                expert_counts = torch.zeros(self.n_experts, device=x.device)
                for i in range(self.n_experts):
                    expert_counts[i] = (selected_experts == i).sum().float()
                self._last_router_logits = router_logits.detach()
                self._last_expert_counts = expert_counts.detach()

                if return_aux:
                    aux_info = {
                        'router_logits': router_logits,
                        'expert_counts': expert_counts,
                        'routing_weights': routing_weights,
                        'selected_experts': selected_experts,
                    }
                    return output, aux_info

        return output

    def get_router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw router logits for supervised training (includes expert bias)."""
        # MLP router: direct forward call
        if self.mlp_router:
            logits = self.router(x)
        elif hasattr(self.router, 'proj'):
            # Standard router: access the projection layer
            logits = self.router.proj(x)
        elif hasattr(self.router, 'gate'):
            logits = self.router.gate(x)
        else:
            # Try forward and extract
            if self.routing_type == "expert_choice":
                _, _, logits, _ = self.router(x)
            else:
                _, _, logits = self.router(x)

        # Add per-expert bias (DeepSeek-style load balancing)
        if self.use_expert_bias and self.expert_bias is not None:
            logits = logits + self.expert_bias

        return logits

    def init_from_dense_ffn(self, dense_ffn: nn.Module):
        """
        Initialize expert weights from dense FFN weights.

        Copies the dense FFN weights to all experts, then adds small noise
        to break symmetry.
        """
        # Try to extract weights from common FFN structures
        gate_weight = None
        up_weight = None
        down_weight = None

        # HuggingFace Llama-style MLP
        if hasattr(dense_ffn, 'gate_proj'):
            gate_weight = dense_ffn.gate_proj.weight.data  # (d_ff, d_model)
            up_weight = dense_ffn.up_proj.weight.data      # (d_ff, d_model)
            down_weight = dense_ffn.down_proj.weight.data  # (d_model, d_ff)
        # Alternative naming
        elif hasattr(dense_ffn, 'w1'):
            gate_weight = dense_ffn.w1.weight.data
            up_weight = dense_ffn.w2.weight.data if hasattr(dense_ffn, 'w2') else None
            down_weight = dense_ffn.w3.weight.data if hasattr(dense_ffn, 'w3') else dense_ffn.w2.weight.data

        if gate_weight is None:
            print("  Warning: Could not extract dense FFN weights, using random init")
            return

        # Our experts store weights as (n_experts, d_model, d_ff) for w1/w2
        # and (n_experts, d_ff, d_model) for w3
        # Dense FFN stores as (d_ff, d_model) for gate/up and (d_model, d_ff) for down

        with torch.no_grad():
            for expert_idx in range(self.n_experts):
                # Add small noise to break symmetry (scale by weight magnitude)
                noise_scale = 0.01

                if hasattr(self.experts, 'w1'):
                    # w1 corresponds to gate_proj: (n_experts, d_model, d_ff)
                    base = gate_weight.T  # (d_model, d_ff)
                    noise = torch.randn_like(base) * base.std() * noise_scale
                    self.experts.w1.data[expert_idx] = base + noise

                if hasattr(self.experts, 'w2') and up_weight is not None:
                    # w2 corresponds to up_proj: (n_experts, d_model, d_ff)
                    base = up_weight.T  # (d_model, d_ff)
                    noise = torch.randn_like(base) * base.std() * noise_scale
                    self.experts.w2.data[expert_idx] = base + noise

                if hasattr(self.experts, 'w3') and down_weight is not None:
                    # w3 corresponds to down_proj: (n_experts, d_ff, d_model)
                    base = down_weight.T  # (d_ff, d_model)
                    noise = torch.randn_like(base) * base.std() * noise_scale
                    self.experts.w3.data[expert_idx] = base + noise


class SparseDistillationTrainer:
    """
    Main trainer for sparse distillation.

    Converts a dense transformer into a sparse MoE model by training
    MoE layers to approximate each dense FFN layer.
    """

    def __init__(
        self,
        dense_model: nn.Module,
        config: Optional[DistillationConfig] = None,
        device: str = "cuda",
        layer_filter=None,
    ):
        """
        Args:
            dense_model: Pretrained dense transformer model
            config: Distillation configuration
            device: Device to train on
            layer_filter: Optional filter for FFN layer detection
        """
        self.dense_model = dense_model.to(device)
        self.dense_model.eval()  # Dense model stays frozen

        self.config = config or DistillationConfig()
        self.device = device

        # Create activation extractor
        self.extractor = StreamingFFNExtractor(
            dense_model,
            layer_filter=layer_filter,
        )

        # Infer model dimensions and dtype from first FFN layer
        self.d_model, self.d_ff, self.dtype = self._infer_dimensions()

        # Create MoE layers for each FFN layer
        self.moe_layers = nn.ModuleDict()
        self._create_moe_layers()

        # Create optimizers (one per layer for flexibility)
        self.optimizers = {}
        self._create_optimizers()

        # Training state
        self.global_step = 0
        self.layer_losses: Dict[int, List[float]] = {
            idx: [] for idx in self.extractor.layer_indices
        }

    def _infer_dimensions(self) -> Tuple[int, int, torch.dtype]:
        """Infer d_model, d_ff, and dtype from the dense model."""
        # Get first FFN module
        first_ffn = list(self.extractor.cache._ffn_modules.values())[0]

        d_model = None
        d_ff = None
        dtype = torch.float32

        # Try common attribute names
        if hasattr(first_ffn, 'gate_proj'):
            d_ff, d_model = first_ffn.gate_proj.weight.shape
            dtype = first_ffn.gate_proj.weight.dtype
        elif hasattr(first_ffn, 'w1'):
            d_ff, d_model = first_ffn.w1.weight.shape
            dtype = first_ffn.w1.weight.dtype
        elif hasattr(first_ffn, 'fc1'):
            d_ff, d_model = first_ffn.fc1.weight.shape
            dtype = first_ffn.fc1.weight.dtype
        else:
            # Try to infer from children
            for child in first_ffn.children():
                if isinstance(child, nn.Linear):
                    out_feat, in_feat = child.weight.shape
                    dtype = child.weight.dtype
                    if d_model is None:
                        d_model = in_feat
                        d_ff = out_feat
                    break

        if d_model is None:
            raise ValueError("Could not infer model dimensions from FFN layer")

        print(f"Inferred dimensions: d_model={d_model}, d_ff={d_ff}, dtype={dtype}")
        return d_model, d_ff, dtype

    def _compute_expert_dim(self) -> int:
        """Compute expert FFN dimension based on config."""
        cfg_dim = self.config.d_ff_expert

        if cfg_dim is None:
            # Match dense FFN dimension
            return self.d_ff
        elif isinstance(cfg_dim, float) and 0 < cfg_dim <= 1:
            # Fraction of dense dimension
            return int(self.d_ff * cfg_dim)
        elif isinstance(cfg_dim, int) and cfg_dim > 1:
            # Absolute dimension
            return cfg_dim
        else:
            raise ValueError(
                f"d_ff_expert must be None, float in (0,1], or int > 1. Got: {cfg_dim}"
            )

    def _count_ffn_params(self, ffn_module: nn.Module) -> int:
        """Count parameters in a dense FFN module."""
        return sum(p.numel() for p in ffn_module.parameters())

    def _create_moe_layers(self):
        """Create MoE layers for each FFN layer in the dense model."""
        import json

        # Load per-layer config if provided
        self.layer_config = None
        if self.config.layer_config_path:
            with open(self.config.layer_config_path, 'r') as f:
                self.layer_config = json.load(f)
            print(f"\nLoaded per-layer config from: {self.config.layer_config_path}")

        # Compute default expert dimension (used if no per-layer config)
        self.d_ff_expert = self._compute_expert_dim()

        # Count dense FFN params (from first layer)
        first_ffn = list(self.extractor.cache._ffn_modules.values())[0]
        dense_ffn_params = self._count_ffn_params(first_ffn)

        print(f"\nCreating {self.extractor.n_layers} MoE layers...")
        print(f"  d_model: {self.d_model}")
        print(f"  d_ff (dense): {self.d_ff}")
        if self.layer_config:
            print(f"  Using per-layer config (variable n_experts and d_ff_expert)")
        else:
            print(f"  d_ff_expert: {self.d_ff_expert}")
            print(f"  n_experts: {self.config.n_experts}")
        print(f"  routing: {self.config.moe_routing}")
        print(f"  dtype: {self.dtype}")

        total_moe_params = 0
        total_active_params = 0

        for layer_idx in self.extractor.layer_indices:
            # Get per-layer config or use defaults
            if self.layer_config and str(layer_idx) in self.layer_config['layers']:
                layer_cfg = self.layer_config['layers'][str(layer_idx)]
                n_experts = layer_cfg['n_experts']
                d_ff_expert = layer_cfg['d_ff_expert']
            else:
                n_experts = self.config.n_experts
                d_ff_expert = self.d_ff_expert

            moe_layer = MoEDistillationLayer(
                d_model=self.d_model,
                d_ff=d_ff_expert,
                n_experts=n_experts,
                capacity_factor=self.config.moe_capacity_factor,
                routing_type=self.config.moe_routing,
                mlp_router=self.config.router_mlp,
                router_hidden_dim=self.config.router_hidden_dim,
                router_n_layers=self.config.router_n_layers,
                use_expert_bias=self.config.use_expert_bias,
            ).to(device=self.device, dtype=self.dtype)

            # Optionally initialize from dense FFN (only works if dimensions match)
            if self.config.init_from_dense and d_ff_expert == self.d_ff:
                dense_ffn = self.extractor.cache._ffn_modules[layer_idx]
                moe_layer.init_from_dense_ffn(dense_ffn)

            self.moe_layers[str(layer_idx)] = moe_layer

            # Track params
            layer_params = n_experts * 3 * self.d_model * d_ff_expert
            total_moe_params += layer_params
            top_k = self.config.moe_top_k if self.config.moe_routing == "token_choice" else 2
            total_active_params += layer_params * top_k / n_experts

        # Print summary
        total_dense_ffn_params = dense_ffn_params * self.extractor.n_layers
        total_dense_params = sum(p.numel() for p in self.dense_model.parameters())
        non_ffn_params = total_dense_params - total_dense_ffn_params

        sparse_total_params = non_ffn_params + total_moe_params
        sparse_active_params = non_ffn_params + total_active_params

        print(f"\nSparse model summary:")
        print(f"  Dense model:     {total_dense_params:,} ({total_dense_params / 1e9:.2f}B)")
        print(f"  Sparse model:    {sparse_total_params:,} ({sparse_total_params / 1e9:.2f}B) total")
        print(f"  Active params:   {sparse_active_params:,} ({sparse_active_params / 1e9:.2f}B)")
        print(f"  Expansion:       {total_moe_params / total_dense_ffn_params:.2f}x")
        print(f"  Sparsity:        {100 * (1 - sparse_active_params / sparse_total_params):.1f}%")

        if self.layer_config:
            n_experts_list = [self.layer_config['layers'][str(i)]['n_experts']
                             for i in self.extractor.layer_indices
                             if str(i) in self.layer_config['layers']]
            print(f"  n_experts range: {min(n_experts_list)} - {max(n_experts_list)}")

        print(f"\nCreated {len(self.moe_layers)} MoE layers")

    def _create_optimizers(self):
        """Create optimizers lazily - only allocate state when needed."""
        # Don't create optimizers here - create them on-demand in train_step
        # This saves memory by not allocating Adam state for all layers upfront
        pass

    def _get_optimizer(self, layer_idx: int) -> torch.optim.Optimizer:
        """Get or create optimizer for a layer (lazy initialization)."""
        if layer_idx not in self.optimizers:
            moe_layer = self.moe_layers[str(layer_idx)]

            # If supervised routing, exclude router params (they're already trained)
            if self.config.supervised_routing:
                params = [p for n, p in moe_layer.named_parameters() if 'router' not in n]
            else:
                params = moe_layer.parameters()

            self.optimizers[layer_idx] = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        return self.optimizers[layer_idx]

    def compute_load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        expert_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert usage.

        L_lb = n_experts * sum(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = mean routing probability for expert i
        """
        # f_i: fraction of tokens per expert
        total_tokens = expert_counts.sum()
        f = expert_counts / (total_tokens + 1e-8)

        # P_i: mean routing probability per expert
        # router_logits: (batch, seq, n_experts) or (batch * seq, n_experts)
        if router_logits.dim() == 3:
            router_logits = router_logits.view(-1, router_logits.shape[-1])
        probs = F.softmax(router_logits, dim=-1)
        P = probs.mean(dim=0)

        # Load balancing loss - use actual n_experts from logits shape
        n_experts = router_logits.shape[-1]
        lb_loss = n_experts * (f * P).sum()
        return lb_loss

    def compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute z-loss to prevent router logits from becoming too large.

        L_z = (1/n) * sum(log(sum(exp(logits))))^2
        """
        if router_logits.dim() == 3:
            router_logits = router_logits.view(-1, router_logits.shape[-1])

        # Log-sum-exp of router logits
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        return z_loss

    def compute_loss(
        self,
        moe_output: torch.Tensor,
        target: torch.Tensor,
        aux_info: dict = None,
        return_metrics: bool = False,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            moe_output: Output from MoE layer
            target: Target output from dense FFN
            aux_info: Auxiliary info from MoE forward (router_logits, expert_counts)
            return_metrics: If True, return (loss, metrics_dict)

        Returns:
            Combined loss, or (loss, metrics) if return_metrics=True
        """
        # MSE loss
        mse_loss = F.mse_loss(moe_output, target)

        # Flatten to (batch * seq, d_model) for cosine similarity
        moe_flat = moe_output.view(-1, moe_output.shape[-1])
        target_flat = target.view(-1, target.shape[-1])

        # Cosine similarity (1 = identical, -1 = opposite)
        cos_sim = F.cosine_similarity(moe_flat, target_flat, dim=-1).mean()

        # Start with MSE
        loss = mse_loss

        # Optional cosine similarity loss
        if self.config.use_cosine_loss:
            cos_loss = 1 - cos_sim
            loss = loss + self.config.cosine_loss_weight * cos_loss

        # Auxiliary losses for routing
        lb_loss = torch.tensor(0.0, device=moe_output.device)
        z_loss = torch.tensor(0.0, device=moe_output.device)

        if aux_info is not None:
            if self.config.use_load_balancing and 'router_logits' in aux_info:
                lb_loss = self.compute_load_balancing_loss(
                    aux_info['router_logits'],
                    aux_info['expert_counts'],
                )
                loss = loss + self.config.load_balancing_weight * lb_loss

            if self.config.use_z_loss and 'router_logits' in aux_info:
                z_loss = self.compute_z_loss(aux_info['router_logits'])
                loss = loss + self.config.z_loss_weight * z_loss

        if return_metrics:
            # Compute relative error
            rel_error = (moe_output - target).norm() / (target.norm() + 1e-8)

            metrics = {
                'mse': mse_loss.item(),
                'cos_sim': cos_sim.item(),
                'rel_error': rel_error.item(),
                'lb_loss': lb_loss.item() if isinstance(lb_loss, torch.Tensor) else lb_loss,
                'z_loss': z_loss.item() if isinstance(z_loss, torch.Tensor) else z_loss,
            }

            # Add routing statistics if available
            if aux_info is not None and 'expert_counts' in aux_info:
                counts = aux_info['expert_counts']
                total = counts.sum().item()
                if total > 0:
                    fracs = counts / total
                    metrics['expert_usage_std'] = fracs.std().item()
                    metrics['expert_usage_min'] = fracs.min().item()
                    metrics['expert_usage_max'] = fracs.max().item()
                    # Detect collapse: if one expert gets > 50% of tokens
                    metrics['max_expert_frac'] = fracs.max().item()

            return loss, metrics

        return loss

    def train_step(
        self,
        activations: Dict[int, FFNActivation],
        return_metrics: bool = False,
    ) -> Dict[int, float]:
        """
        Perform one training step on all MoE layers.

        Args:
            activations: FFN activations from dense model forward pass
            return_metrics: If True, return detailed metrics

        Returns:
            Dict mapping layer_idx to loss value (or metrics dict if return_metrics)
        """
        losses = {}
        all_metrics = {} if return_metrics else None

        for layer_idx, act in activations.items():
            moe_layer = self.moe_layers[str(layer_idx)]
            optimizer = self._get_optimizer(layer_idx)

            # Move to device (activations are already detached)
            x = act.input.to(self.device)
            target = act.output.to(self.device)

            # Forward through MoE
            moe_layer.train()
            moe_output = moe_layer(x)

            # Compute loss
            if return_metrics:
                loss, metrics = self.compute_loss(moe_output, target, return_metrics=True)
                all_metrics[layer_idx] = metrics
            else:
                loss = self.compute_loss(moe_output, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    moe_layer.parameters(),
                    self.config.max_grad_norm,
                )

            # Capture loss before stepping
            loss_val = loss.item()

            # Update
            optimizer.step()

            # Clear intermediate tensors to free memory
            del moe_output, loss, x, target
            torch.cuda.empty_cache()

            losses[layer_idx] = loss_val
            self.layer_losses[layer_idx].append(loss_val)

        if return_metrics:
            return losses, all_metrics
        return losses

    def distill(
        self,
        dataloader: Optional[DataLoader] = None,
        epochs: int = 1,
        max_steps: Optional[int] = None,
    ) -> Dict[int, List[float]]:
        """
        Run distillation training.

        Args:
            dataloader: DataLoader providing input_ids (optional if data_free=True)
            epochs: Number of epochs to train
            max_steps: Optional maximum number of steps (overrides epochs)

        Returns:
            Dict mapping layer_idx to list of losses
        """
        # Data-free distillation: use random inputs instead of real data
        # Pass dataloader for clustering (supervised routing needs real data for good clusters)
        if self.config.data_free:
            return self._distill_data_free(dataloader=dataloader)

        if dataloader is None:
            raise ValueError("dataloader required when data_free=False")

        if self.config.sequential_layers:
            return self._distill_sequential(dataloader, epochs, max_steps)

        if self.config.empirical_routing:
            return self._distill_empirical_routing(dataloader, epochs, max_steps)

        if self.config.dynamic_experts:
            if not DYNAMIC_EXPERTS_AVAILABLE:
                raise ImportError("Dynamic experts module not available")
            return self._distill_dynamic(dataloader, epochs, max_steps)

        print(f"\nStarting sparse distillation (parallel mode)...")
        print(f"  Layers: {self.extractor.n_layers}")
        if self.layer_config:
            n_experts_list = [self.moe_layers[str(i)].n_experts for i in self.extractor.layer_indices]
            print(f"  Experts per layer: {min(n_experts_list)}-{max(n_experts_list)} (variable per layer)")
        else:
            print(f"  Experts per layer: {self.config.n_experts}")
        print(f"  Routing: {self.config.moe_routing}")
        print(f"  Epochs: {epochs}")
        if self.config.supervised_routing:
            print(f"  Supervised routing: ON")
        print(f"  WARNING: Training all layers in parallel - high memory usage!")
        print(f"  TIP: Set sequential_layers=True to train one layer at a time")
        print()

        # Phase 1: Supervised router training (if enabled)
        if self.config.supervised_routing:
            print("=" * 50)
            print("PHASE 1: Supervised Router Training")
            print("=" * 50)

            for layer_idx in self.extractor.layer_indices:
                moe_layer = self.moe_layers[str(layer_idx)]
                n_experts = moe_layer.n_experts  # Use layer's actual n_experts
                print(f"\nLayer {layer_idx} (n_experts={n_experts}):")
                ffn = self.extractor.cache._ffn_modules[layer_idx]

                # Collect inputs and outputs from real data
                ffn_inputs, ffn_outputs = self._collect_ffn_activations_from_data(
                    dataloader, layer_idx,
                    max_samples=self.config.cluster_samples,
                )

                # Cluster outputs using layer's n_experts
                centroids = self._cluster_outputs(ffn_outputs, n_experts)

                # Train router using real inputs
                self._train_router_supervised(
                    moe_layer, ffn, centroids,
                    steps=self.config.router_train_steps,
                    ffn_inputs=ffn_inputs,
                    ffn_outputs=ffn_outputs,
                )

            print("\n" + "=" * 50)
            print("PHASE 2: Expert Training (routers frozen)")
            print("=" * 50)

            # Recreate optimizers to exclude router params
            self.optimizers = {}  # Clear existing optimizers

        total_steps = 0

        for epoch in range(epochs):
            epoch_losses = {idx: [] for idx in self.extractor.layer_indices}

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                # Extract input_ids from batch
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids', batch.get('tokens'))
                    attention_mask = batch.get('attention_mask', None)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0]
                    attention_mask = batch[1] if len(batch) > 1 else None
                else:
                    input_ids = batch
                    attention_mask = None

                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Extract activations from dense model
                activations = self.extractor.extract(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Train all MoE layers (get metrics every log_interval)
                get_metrics = (total_steps + 1) % self.config.log_interval == 0
                result = self.train_step(activations, return_metrics=get_metrics)

                if get_metrics:
                    losses, metrics = result
                else:
                    losses = result

                # Update epoch losses
                for layer_idx, loss in losses.items():
                    epoch_losses[layer_idx].append(loss)

                # Logging
                total_steps += 1
                self.global_step = total_steps

                if total_steps % self.config.log_interval == 0:
                    avg_loss = sum(losses.values()) / len(losses)
                    # Average cosine similarity across layers
                    avg_cos_sim = sum(m['cos_sim'] for m in metrics.values()) / len(metrics)
                    avg_rel_err = sum(m['rel_error'] for m in metrics.values()) / len(metrics)
                    pbar.set_postfix({
                        'step': total_steps,
                        'loss': f'{avg_loss:.4f}',
                        'cos_sim': f'{avg_cos_sim:.3f}',
                    })
                    # Print detailed metrics periodically
                    if total_steps % (self.config.log_interval * 5) == 0:
                        tqdm.write(f"  Step {total_steps}: loss={avg_loss:.4f}, cos_sim={avg_cos_sim:.4f}, rel_err={avg_rel_err:.4f}")

                # Check max steps
                if max_steps and total_steps >= max_steps:
                    break

            # Epoch summary
            print(f"\nEpoch {epoch+1} summary:")
            for layer_idx in sorted(epoch_losses.keys()):
                avg_loss = sum(epoch_losses[layer_idx]) / len(epoch_losses[layer_idx])
                print(f"  Layer {layer_idx}: avg_loss = {avg_loss:.6f}")

            # Print routing stats for all layers
            print(f"\nRouting statistics:")
            for layer_idx in sorted(self.moe_layers.keys(), key=lambda x: int(x)):
                print_layer_routing_stats(self.moe_layers[layer_idx], int(layer_idx), prefix="  ")

            if max_steps and total_steps >= max_steps:
                break

        return self.layer_losses

    def _distill_sequential(
        self,
        dataloader: DataLoader,
        epochs: int = 1,
        max_steps: Optional[int] = None,
    ) -> Dict[int, List[float]]:
        """
        Train MoE layers one at a time (memory-efficient mode).

        Only keeps one MoE layer + its optimizer in GPU memory at a time.
        """
        print(f"\nStarting sparse distillation (SEQUENTIAL mode)...")
        print(f"  Layers: {self.extractor.n_layers}")
        if self.layer_config:
            n_experts_list = [self.moe_layers[str(i)].n_experts for i in self.extractor.layer_indices]
            print(f"  Experts per layer: {min(n_experts_list)}-{max(n_experts_list)} (variable per layer)")
        else:
            print(f"  Experts per layer: {self.config.n_experts}")
        print(f"  Routing: {self.config.moe_routing}")
        print(f"  Epochs per layer: {epochs}")
        print(f"  Memory: Only 1 MoE layer in GPU at a time")
        if self.config.supervised_routing:
            print(f"  Supervised routing: ON")
        print()

        # Move all MoE layers to CPU first
        for layer_idx in self.extractor.layer_indices:
            self.moe_layers[str(layer_idx)] = self.moe_layers[str(layer_idx)].cpu()
        torch.cuda.empty_cache()

        for layer_idx in self.extractor.layer_indices:
            # Move this layer to GPU
            moe_layer = self.moe_layers[str(layer_idx)].to(self.device)
            self.moe_layers[str(layer_idx)] = moe_layer
            n_experts = moe_layer.n_experts  # Use layer's actual n_experts

            print(f"\n{'='*50}")
            print(f"Training MoE layer {layer_idx} (n_experts={n_experts})")
            print(f"{'='*50}")

            ffn = self.extractor.cache._ffn_modules[layer_idx]

            # Phase 1: Supervised router training (if enabled)
            if self.config.supervised_routing:
                print(f"  Phase 1: Supervised router training")

                # Collect inputs and outputs from real data
                ffn_inputs, ffn_outputs = self._collect_ffn_activations_from_data(
                    dataloader, layer_idx,
                    max_samples=self.config.cluster_samples,
                )

                # Cluster outputs using layer's n_experts
                centroids = self._cluster_outputs(ffn_outputs, n_experts)

                # Train router using real inputs
                self._train_router_supervised(
                    moe_layer, ffn, centroids,
                    steps=self.config.router_train_steps,
                    ffn_inputs=ffn_inputs,
                    ffn_outputs=ffn_outputs,
                )
                print(f"  Phase 2: Expert training (router frozen)")

            # Create optimizer for this layer only
            if self.config.supervised_routing:
                # Exclude router params
                params = [p for n, p in moe_layer.named_parameters() if 'router' not in n]
            else:
                params = list(moe_layer.parameters())

            optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            total_steps = 0
            for epoch in range(epochs):
                epoch_loss = []
                pbar = tqdm(dataloader, desc=f"Layer {layer_idx} Epoch {epoch+1}/{epochs}")

                for batch in pbar:
                    # Extract input_ids
                    if isinstance(batch, dict):
                        input_ids = batch.get('input_ids', batch.get('tokens'))
                        attention_mask = batch.get('attention_mask', None)
                    elif isinstance(batch, (list, tuple)):
                        input_ids = batch[0]
                        attention_mask = batch[1] if len(batch) > 1 else None
                    else:
                        input_ids = batch
                        attention_mask = None

                    input_ids = input_ids.to(self.device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    # Extract activations (only need this layer's activation)
                    activations = self.extractor.extract(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    if layer_idx not in activations:
                        continue

                    act = activations[layer_idx]
                    x = act.input.to(self.device)
                    target = act.output.to(self.device)

                    # Forward
                    moe_layer.train()
                    moe_output = moe_layer(x)

                    # Loss
                    loss = self.compute_loss(moe_output, target)
                    loss_val = loss.item()

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()

                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            moe_layer.parameters(),
                            self.config.max_grad_norm,
                        )

                    optimizer.step()

                    # Clean up
                    del moe_output, loss, x, target, activations

                    epoch_loss.append(loss_val)
                    self.layer_losses[layer_idx].append(loss_val)
                    total_steps += 1

                    if total_steps % self.config.log_interval == 0:
                        pbar.set_postfix({'loss': f'{loss_val:.4f}'})

                    if max_steps and total_steps >= max_steps:
                        break

                avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
                print(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.6f}")

                # Print routing stats for this layer
                print_layer_routing_stats(moe_layer, layer_idx, prefix="  ")

                if max_steps and total_steps >= max_steps:
                    break

            # Move trained layer back to CPU, free optimizer
            self.moe_layers[str(layer_idx)] = moe_layer.cpu()
            del optimizer
            torch.cuda.empty_cache()

        # Move all layers back to GPU for evaluation/saving
        print("\nMoving all trained layers to GPU...")
        for layer_idx in self.extractor.layer_indices:
            self.moe_layers[str(layer_idx)] = self.moe_layers[str(layer_idx)].to(self.device)

        return self.layer_losses

    def _collect_ffn_activations_from_data(
        self,
        dataloader: DataLoader,
        layer_idx: int,
        max_samples: int = 10000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect FFN inputs and outputs from real data for clustering and router training.

        Returns:
            inputs: (N, d_model) tensor of FFN inputs
            outputs: (N, d_model) tensor of FFN outputs
        """
        print(f"    Collecting FFN activations from real data...")

        all_inputs = []
        all_outputs = []
        total_collected = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting", leave=False):
                if total_collected >= max_samples:
                    break

                # Extract input_ids
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids', batch.get('tokens'))
                    attention_mask = batch.get('attention_mask', None)
                else:
                    input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                    attention_mask = None

                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Get activations for this layer
                activations = self.extractor.extract(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                if layer_idx in activations:
                    inp = activations[layer_idx].input  # (batch, seq, d_model)
                    output = activations[layer_idx].output  # (batch, seq, d_model)
                    # Flatten and move to CPU
                    inp_flat = inp.view(-1, self.d_model).float().cpu()
                    output_flat = output.view(-1, self.d_model).float().cpu()
                    all_inputs.append(inp_flat)
                    all_outputs.append(output_flat)
                    total_collected += output_flat.shape[0]

        if not all_outputs:
            raise ValueError(f"No activations collected for layer {layer_idx}")

        all_inputs = torch.cat(all_inputs, dim=0)[:max_samples]
        all_outputs = torch.cat(all_outputs, dim=0)[:max_samples]
        print(f"    Collected {all_outputs.shape[0]} activation pairs")
        return all_inputs, all_outputs

    def _kmeans_gpu(
        self,
        data: torch.Tensor,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        GPU-accelerated k-means clustering using PyTorch.

        Args:
            data: (N, D) tensor on GPU
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            centroids: (n_clusters, D) cluster centroids
            labels: (N,) cluster assignments
            inertia: sum of squared distances to centroids
        """
        N, D = data.shape

        # k-means++ initialization
        centroids = torch.zeros(n_clusters, D, device=data.device, dtype=data.dtype)
        # First centroid: random point
        idx = torch.randint(N, (1,)).item()
        centroids[0] = data[idx]

        for k in range(1, n_clusters):
            # Compute distances to nearest centroid
            dists = torch.cdist(data, centroids[:k])  # (N, k)
            min_dists = dists.min(dim=1).values  # (N,)
            # Sample proportional to squared distance
            probs = min_dists ** 2
            probs = probs / (probs.sum() + 1e-10)
            idx = torch.multinomial(probs, 1).item()
            centroids[k] = data[idx]

        # Iterate
        for iteration in range(max_iter):
            # Assign points to nearest centroid
            dists = torch.cdist(data, centroids)  # (N, n_clusters)
            labels = dists.argmin(dim=1)  # (N,)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = data[mask].mean(dim=0)
                else:
                    # Empty cluster: reinitialize to random point
                    idx = torch.randint(N, (1,)).item()
                    new_centroids[k] = data[idx]

            # Check convergence
            shift = (new_centroids - centroids).norm()
            centroids = new_centroids

            if shift < tol:
                print(f"    k-means converged at iteration {iteration}")
                break

        # Final assignment
        dists = torch.cdist(data, centroids)
        labels = dists.argmin(dim=1)

        # Compute inertia
        inertia = dists.min(dim=1).values.pow(2).sum().item()

        return centroids, labels, inertia

    def analyze_cluster_quality(
        self,
        outputs: torch.Tensor,
        k_values: List[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze clustering quality at different k values to find optimal n_experts.

        Args:
            outputs: (N, d_model) tensor of FFN outputs
            k_values: List of cluster counts to try (default: [2,4,8,16,32,64,128])

        Returns:
            Dict mapping k -> {inertia, silhouette, calinski_harabasz, cluster_balance}
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        from sklearn.cluster import KMeans

        # Try to import faiss
        try:
            import faiss
            use_faiss = True
        except ImportError:
            print("faiss not found, using sklearn (slower). TIP: pip install faiss-gpu")
            use_faiss = False

        if k_values is None:
            # Step by 2 for faster analysis: 4, 6, 8, ..., 64
            k_values = list(range(4, 65, 2))

        N, D = outputs.shape
        print(f"\n{'='*60}")
        print(f"CLUSTER QUALITY ANALYSIS (N={N}, D={D})")
        print(f"{'='*60}")

        # Subsample if too large (silhouette is O(N^2))
        max_samples = 50000
        if N > max_samples:
            print(f"Subsampling to {max_samples} for analysis...")
            indices = torch.randperm(N)[:max_samples]
            outputs = outputs[indices]
            N = max_samples

        # L2 normalize on GPU, then convert to numpy
        outputs_gpu = outputs.to(device=self.device, dtype=torch.float32)
        outputs_normalized = F.normalize(outputs_gpu, p=2, dim=1)
        # faiss requires float32 contiguous array
        data = np.ascontiguousarray(outputs_normalized.cpu().numpy(), dtype=np.float32)

        if use_faiss:
            gpu_id = 0 if self.device == 'cuda' else int(self.device.split(':')[-1]) if ':' in self.device else 0

        results = {}
        print(f"\n{'k':>6} | {'Inertia':>12} | {'Silhouette':>10} | {'CH Index':>12} | {'Balance':>10}")
        print("-" * 60)

        for k in k_values:
            if k >= N:
                continue

            if use_faiss:
                kmeans = faiss.Kmeans(
                    D, k,
                    niter=50,
                    verbose=False,
                    gpu=gpu_id,
                    spherical=True,
                    seed=42,
                )
                kmeans.train(data)
                distances, labels = kmeans.index.search(data, 1)
                labels = labels.flatten()
                inertia = distances.sum()
            else:
                kmeans = KMeans(n_clusters=k, init='k-means++', n_init=3, max_iter=50, random_state=42)
                kmeans.fit(data)
                labels = kmeans.labels_
                inertia = kmeans.inertia_

            # Cluster balance
            counts = np.array([(labels == i).sum() for i in range(k)])
            raw_balance = counts.min() / max(counts.max(), 1)

            # Normalize balance: compare to expected random assignment
            # For uniform random assignment, counts follow multinomial distribution
            # Expected: N/k per cluster, std ≈ sqrt(N * (1/k) * (1 - 1/k)) ≈ sqrt(N/k)
            # Expected min/max ratio for random assignment (approximation):
            # As k increases, expected balance decreases roughly as 1 - c*sqrt(k/N)
            # We use entropy-based normalized balance instead:
            # Entropy of uniform = log(k), entropy of actual = -sum(p*log(p))
            # Normalized entropy = actual_entropy / log(k) (1.0 = perfectly uniform)
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # Avoid log(0)
            actual_entropy = -np.sum(probs * np.log(probs))
            max_entropy = np.log(k)
            norm_balance = actual_entropy / max_entropy if max_entropy > 0 else 1.0

            # Silhouette score (only if k > 1 and < N)
            if k > 1 and len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data, labels, sample_size=min(10000, N))
                ch_score = calinski_harabasz_score(data, labels)
            else:
                silhouette = 0.0
                ch_score = 0.0

            results[k] = {
                'inertia': float(inertia),
                'silhouette': float(silhouette),
                'calinski_harabasz': float(ch_score),
                'balance': float(norm_balance),  # Normalized entropy-based balance
                'raw_balance': float(raw_balance),  # Original min/max ratio
                'min_cluster': int(counts.min()),
                'max_cluster': int(counts.max()),
            }

            print(f"{k:>6} | {inertia:>12.1f} | {silhouette:>10.4f} | {ch_score:>12.1f} | {norm_balance:>10.4f}")

        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS:")
        print(f"{'='*60}")

        # Best balance (primary metric for MoE)
        best_bal_k = max(results.keys(), key=lambda k: results[k]['balance'])
        print(f"  Best balance: k={best_bal_k} (balance={results[best_bal_k]['balance']:.4f})")

        # Top 5 by balance
        sorted_by_balance = sorted(results.keys(), key=lambda k: results[k]['balance'], reverse=True)[:5]
        print(f"  Top 5 by balance: {sorted_by_balance}")

        # Best silhouette (secondary)
        best_sil_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
        print(f"  Best silhouette: k={best_sil_k} (score={results[best_sil_k]['silhouette']:.4f})")

        # Best combined: high balance AND reasonable silhouette
        # Score = balance * (1 + silhouette) to favor balanced clusters with some structure
        combined_scores = {k: v['balance'] * (1 + v['silhouette']) for k, v in results.items()}
        best_combined_k = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        print(f"  Best combined (balance × (1+silhouette)): k={best_combined_k}")

        print(f"{'='*60}\n")

        return results

    def _cluster_outputs(self, outputs: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Cluster outputs using faiss GPU k-means (with sklearn fallback).

        Args:
            outputs: (N, d_model) tensor of outputs
            n_clusters: Number of clusters

        Returns:
            centroids: (n_clusters, d_model) cluster centroids
        """
        N, D = outputs.shape

        # Subsample if too large (consistent with analyze_cluster_quality)
        max_samples = 50000
        if N > max_samples:
            print(f"    Subsampling {N} -> {max_samples} for clustering...")
            indices = torch.randperm(N)[:max_samples]
            outputs = outputs[indices]
            N = max_samples

        # L2 normalize on GPU
        outputs_gpu = outputs.to(device=self.device, dtype=torch.float32)
        outputs_normalized = F.normalize(outputs_gpu, p=2, dim=1)

        # Convert to numpy for clustering (faiss requires float32 contiguous array)
        data = np.ascontiguousarray(outputs_normalized.cpu().numpy(), dtype=np.float32)

        try:
            import faiss
            print(f"    Running faiss k-means with {n_clusters} clusters (N={N}, D={D})...")
            print(f"    Data stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")

            # Setup GPU k-means (same params as analyze_cluster_quality)
            gpu_id = 0 if self.device == 'cuda' else int(self.device.split(':')[-1]) if ':' in self.device else 0
            kmeans = faiss.Kmeans(
                D,
                n_clusters,
                niter=50,  # Match analyze_cluster_quality
                verbose=True,
                gpu=gpu_id,
                spherical=True,
                seed=42,
            )
            kmeans.train(data)

            # Get assignments
            _, labels = kmeans.index.search(data, 1)
            labels = labels.flatten()
            centroids_np = kmeans.centroids

        except ImportError:
            print(f"    faiss not found, using sklearn k-means (slower)...")
            print(f"    TIP: pip install faiss-gpu  (or faiss-cpu)")
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=100,
                random_state=42,
            )
            kmeans.fit(data)
            labels = kmeans.labels_
            centroids_np = kmeans.cluster_centers_

        # Get centroids
        centroids = torch.tensor(centroids_np, dtype=self.dtype, device=self.device)

        # Print cluster balance with detailed stats
        counts = np.array([(labels == i).sum() for i in range(n_clusters)])
        min_count, max_count = int(counts.min()), int(counts.max())

        # Entropy-based balance (same as analyze_cluster_quality)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        actual_entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(n_clusters)
        balance = actual_entropy / max_entropy if max_entropy > 0 else 1.0

        print(f"    Cluster sizes: min={min_count}, max={max_count}, ratio={max_count/max(min_count,1):.1f}x")
        print(f"    Balance (entropy): {balance:.4f} (1.0 = perfect)")
        print(f"    Distribution: {sorted(counts)[:5]}...{sorted(counts)[-5:]}")

        if balance < 0.9:
            print(f"    WARNING: Imbalanced clusters (balance < 0.9)")

        return centroids

    def _cluster_ffn_outputs(self, ffn: nn.Module, n_clusters: int) -> torch.Tensor:
        """
        Collect FFN outputs and cluster them to create expert specializations.
        Used in data-free mode.

        Returns:
            centroids: (n_clusters, d_model) cluster centroids
        """
        print(f"    Collecting {self.config.cluster_samples} samples for clustering...")

        all_outputs = []
        n_batches = self.config.cluster_samples // self.config.cluster_batch_size

        with torch.no_grad():
            for _ in tqdm(range(n_batches), desc="Collecting", leave=False):
                # Generate random input
                x = torch.randn(
                    self.config.cluster_batch_size,
                    self.config.data_free_seq_len,
                    self.d_model,
                    device=self.device,
                    dtype=self.dtype,
                )
                x = F.normalize(x, dim=-1) * math.sqrt(self.d_model)

                # Get FFN output
                output = ffn(x)  # (batch, seq, d_model)
                # Flatten to (batch * seq, d_model)
                output_flat = output.view(-1, self.d_model).float().cpu()
                all_outputs.append(output_flat)

        # Concatenate all outputs
        all_outputs = torch.cat(all_outputs, dim=0)  # (N, d_model)
        print(f"    Collected {all_outputs.shape[0]} output vectors")

        # Use shared clustering function
        return self._cluster_outputs(all_outputs, n_clusters)

    def _train_router_supervised(
        self,
        moe_layer: nn.Module,
        ffn: nn.Module,
        centroids: torch.Tensor,
        steps: int = 1000,
        ffn_inputs: Optional[torch.Tensor] = None,
        ffn_outputs: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Train router with supervised learning based on cluster assignments.

        For each input x:
        1. Compute FFN output y = FFN(x)
        2. Find distances to all centroids
        3. Target = softmax(-distances) (soft assignment)
        4. Train router to predict this distribution

        Args:
            moe_layer: The MoE layer being trained
            ffn: The dense FFN layer
            centroids: Cluster centroids (n_experts, d_model)
            steps: Number of training steps
            ffn_inputs: Pre-collected FFN inputs for real data mode (N, d_model)
            ffn_outputs: Pre-collected FFN outputs for real data mode (N, d_model)
        """
        print(f"    Training router with supervised clustering ({steps} steps)...")

        use_real_data = ffn_inputs is not None and ffn_outputs is not None
        if use_real_data:
            print(f"    Using {ffn_inputs.shape[0]} real data samples")
            # Move to device
            ffn_inputs = ffn_inputs.to(device=self.device, dtype=self.dtype)
            ffn_outputs = ffn_outputs.to(device=self.device, dtype=self.dtype)
            n_samples = ffn_inputs.shape[0]

        # Only train router parameters
        router_params = list(moe_layer.router.parameters())
        optimizer = torch.optim.AdamW(router_params, lr=self.config.learning_rate * 10)  # Higher LR for router

        pbar = tqdm(range(steps), desc="Router training", leave=False)
        recent_losses = []
        recent_acc = []

        batch_size = self.config.data_free_batch_size * self.config.data_free_seq_len

        for step in pbar:
            if use_real_data:
                # Sample from real data
                indices = torch.randint(0, n_samples, (batch_size,))
                x = ffn_inputs[indices]  # (batch_size, d_model)
                ffn_output = ffn_outputs[indices]  # (batch_size, d_model)
            else:
                # Generate random input (data-free mode)
                x = torch.randn(
                    self.config.data_free_batch_size,
                    self.config.data_free_seq_len,
                    self.d_model,
                    device=self.device,
                    dtype=self.dtype,
                )
                x = F.normalize(x, dim=-1) * math.sqrt(self.d_model)

                # Get FFN output
                with torch.no_grad():
                    ffn_output = ffn(x)  # (batch, seq, d_model)
                ffn_output = ffn_output.view(-1, self.d_model)
                x = x.view(-1, self.d_model)

            # Compute distances to centroids
            distances = torch.cdist(ffn_output.float(), centroids.float())  # (batch_size, n_experts)

            # Target distribution: softmax of negative distances (closer = higher prob)
            temperature = 0.1
            target_probs = F.softmax(-distances / temperature, dim=-1)

            # Get router logits - need to add seq dim for router
            x_for_router = x.unsqueeze(1)  # (batch_size, 1, d_model)
            router_logits = moe_layer.get_router_logits(x_for_router)  # (batch_size, 1, n_experts)
            n_experts = moe_layer.n_experts  # Use layer's actual n_experts
            router_logits_flat = router_logits.view(-1, n_experts)

            # Cross-entropy loss (KL divergence)
            router_probs = F.log_softmax(router_logits_flat, dim=-1)
            loss = F.kl_div(router_probs, target_probs, reduction='batchmean')

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            loss_val = loss.item()
            recent_losses.append(loss_val)

            # Compute top-k accuracy
            top_k = min(self.config.moe_top_k, n_experts)
            _, target_topk = distances.topk(top_k, dim=-1, largest=False)
            _, router_topk = router_logits_flat.topk(top_k, dim=-1)
            # Check overlap
            matches = sum((target_topk == router_topk[:, i:i+1]).any(dim=-1).float().mean().item()
                         for i in range(self.config.moe_top_k)) / self.config.moe_top_k
            recent_acc.append(matches)

            if len(recent_losses) > 100:
                recent_losses.pop(0)
                recent_acc.pop(0)

            if (step + 1) % 100 == 0:
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_acc = sum(recent_acc) / len(recent_acc)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.2%}'})

        final_loss = sum(recent_losses) / len(recent_losses)
        final_acc = sum(recent_acc) / len(recent_acc)
        print(f"    Router training done: loss={final_loss:.4f}, top-{self.config.moe_top_k} acc={final_acc:.2%}")

    def _compute_expert_losses(
        self,
        moe_layer: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run all experts on input and compute per-expert losses.

        Args:
            moe_layer: MoE layer with experts
            x: Input tensor (batch, d_model) - flattened tokens
            target: Target tensor (batch, d_model) - dense FFN output

        Returns:
            expert_losses: (batch, n_experts) per-sample per-expert MSE losses
        """
        n_experts = moe_layer.n_experts
        batch_size = x.shape[0]

        expert_losses = torch.zeros(batch_size, n_experts, device=x.device)

        with torch.no_grad():
            for expert_idx in range(n_experts):
                # Get expert weights (w1, w2, w3 for SwiGLU)
                # x: (batch, d_model), expert weights: (d_model, d_ff) for w1/w2
                experts = moe_layer.experts

                if hasattr(experts, 'w1'):
                    # Grouped experts format: w1[expert_idx] is (d_model, d_ff)
                    w1 = experts.w1[expert_idx]  # (d_model, d_ff)
                    w2 = experts.w2[expert_idx]  # (d_model, d_ff)
                    w3 = experts.w3[expert_idx]  # (d_ff, d_model)

                    # SwiGLU: output = (silu(x @ w1) * (x @ w2)) @ w3
                    gate = F.silu(x @ w1)  # (batch, d_ff)
                    up = x @ w2  # (batch, d_ff)
                    expert_out = (gate * up) @ w3  # (batch, d_model)
                else:
                    raise ValueError("Unsupported expert format")

                # Per-sample MSE loss
                losses = ((expert_out - target) ** 2).mean(dim=-1)  # (batch,)
                expert_losses[:, expert_idx] = losses

        return expert_losses

    def _balanced_topk_assignment(
        self,
        routing_signal: torch.Tensor,
        top_k: int,
        n_experts: int,
    ) -> torch.Tensor:
        """
        Assign top-k experts per token with global capacity constraints.

        Unlike pure top-k (which allows one expert to dominate), this ensures
        each expert gets roughly equal number of tokens across the batch.

        Uses vectorized greedy assignment with capacity constraints.
        Complexity: O(top_k) iterations instead of O(n_tokens * top_k).

        Args:
            routing_signal: (n_tokens, n_experts) quality scores (higher = better)
            top_k: number of experts per token
            n_experts: total number of experts

        Returns:
            selected_experts: (n_tokens, top_k) indices of selected experts
        """
        n_tokens = routing_signal.shape[0]
        device = routing_signal.device

        # Capacity per expert: total assignments / n_experts
        capacity_per_expert = max((n_tokens * top_k) // n_experts, 1)

        # Output tensor
        selected_experts = torch.zeros(n_tokens, top_k, dtype=torch.long, device=device)

        # Work with a copy we can modify
        scores = routing_signal.clone()

        # Track capacity per expert
        expert_counts = torch.zeros(n_experts, dtype=torch.long, device=device)

        for k in range(top_k):
            # Create capacity mask: experts at capacity get -inf
            at_capacity = expert_counts >= capacity_per_expert
            masked_scores = scores.clone()
            masked_scores[:, at_capacity] = float('-inf')

            # Each token picks its best available expert
            best_scores, best_experts = masked_scores.max(dim=-1)

            # Handle edge case: all experts at capacity for some tokens
            needs_fallback = best_scores == float('-inf')
            if needs_fallback.any():
                # For these tokens, just pick from original scores
                best_experts[needs_fallback] = routing_signal[needs_fallback].argmax(dim=-1)

            # Store selections
            selected_experts[:, k] = best_experts

            # Update expert counts (vectorized)
            expert_counts.scatter_add_(0, best_experts, torch.ones_like(best_experts, dtype=torch.long))

            # Mask out selected experts for next k (so tokens don't pick same expert twice)
            scores.scatter_(1, best_experts.unsqueeze(-1), float('-inf'))

        return selected_experts

    def _compute_expert_gradient_norms(
        self,
        moe_layer: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expert quality signal using per-sample losses.

        This is the "exact" method - computes MSE loss per expert per token.
        Returns negative normalized loss so higher = better expert for token.

        Note: The "fast" method is preferred as it's much faster and equivalent.

        Args:
            moe_layer: MoE layer with experts
            x: Input tensor (batch, d_model) - flattened tokens
            target: Target tensor (batch, d_model) - dense FFN output

        Returns:
            routing_signal: (batch, n_experts) - higher = better expert for token
        """
        n_experts = moe_layer.n_experts
        batch_size = x.shape[0]
        experts = moe_layer.experts

        expert_losses = torch.zeros(batch_size, n_experts, device=x.device)

        with torch.no_grad():
            for expert_idx in range(n_experts):
                if hasattr(experts, 'w1'):
                    w1 = experts.w1[expert_idx]
                    w2 = experts.w2[expert_idx]
                    w3 = experts.w3[expert_idx]

                    # Forward
                    gate = F.silu(x @ w1)
                    up = x @ w2
                    expert_out = (gate * up) @ w3

                    # Per-sample MSE loss
                    per_sample_loss = ((expert_out - target) ** 2).mean(dim=-1)
                    expert_losses[:, expert_idx] = per_sample_loss
                else:
                    raise ValueError("Unsupported expert format")

            # Normalize per token (z-score)
            mean = expert_losses.mean(dim=-1, keepdim=True)
            std = expert_losses.std(dim=-1, keepdim=True).clamp(min=1e-6)
            normalized_losses = (expert_losses - mean) / std

            # Return NEGATIVE loss so that lower loss = higher routing signal
            routing_signal = -normalized_losses

        return routing_signal

    def _compute_expert_gradient_norms_fast(
        self,
        moe_layer: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast approximation of expert quality using error magnitude.

        We compute per-expert error and use NEGATIVE error as routing signal.
        Lower error = better expert for this token = should be selected.

        This is essentially loss-based routing but computed efficiently.

        Args:
            moe_layer: MoE layer with experts
            x: Input tensor (batch, d_model) - flattened tokens
            target: Target tensor (batch, d_model) - dense FFN output

        Returns:
            routing_signal: (batch, n_experts) - higher = better expert for token
        """
        n_experts = moe_layer.n_experts
        batch_size = x.shape[0]
        experts = moe_layer.experts

        expert_errors = torch.zeros(batch_size, n_experts, device=x.device)

        with torch.no_grad():
            for expert_idx in range(n_experts):
                if hasattr(experts, 'w1'):
                    w1 = experts.w1[expert_idx]
                    w2 = experts.w2[expert_idx]
                    w3 = experts.w3[expert_idx]

                    # Forward
                    gate = F.silu(x @ w1)
                    up = x @ w2
                    expert_out = (gate * up) @ w3

                    # Error magnitude (lower = better)
                    error = (expert_out - target).norm(dim=-1)
                    expert_errors[:, expert_idx] = error
                else:
                    raise ValueError("Unsupported expert format")

            # Normalize per token (z-score) to prevent scale issues
            mean = expert_errors.mean(dim=-1, keepdim=True)
            std = expert_errors.std(dim=-1, keepdim=True).clamp(min=1e-6)
            normalized_errors = (expert_errors - mean) / std

            # Return NEGATIVE error so that lower error = higher routing signal
            routing_signal = -normalized_errors

        return routing_signal

    def _distill_empirical_routing(
        self,
        dataloader: DataLoader,
        epochs: int = 1,
        max_steps: Optional[int] = None,
    ) -> Dict[int, List[float]]:
        """
        Train MoE layers with empirical routing.

        Instead of clustering, we:
        1. Run ALL experts on each input
        2. Compute per-expert losses against FFN target
        3. Train router to predict best experts (softmax over -loss)
        4. Train experts jointly

        This provides ground-truth routing labels based on actual performance.
        """
        print(f"\nStarting sparse distillation (EMPIRICAL ROUTING mode)...")
        print(f"  Layers: {self.extractor.n_layers}")
        if self.layer_config:
            n_experts_list = [self.moe_layers[str(i)].n_experts for i in self.extractor.layer_indices]
            print(f"  Experts per layer: {min(n_experts_list)}-{max(n_experts_list)} (variable per layer)")
        else:
            print(f"  Experts per layer: {self.config.n_experts}")
        print(f"  Routing: {self.config.moe_routing}")
        print(f"  Empirical eval interval: {self.config.empirical_eval_interval}")
        print(f"  Temperature: {self.config.empirical_temperature}")
        print(f"  Entropy weight: {self.config.empirical_entropy_weight}")
        if self.config.curriculum_routing:
            print(f"  Curriculum routing: ON")
            print(f"    Warmup steps: {self.config.curriculum_warmup_steps}")
            print(f"    Initial balance: {self.config.curriculum_initial_balance} (0=empirical, 1=uniform)")
            print(f"    Final balance: {self.config.curriculum_final_balance}")
            print(f"    Max target prob: {self.config.target_max_prob} (clips dominant experts)")
            if self.config.target_temperature_schedule:
                print(f"    Temperature: {self.config.target_initial_temperature} -> {self.config.target_final_temperature}")
            if self.config.detach_router_from_expert_loss:
                print(f"    Router isolation: ON (expert loss doesn't train router)")
            print(f"    Router loss weight: {self.config.router_loss_weight}")
        else:
            print(f"  Curriculum routing: OFF")
        if self.config.use_expert_bias:
            print(f"  Expert bias: ON (DeepSeek-style learnable per-expert bias)")
        if self.config.gradient_routing:
            mode = "fast (error magnitude)" if self.config.gradient_routing_fast else "exact (per-sample gradients)"
            print(f"  Gradient routing: ON ({mode})")
            print(f"    Two-pass training:")
            print(f"      Pass 1: Dense forward (all experts) -> compute expert errors")
            print(f"      Pass 2: Sparse forward (top-{self.config.moe_top_k} by quality) -> train")
            print(f"    Routes to experts with LOWEST error (best fit for each token)")
        print(f"  Epochs: {epochs}")
        print()

        # Move all MoE layers to CPU first for sequential training
        for layer_idx in self.extractor.layer_indices:
            self.moe_layers[str(layer_idx)] = self.moe_layers[str(layer_idx)].cpu()
        torch.cuda.empty_cache()

        for layer_idx in self.extractor.layer_indices:
            # Move this layer to GPU
            moe_layer = self.moe_layers[str(layer_idx)].to(self.device)
            self.moe_layers[str(layer_idx)] = moe_layer
            n_experts = moe_layer.n_experts

            print(f"\n{'='*60}")
            print(f"Training MoE layer {layer_idx} (n_experts={n_experts}) with empirical routing")
            print(f"{'='*60}")

            ffn = self.extractor.cache._ffn_modules[layer_idx]

            # Create joint optimizer for router + experts
            optimizer = torch.optim.AdamW(
                moe_layer.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            total_steps = 0

            for epoch in range(epochs):
                epoch_losses = []
                epoch_router_losses = []
                epoch_expert_losses = []

                pbar = tqdm(dataloader, desc=f"Layer {layer_idx} Epoch {epoch+1}/{epochs}")

                for batch in pbar:
                    # Extract input_ids
                    if isinstance(batch, dict):
                        input_ids = batch.get('input_ids', batch.get('tokens'))
                        attention_mask = batch.get('attention_mask', None)
                    elif isinstance(batch, (list, tuple)):
                        input_ids = batch[0]
                        attention_mask = batch[1] if len(batch) > 1 else None
                    else:
                        input_ids = batch
                        attention_mask = None

                    input_ids = input_ids.to(self.device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    # Extract activations
                    activations = self.extractor.extract(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    if layer_idx not in activations:
                        continue

                    act = activations[layer_idx]
                    x = act.input.to(self.device)  # (batch, seq, d_model)
                    target = act.output.to(self.device)  # (batch, seq, d_model)

                    batch_size, seq_len, d_model = x.shape
                    x_flat = x.view(-1, d_model)  # (batch * seq, d_model)
                    target_flat = target.view(-1, d_model)

                    # ============================================================
                    # TWO-PASS TRAINING:
                    # Pass 1 (Dense): Run all experts, compute gradient norms
                    # Pass 2 (Sparse): Run only top-k experts, train them
                    # ============================================================

                    moe_layer.train()
                    top_k = self.config.moe_top_k

                    # --- PASS 1: Dense forward to compute expert quality ---
                    if self.config.gradient_routing:
                        if self.config.gradient_routing_fast:
                            routing_signal = self._compute_expert_gradient_norms_fast(
                                moe_layer, x_flat, target_flat
                            )
                        else:
                            routing_signal = self._compute_expert_gradient_norms(
                                moe_layer, x_flat, target_flat
                            )
                        # Returns negative normalized error: higher = better expert
                    else:
                        # Loss-based fallback (already uses torch.no_grad internally)
                        expert_losses = self._compute_expert_losses(
                            moe_layer, x_flat, target_flat
                        )
                        # Normalize per token (z-score) for consistent scale
                        mean = expert_losses.mean(dim=-1, keepdim=True)
                        std = expert_losses.std(dim=-1, keepdim=True).clamp(min=1e-6)
                        normalized_losses = (expert_losses - mean) / std
                        # Lower loss = higher priority (negate)
                        routing_signal = -normalized_losses

                    # Select top-k experts per token with BALANCED assignment
                    # This prevents one expert from dominating all tokens
                    selected_experts = self._balanced_topk_assignment(
                        routing_signal, top_k, n_experts
                    )  # (n_tokens, top_k)

                    # Create hard routing targets from selection
                    hard_targets = torch.zeros_like(routing_signal)
                    hard_targets.scatter_(-1, selected_experts, 1.0 / top_k)

                    # Apply curriculum: blend hard targets with uniform
                    if self.config.curriculum_routing:
                        progress = min(1.0, total_steps / self.config.curriculum_warmup_steps)
                        balance_weight = (
                            self.config.curriculum_initial_balance +
                            progress * (self.config.curriculum_final_balance - self.config.curriculum_initial_balance)
                        )
                        uniform_targets = torch.ones_like(hard_targets) / n_experts
                        routing_targets = balance_weight * uniform_targets + (1 - balance_weight) * hard_targets
                    else:
                        routing_targets = hard_targets

                    # --- PASS 2: Sparse forward with only selected experts ---
                    # Compute weighted output from top-k experts only (BATCHED by expert)
                    experts = moe_layer.experts
                    n_tokens = x_flat.shape[0]

                    # Compute routing weights (softmax over selected experts' signals)
                    selected_signals = torch.gather(routing_signal, -1, selected_experts)
                    routing_weights = F.softmax(selected_signals, dim=-1)  # (n_tokens, top_k)

                    # Initialize output and usage tracking
                    sparse_output = torch.zeros_like(x_flat)
                    expert_usage = torch.zeros(n_experts, device=x.device)

                    # Process experts in batches (O(n_experts) instead of O(n_tokens * top_k))
                    for expert_idx in range(n_experts):
                        # Find all (token, k) pairs where this expert was selected
                        # selected_experts is (n_tokens, top_k)
                        mask = (selected_experts == expert_idx)  # (n_tokens, top_k)

                        if not mask.any():
                            continue

                        # Get token indices and k indices where this expert is used
                        token_indices, k_indices = torch.where(mask)

                        # Get inputs for these tokens
                        expert_inputs = x_flat[token_indices]  # (n_selected, d_model)

                        # Get weights for these tokens
                        weights = routing_weights[token_indices, k_indices]  # (n_selected,)

                        # Batched forward through this expert
                        w1 = experts.w1[expert_idx]
                        w2 = experts.w2[expert_idx]
                        w3 = experts.w3[expert_idx]

                        gate = F.silu(expert_inputs @ w1)  # (n_selected, d_ff)
                        up = expert_inputs @ w2  # (n_selected, d_ff)
                        expert_out = (gate * up) @ w3  # (n_selected, d_model)

                        # Weighted outputs (ensure same dtype as input)
                        weighted_out = (weights.unsqueeze(-1) * expert_out).to(sparse_output.dtype)

                        # Scatter-add to sparse_output
                        sparse_output.index_add_(0, token_indices, weighted_out)

                        # Track usage
                        expert_usage[expert_idx] = len(token_indices)

                    # Reshape output
                    sparse_output = sparse_output.view(batch_size, seq_len, d_model)

                    # Cache expert usage for stats
                    moe_layer._last_expert_counts = expert_usage.detach()
                    moe_layer._last_router_logits = routing_signal.view(batch_size, seq_len, n_experts).detach()

                    # Debug: print expert usage on first batch
                    if total_steps == 0:
                        usage_pct = (expert_usage / expert_usage.sum() * 100).cpu().numpy()
                        usage_str = " ".join([f"E{i}:{p:.1f}%" for i, p in enumerate(usage_pct)])
                        print(f"    First batch expert usage (balanced assignment): {usage_str}")

                    # Expert distillation loss (on sparse output)
                    expert_loss = F.mse_loss(sparse_output, target)

                    # Router loss: train to predict the gradient-based selection
                    router_logits = moe_layer.get_router_logits(x)
                    router_logits_flat = router_logits.view(-1, n_experts)

                    router_log_probs = F.log_softmax(router_logits_flat, dim=-1)
                    router_loss = F.kl_div(
                        router_log_probs,
                        routing_targets,
                        reduction='batchmean'
                    )

                    # Entropy bonus
                    router_probs = F.softmax(router_logits_flat, dim=-1)
                    entropy = -(router_probs * (router_probs + 1e-10).log()).sum(dim=-1).mean()
                    entropy_bonus = -self.config.empirical_entropy_weight * entropy

                    # Backward with optional router isolation
                    optimizer.zero_grad()

                    if self.config.detach_router_from_expert_loss:
                        # Separate backward passes to isolate router from expert loss
                        # 1. Backprop expert loss (only to experts, not router)
                        expert_loss.backward(retain_graph=True)

                        # Zero out router gradients from expert loss
                        for name, param in moe_layer.named_parameters():
                            if 'router' in name.lower() and param.grad is not None:
                                param.grad.zero_()

                        # 2. Backprop router loss (only affects router)
                        scaled_router_loss = self.config.router_loss_weight * (router_loss + entropy_bonus)
                        scaled_router_loss.backward()
                    else:
                        # Combined loss (original behavior - router trained by both)
                        loss = expert_loss + self.config.router_loss_weight * (router_loss + entropy_bonus)
                        loss.backward()

                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            moe_layer.parameters(),
                            self.config.max_grad_norm,
                        )

                    optimizer.step()

                    # Track metrics
                    expert_loss_val = expert_loss.item()
                    router_loss_val = router_loss.item() if isinstance(router_loss, torch.Tensor) else router_loss
                    loss_val = expert_loss_val + self.config.router_loss_weight * router_loss_val

                    epoch_losses.append(loss_val)
                    epoch_expert_losses.append(expert_loss_val)
                    epoch_router_losses.append(router_loss_val)
                    self.layer_losses[layer_idx].append(loss_val)

                    total_steps += 1

                    if total_steps % self.config.log_interval == 0:
                        avg_loss = sum(epoch_losses[-100:]) / min(len(epoch_losses), 100)
                        avg_expert = sum(epoch_expert_losses[-100:]) / min(len(epoch_expert_losses), 100)
                        avg_router = sum(epoch_router_losses[-100:]) / min(len(epoch_router_losses), 100)

                        # Compute cosine similarity
                        with torch.no_grad():
                            cos_sim = F.cosine_similarity(
                                sparse_output.view(-1, d_model),
                                target.view(-1, d_model),
                                dim=-1
                            ).mean().item()

                        # Compute expert selection balance from this batch
                        selection_counts = expert_usage.float()
                        if selection_counts.sum() > 0:
                            selection_probs = selection_counts / selection_counts.sum()
                            max_expert_pct = selection_probs.max().item() * 100
                            min_expert_pct = selection_probs.min().item() * 100
                            # Entropy-based balance (1.0 = perfect balance)
                            selection_entropy = -(selection_probs * (selection_probs + 1e-10).log()).sum().item()
                            max_entropy = math.log(n_experts)
                            selection_balance = selection_entropy / max_entropy if max_entropy > 0 else 0
                        else:
                            max_expert_pct = 0
                            min_expert_pct = 0
                            selection_balance = 0

                        postfix = {
                            'loss': f'{avg_loss:.4f}',
                            'expert': f'{avg_expert:.4f}',
                            'router': f'{avg_router:.4f}',
                            'cos': f'{cos_sim:.3f}',
                            'sel_bal': f'{selection_balance:.2f}',
                        }

                        # Show curriculum balance weight if enabled
                        if self.config.curriculum_routing:
                            progress = min(1.0, total_steps / self.config.curriculum_warmup_steps)
                            balance_weight = (
                                self.config.curriculum_initial_balance +
                                progress * (self.config.curriculum_final_balance - self.config.curriculum_initial_balance)
                            )
                            postfix['cur'] = f'{balance_weight:.2f}'

                        pbar.set_postfix(postfix)

                    # Clean up
                    del sparse_output, x, target, activations, expert_loss, router_loss

                    if max_steps and total_steps >= max_steps:
                        break

                # Epoch summary
                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                avg_expert = sum(epoch_expert_losses) / len(epoch_expert_losses) if epoch_expert_losses else 0
                avg_router = sum(epoch_router_losses) / len(epoch_router_losses) if epoch_router_losses else 0
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, expert={avg_expert:.4f}, router={avg_router:.4f}")

                # Print routing stats
                print_layer_routing_stats(moe_layer, layer_idx, prefix="  ")

                if max_steps and total_steps >= max_steps:
                    break

            # Move trained layer back to CPU
            self.moe_layers[str(layer_idx)] = moe_layer.cpu()
            del optimizer
            torch.cuda.empty_cache()

        # Move all layers back to GPU
        print("\nMoving all trained layers to GPU...")
        for layer_idx in self.extractor.layer_indices:
            self.moe_layers[str(layer_idx)] = self.moe_layers[str(layer_idx)].to(self.device)

        return self.layer_losses

    def _distill_data_free(
        self,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[int, List[float]]:
        """
        Data-free distillation using random inputs.

        Instead of running real data through the model, we:
        1. Generate random tensors that match FFN input statistics
        2. Run them through the dense FFN to get targets
        3. Train MoE to match

        This is faster and often provides better input space coverage.

        Args:
            dataloader: Optional dataloader for clustering (supervised routing).
                        If provided, uses real data for clustering even in data-free mode.
        """
        print(f"\nStarting sparse distillation (DATA-FREE mode)...")
        print(f"  Layers: {self.extractor.n_layers}")
        if self.layer_config:
            n_experts_list = [self.moe_layers[str(i)].n_experts for i in self.extractor.layer_indices]
            print(f"  Experts per layer: {min(n_experts_list)}-{max(n_experts_list)} (variable per layer)")
        else:
            print(f"  Experts per layer: {self.config.n_experts}")
        print(f"  Routing: {self.config.moe_routing}")
        print(f"  Steps per layer: {self.config.data_free_steps}")
        print(f"  Batch size: {self.config.data_free_batch_size}")
        print(f"  Seq len: {self.config.data_free_seq_len}")
        if self.config.supervised_routing:
            print(f"  Supervised routing: ON")
            if dataloader is not None:
                print(f"    - Using REAL DATA for clustering (dataloader provided)")
            else:
                print(f"    - Using random inputs for clustering (no dataloader)")
            print(f"    - Train router to predict cluster assignments")
            print(f"    - Then train experts with frozen router")
        print()

        # Get FFN modules from extractor
        ffn_modules = self.extractor.cache._ffn_modules

        for layer_idx in tqdm(self.extractor.layer_indices, desc="Distilling layers"):
            ffn = ffn_modules[layer_idx]
            moe_layer = self.moe_layers[str(layer_idx)].to(self.device)
            n_experts = moe_layer.n_experts  # Use layer's actual n_experts

            print(f"\n{'='*50}")
            print(f"Layer {layer_idx} (n_experts={n_experts})")
            print(f"{'='*50}")

            # Phase 1: Supervised router training (if enabled)
            if self.config.supervised_routing:
                print(f"  Phase 1: Supervised router training")

                if dataloader is not None:
                    # Use real data for clustering (better cluster quality)
                    ffn_inputs, ffn_outputs = self._collect_ffn_activations_from_data(
                        dataloader, layer_idx,
                        max_samples=min(self.config.cluster_samples, 100000),
                    )
                    centroids = self._cluster_outputs(ffn_outputs, n_experts)

                    # Train router using real inputs
                    self._train_router_supervised(
                        moe_layer, ffn, centroids,
                        steps=self.config.router_train_steps,
                        ffn_inputs=ffn_inputs,
                        ffn_outputs=ffn_outputs,
                    )
                else:
                    # Fall back to random inputs for clustering
                    centroids = self._cluster_ffn_outputs(ffn, n_experts)

                    # Train router with random inputs
                    self._train_router_supervised(
                        moe_layer, ffn, centroids,
                        steps=self.config.router_train_steps,
                    )
                print(f"  Phase 2: Expert training (router frozen)")
            else:
                print(f"  Training router and experts jointly")

            # Compile for faster training
            if self.config.use_compile:
                ffn_compiled = torch.compile(ffn)
                moe_compiled = torch.compile(moe_layer)
                print(f"  Using torch.compile")
            else:
                ffn_compiled = ffn
                moe_compiled = moe_layer

            # Create optimizer - optionally freeze router if supervised
            if self.config.supervised_routing:
                # Only train expert parameters, freeze router
                expert_params = [p for n, p in moe_layer.named_parameters() if 'router' not in n]
                optimizer = torch.optim.AdamW(
                    expert_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    moe_layer.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )

            pbar = tqdm(range(self.config.data_free_steps), desc=f"Layer {layer_idx}")
            recent_losses = []
            recent_cos_sims = []
            recent_expert_std = []

            for step in pbar:
                # Generate random input matching FFN input distribution
                # Use standard normal, scaled to approximate real activations
                x = torch.randn(
                    self.config.data_free_batch_size,
                    self.config.data_free_seq_len,
                    self.d_model,
                    device=self.device,
                    dtype=self.dtype,
                )
                # Scale to have unit norm per token (like normalized activations)
                x = F.normalize(x, dim=-1) * math.sqrt(self.d_model)

                # Get dense FFN output (target)
                with torch.no_grad():
                    target = ffn_compiled(x)

                # Forward through MoE
                moe_layer.train()

                # For compiled models, we can't easily get aux_info, so we:
                # - Run compiled forward for speed (most steps)
                # - Run uncompiled forward occasionally for routing stats
                get_aux = (step + 1) % 50 == 0  # Get aux info every 50 steps

                if get_aux:
                    # Uncompiled forward with aux info for routing stats
                    moe_output, aux_info = moe_layer(x, return_aux=True)
                else:
                    # Fast compiled forward
                    moe_output = moe_compiled(x)
                    aux_info = None

                # Compute loss with metrics
                loss, metrics = self.compute_loss(moe_output, target, aux_info=aux_info, return_metrics=True)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        moe_layer.parameters(),
                        self.config.max_grad_norm,
                    )

                optimizer.step()

                # Track metrics
                loss_val = loss.item()
                self.layer_losses[layer_idx].append(loss_val)
                recent_losses.append(loss_val)
                recent_cos_sims.append(metrics['cos_sim'])
                if 'expert_usage_std' in metrics and metrics['expert_usage_std'] is not None:
                    recent_expert_std.append(metrics['expert_usage_std'])

                # Keep only last 100 for averaging
                if len(recent_losses) > 100:
                    recent_losses.pop(0)
                    recent_cos_sims.pop(0)
                    if recent_expert_std:
                        recent_expert_std.pop(0)

                # Update progress bar
                if (step + 1) % 10 == 0:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    avg_cos = sum(recent_cos_sims) / len(recent_cos_sims)
                    postfix = {
                        'loss': f'{avg_loss:.4f}',
                        'cos': f'{avg_cos:.3f}',
                    }
                    if recent_expert_std:
                        avg_std = sum(recent_expert_std) / len(recent_expert_std)
                        postfix['exp_std'] = f'{avg_std:.3f}'
                    pbar.set_postfix(postfix)

                # Detailed logging with routing stats
                if (step + 1) % 200 == 0:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    avg_cos = sum(recent_cos_sims) / len(recent_cos_sims)
                    routing_info = ""
                    if recent_expert_std:
                        avg_std = sum(recent_expert_std) / len(recent_expert_std)
                        routing_info = f", expert_std={avg_std:.3f}"
                        if 'max_expert_frac' in metrics and metrics['max_expert_frac'] is not None:
                            routing_info += f", max_frac={metrics['max_expert_frac']:.2f}"
                            # Warn about potential collapse
                            if metrics['max_expert_frac'] > 0.3:
                                routing_info += " [COLLAPSE?]"
                    tqdm.write(f"  Step {step+1}: loss={avg_loss:.4f}, cos_sim={avg_cos:.4f}{routing_info}")

                # Clean up
                del x, target, moe_output, loss

            # Final metrics for this layer
            final_cos = sum(recent_cos_sims) / len(recent_cos_sims)
            final_loss = sum(recent_losses) / len(recent_losses)
            routing_summary = ""
            final_std = None
            if recent_expert_std:
                final_std = sum(recent_expert_std) / len(recent_expert_std)
                routing_summary = f", expert_std={final_std:.3f}"
            print(f"  Layer {layer_idx} done: loss={final_loss:.4f}, cos_sim={final_cos:.4f}{routing_summary}")

            if final_cos < 0.9:
                print(f"  [WARNING] Low cosine similarity ({final_cos:.3f}) - consider more steps or larger experts")

            if final_std is not None:
                # For n_experts experts, ideal std is ~0 (perfect balance)
                # High std means imbalanced routing
                ideal_frac = 1.0 / n_experts
                if final_std > ideal_frac * 2:
                    print(f"  [WARNING] High expert usage variance (std={final_std:.3f}) - routing may be imbalanced")

            # Print detailed routing stats for this layer
            print_layer_routing_stats(moe_layer, layer_idx, prefix="  ")

            # Keep layer on device
            del optimizer
            torch.cuda.empty_cache()

        print("\n" + "="*50)
        print("Data-free distillation complete!")
        print("="*50)

        # Print summary
        for layer_idx in self.extractor.layer_indices:
            losses = self.layer_losses[layer_idx]
            if losses:
                final_loss = sum(losses[-100:]) / min(len(losses), 100)
                print(f"  Layer {layer_idx}: final_loss={final_loss:.4f}")

        return self.layer_losses

    def _distill_dynamic(
        self,
        dataloader: DataLoader,
        epochs: int = 1,
        max_steps: Optional[int] = None,
    ) -> Dict[int, List[float]]:
        """
        Train with dynamic expert reallocation.

        Tracks per-expert loss and periodically reallocates parameters:
        - High-loss experts grow (more d_ff dimensions)
        - Low-loss experts shrink (fewer dimensions)
        """
        print(f"\nStarting sparse distillation (DYNAMIC mode)...")
        print(f"  Layers: {self.extractor.n_layers}")
        print(f"  Experts per layer: {self.config.n_experts}")
        print(f"  Routing: {self.config.moe_routing}")
        print(f"  Reallocate every: {self.config.reallocate_every_n_steps} steps")
        print(f"  Growth/shrink: {self.config.growth_factor:.2f}x / {self.config.shrink_factor:.2f}x")
        print()

        # Create dynamic expert config
        dynamic_config = DynamicExpertConfig(
            reallocate_every_n_steps=self.config.reallocate_every_n_steps,
            growth_factor=self.config.growth_factor,
            shrink_factor=self.config.shrink_factor,
            min_d_ff=self.config.min_d_ff,
            max_d_ff=self.config.max_d_ff,
            top_k_grow=self.config.top_k_grow,
            top_k_shrink=self.config.top_k_shrink,
            maintain_param_budget=True,
        )

        # Convert MoE layers to dynamic expert layers
        self.dynamic_layers: Dict[int, DynamicExpertLayer] = {}
        self.dynamic_trackers: Dict[int, DynamicExpertTracker] = {}

        for layer_idx in self.extractor.layer_indices:
            # Create dynamic layer with uniform initial dimensions
            d_ff_per_expert = [self.d_ff_expert] * self.config.n_experts

            dynamic_layer = DynamicExpertLayer(
                n_experts=self.config.n_experts,
                d_model=self.d_model,
                d_ff_per_expert=d_ff_per_expert,
                activation="swiglu",
            ).to(device=self.device, dtype=self.dtype)

            self.dynamic_layers[layer_idx] = dynamic_layer

            # Create tracker
            self.dynamic_trackers[layer_idx] = DynamicExpertTracker(
                n_experts=self.config.n_experts,
                initial_d_ff=self.d_ff_expert,
                d_model=self.d_model,
                config=dynamic_config,
            )

        # Print initial state
        print("Initial dynamic expert state:")
        print_active_params_report(self.dynamic_layers, top_k=2)

        # Create router (shared across dynamic training)
        router_config = AdvancedMoEConfig(
            d_model=self.d_model,
            n_experts=self.config.n_experts,
            moe_capacity_factor=self.config.moe_capacity_factor,
        )
        router = ExpertChoiceRouter(router_config).to(device=self.device, dtype=self.dtype)

        # Single optimizer for router + all dynamic layers
        all_params = list(router.parameters())
        for layer in self.dynamic_layers.values():
            all_params.extend(layer.parameters())

        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                # Extract input_ids
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids', batch.get('tokens'))
                    attention_mask = batch.get('attention_mask', None)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0]
                    attention_mask = batch[1] if len(batch) > 1 else None
                else:
                    input_ids = batch
                    attention_mask = None

                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Extract activations
                activations = self.extractor.extract(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                total_loss = 0.0
                optimizer.zero_grad()

                for layer_idx, act in activations.items():
                    x = act.input.to(self.device)
                    target = act.output.to(self.device)

                    dynamic_layer = self.dynamic_layers[layer_idx]
                    tracker = self.dynamic_trackers[layer_idx]

                    # Route tokens
                    expert_weights, token_indices, _, capacity = router(x)

                    # Forward through dynamic layer
                    output, expert_outputs = dynamic_layer(x, expert_weights, token_indices)

                    # Compute per-expert targets for tracking
                    expert_targets = {}
                    batch_size, seq_len, d_model = x.shape
                    for expert_idx in range(self.config.n_experts):
                        indices = token_indices[:, expert_idx, :]
                        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d_model)
                        expert_targets[expert_idx] = torch.gather(target, 1, indices_expanded)

                    # Update tracker with per-expert losses
                    tracker.update_stats(expert_outputs, expert_targets)

                    # Overall loss
                    loss = self.compute_loss(output, target)
                    total_loss += loss

                    self.layer_losses[layer_idx].append(loss.item())

                # Backward and update
                total_loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)

                optimizer.step()

                total_steps += 1

                # Check for reallocation
                if total_steps % self.config.reallocate_every_n_steps == 0:
                    print(f"\n\nStep {total_steps}: Reallocating experts...")

                    for layer_idx in self.extractor.layer_indices:
                        tracker = self.dynamic_trackers[layer_idx]
                        dynamic_layer = self.dynamic_layers[layer_idx]

                        print(f"\nLayer {layer_idx}:")
                        print(tracker.get_report())

                        # Compute new dimensions
                        new_d_ff = tracker.compute_reallocation()

                        # Check if any changes
                        changes = {i: (dynamic_layer.d_ff_per_expert[i], new_d_ff[i])
                                   for i in range(self.config.n_experts)
                                   if dynamic_layer.d_ff_per_expert[i] != new_d_ff[i]}

                        if changes:
                            print(f"  Reallocating: {changes}")

                            # Create reallocated layer
                            new_layer = reallocate_expert_layer(
                                dynamic_layer, new_d_ff, self.device
                            )
                            new_layer = new_layer.to(dtype=self.dtype)

                            # Update references
                            self.dynamic_layers[layer_idx] = new_layer

                            # Update tracker dimensions
                            for i, d_ff in new_d_ff.items():
                                tracker.expert_stats[i].d_ff = d_ff

                        tracker.reset_stats()

                    # Recreate optimizer with new parameters
                    all_params = list(router.parameters())
                    for layer in self.dynamic_layers.values():
                        all_params.extend(layer.parameters())

                    optimizer = torch.optim.AdamW(
                        all_params,
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay,
                    )

                    print("\nAfter reallocation:")
                    print_active_params_report(self.dynamic_layers, top_k=2)
                    print()

                if total_steps % self.config.log_interval == 0:
                    avg_loss = total_loss.item() / len(activations)
                    pbar.set_postfix({'step': total_steps, 'loss': f'{avg_loss:.4f}'})

                if max_steps and total_steps >= max_steps:
                    break

            if max_steps and total_steps >= max_steps:
                break

        # Final report
        print("\n" + "=" * 60)
        print("FINAL DYNAMIC EXPERT STATE")
        print("=" * 60)
        print_active_params_report(self.dynamic_layers, top_k=2)

        return self.layer_losses

    def get_moe_layers(self) -> nn.ModuleDict:
        """Get the trained MoE layers."""
        return self.moe_layers

    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Evaluate MoE layers against dense FFN layers.

        Returns:
            Dict mapping layer_idx to average MSE
        """
        print("\nEvaluating distillation quality...")

        total_mse = {idx: 0.0 for idx in self.extractor.layer_indices}
        total_cos = {idx: 0.0 for idx in self.extractor.layer_indices}
        n_batches = 0

        for moe_layer in self.moe_layers.values():
            moe_layer.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if max_batches and i >= max_batches:
                    break

                # Extract input_ids
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids', batch.get('tokens'))
                    attention_mask = batch.get('attention_mask', None)
                else:
                    input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                    attention_mask = None

                input_ids = input_ids.to(self.device)

                # Get dense activations
                activations = self.extractor.extract(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Compare MoE outputs
                for layer_idx, act in activations.items():
                    moe_layer = self.moe_layers[str(layer_idx)]

                    x = act.input.to(self.device)
                    target = act.output.to(self.device)

                    moe_output = moe_layer(x)

                    # MSE
                    mse = F.mse_loss(moe_output, target).item()
                    total_mse[layer_idx] += mse

                    # Cosine similarity
                    cos_sim = F.cosine_similarity(
                        moe_output.view(-1, moe_output.shape[-1]),
                        target.view(-1, target.shape[-1]),
                        dim=-1,
                    ).mean().item()
                    total_cos[layer_idx] += cos_sim

                n_batches += 1

        # Average
        results = {}
        print("\nDistillation quality:")
        for layer_idx in sorted(total_mse.keys()):
            avg_mse = total_mse[layer_idx] / n_batches
            avg_cos = total_cos[layer_idx] / n_batches
            results[layer_idx] = {'mse': avg_mse, 'cosine_sim': avg_cos}
            print(f"  Layer {layer_idx}: MSE={avg_mse:.6f}, CosSim={avg_cos:.4f}")

        return results

    def save(self, path: str):
        """Save trained MoE layers."""
        # Build per-layer dimensions dict
        layer_dimensions = {}
        for layer_idx in self.extractor.layer_indices:
            moe_layer = self.moe_layers[str(layer_idx)]
            layer_dimensions[layer_idx] = {
                'n_experts': moe_layer.n_experts,
                'd_ff': moe_layer.d_ff,
            }

        state = {
            'moe_layers': self.moe_layers.state_dict(),
            'config': self.config,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'd_ff_expert': self.d_ff_expert,
            'dtype': str(self.dtype),  # Save as string for portability
            'layer_indices': self.extractor.layer_indices,
            'layer_losses': self.layer_losses,
            'global_step': self.global_step,
            # Per-layer dimensions (for variable n_experts / d_ff_expert)
            'layer_dimensions': layer_dimensions,
            'layer_config': self.layer_config,  # Original layer_config.json data if used
        }
        torch.save(state, path)
        print(f"Saved distilled MoE layers to {path}")

    def load(self, path: str):
        """Load trained MoE layers."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.moe_layers.load_state_dict(state['moe_layers'])
        self.layer_losses = state.get('layer_losses', {})
        self.global_step = state.get('global_step', 0)
        print(f"Loaded distilled MoE layers from {path}")
