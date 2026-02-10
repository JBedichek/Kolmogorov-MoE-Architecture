"""
Transformer block implementation combining attention and FFN.

Phase 1: Basic transformer block with attention + FFN
Phase 2: Added MoE support
Phase 3: Added Mamba and MoD support
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .config import AdvancedMoEConfig
from .attention import GroupedQueryAttention
from .ffn import get_ffn_module
from .embeddings import get_norm_layer
from .moe import MoELayer
from .mamba import MambaBlock, RoutingMamba
from .mod import MoDRouter


class TransformerBlock(nn.Module):
    """
    Advanced Transformer Block with comprehensive architectural features.

    Architecture (with MoD):
        if token_selected_by_MoD:
            x = x + SeqLayer(Norm(x))  # Attention or Mamba
        if token_selected_by_MoD:
            x = x + FFN(Norm(x))        # Standard or MoE

    Features:
    - Grouped Query Attention (GQA) or Mamba SSM blocks
    - Routing Mamba (RoM) for Mamba layers with MoE-style routing
    - Mixture of Experts (MoE) for FFN layers
    - Mixture of Depths (MoD) for conditional layer skipping
    - Pre-normalization (RMSNorm or LayerNorm)
    - Residual dropout
    - Comprehensive auxiliary losses (MoE, RoM, MoD)
    """

    def __init__(
        self,
        config: AdvancedMoEConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine layer configuration
        self.use_mamba = config.mamba_enabled and (layer_idx in config.mamba_layers)
        self.use_moe = layer_idx in config.moe_layers
        self.use_mod = config.mod_enabled

        # Sequence modeling layer (Attention or Mamba)
        self.seq_norm = get_norm_layer(config)
        if self.use_mamba:
            # Use Routing Mamba if this layer also has MoE
            # (shared router pattern from RoM paper)
            if self.use_moe:
                self.seq_layer = RoutingMamba(config, layer_idx)
            else:
                self.seq_layer = MambaBlock(config)
        else:
            # Use standard attention
            self.seq_layer = GroupedQueryAttention(config)

        # FFN (Standard or MoE)
        self.ffn_norm = get_norm_layer(config)
        if self.use_moe and not self.use_mamba:
            # Use MoE FFN for non-Mamba layers
            # (Mamba layers use RoutingMamba which already has routing)
            self.ffn = MoELayer(config, layer_idx)
        else:
            # Use standard FFN
            self.ffn = get_ffn_module(
                config.d_model,
                config.d_ff,
                config.ffn_activation,
                config.dropout,
            )

        # MoD routers (if enabled)
        if self.use_mod:
            self.seq_mod_router = MoDRouter(config)
            self.ffn_mod_router = MoDRouter(config)
        else:
            self.seq_mod_router = None
            self.ffn_mod_router = None

        # Residual dropout
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        # Track auxiliary loss (for MoE, Routing Mamba, and MoD)
        self.aux_loss = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of transformer block.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        total_aux_loss = 0.0

        # Sequence modeling layer (Attention or Mamba) with pre-norm and residual
        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states)

        # Apply MoD if enabled
        if self.use_mod and self.training:
            # Route tokens for sequence layer
            selected_mask, selected_indices, scores = self.seq_mod_router(hidden_states)
            total_aux_loss += self.seq_mod_router.aux_loss

            # Gather selected tokens
            selected_tokens = torch.gather(
                hidden_states,
                1,
                selected_indices.unsqueeze(-1).expand(-1, -1, d_model),
            )

            # Process selected tokens
            if self.use_mamba:
                seq_output_selected = self.seq_layer(selected_tokens)
            else:
                seq_output_selected = self.seq_layer(
                    selected_tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

            # Scatter back (ensure dtype matches for mixed precision)
            seq_output = residual.clone()
            seq_output.scatter_(
                1,
                selected_indices.unsqueeze(-1).expand(-1, -1, d_model),
                seq_output_selected.to(seq_output.dtype),
            )
        else:
            # Process all tokens (no MoD)
            if self.use_mamba:
                seq_output = self.seq_layer(hidden_states)
            else:
                seq_output = self.seq_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

        seq_output = self.residual_dropout(seq_output)
        hidden_states = residual + seq_output

        # Collect aux loss from Routing Mamba if applicable
        if self.use_mamba and hasattr(self.seq_layer, 'aux_loss') and self.training:
            total_aux_loss += self.seq_layer.aux_loss

        # FFN (Standard or MoE) with pre-norm and residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)

        # Apply MoD if enabled
        if self.use_mod and self.training:
            # Route tokens for FFN layer
            selected_mask, selected_indices, scores = self.ffn_mod_router(hidden_states)
            total_aux_loss += self.ffn_mod_router.aux_loss

            # Gather selected tokens
            selected_tokens = torch.gather(
                hidden_states,
                1,
                selected_indices.unsqueeze(-1).expand(-1, -1, d_model),
            )

            # Process selected tokens
            ffn_output_selected = self.ffn(selected_tokens)

            # Scatter back (ensure dtype matches for mixed precision)
            ffn_output = residual.clone()
            ffn_output.scatter_(
                1,
                selected_indices.unsqueeze(-1).expand(-1, -1, d_model),
                ffn_output_selected.to(ffn_output.dtype),
            )
        else:
            # Process all tokens (no MoD)
            ffn_output = self.ffn(hidden_states)

        ffn_output = self.residual_dropout(ffn_output)
        hidden_states = residual + ffn_output

        # Collect aux loss from MoE if applicable
        if self.use_moe and hasattr(self.ffn, 'aux_loss') and self.training:
            total_aux_loss += self.ffn.aux_loss

        # Store total auxiliary loss
        self.aux_loss = total_aux_loss if self.training else 0.0

        return hidden_states


if __name__ == "__main__":
    # Test TransformerBlock
    from .config import get_test_config

    config = get_test_config()
    config.use_flash_attention = False

    print(f"Testing TransformerBlock with config:")
    print(f"  d_model={config.d_model}, d_ff={config.d_ff}")
    print(f"  n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}")
    print(f"  ffn_activation={config.ffn_activation}")

    # Create transformer block
    block = TransformerBlock(config, layer_idx=0)
    print(f"  Parameters: {sum(p.numel() for p in block.parameters()):,}")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)

    print(f"\nInput shape: {hidden_states.shape}")

    output = block(hidden_states)
    print(f"Output shape: {output.shape}")
    assert output.shape == hidden_states.shape

    # Test backward pass
    print("\nTesting backward pass...")
    loss = output.sum()
    loss.backward()
    print("Backward pass successful!")

    # Test with different sequence lengths
    print("\nTesting with seq_len=32...")
    hidden_states_32 = torch.randn(batch_size, 32, config.d_model)
    output_32 = block(hidden_states_32)
    print(f"Output shape: {output_32.shape}")
    assert output_32.shape == hidden_states_32.shape

    # Parameter breakdown
    print("\nParameter breakdown:")
    attn_params = sum(p.numel() for p in block.attn.parameters())
    ffn_params = sum(p.numel() for p in block.ffn.parameters())
    norm_params = sum(p.numel() for p in block.attn_norm.parameters()) + \
                   sum(p.numel() for p in block.ffn_norm.parameters())
    total_params = sum(p.numel() for p in block.parameters())

    print(f"  Attention: {attn_params:,} ({100*attn_params/total_params:.1f}%)")
    print(f"  FFN: {ffn_params:,} ({100*ffn_params/total_params:.1f}%)")
    print(f"  Norms: {norm_params:,} ({100*norm_params/total_params:.1f}%)")
    print(f"  Total: {total_params:,}")

    print("\nAll TransformerBlock tests passed!")
