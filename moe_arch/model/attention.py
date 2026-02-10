"""
Multi-Head Attention and Grouped Query Attention with RoPE support.

Supports:
- Grouped Query Attention (GQA) - reduces KV cache size
- Flash Attention 2 - memory-efficient attention
- Causal masking for autoregressive generation
- RoPE positional embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import AdvancedMoEConfig
from .embeddings import RotaryPositionalEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    GQA reduces the number of key-value heads while keeping the same number
    of query heads. This reduces KV cache size significantly while maintaining
    most of the model quality.

    For example, with 16 query heads and 4 KV heads, we have a 4:1 ratio,
    reducing KV cache by 4x.

    Reference: "GQA: Training Generalized Multi-Query Transformer Models" (Google, 2023)
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout

        # Validate configuration
        assert self.d_model == self.n_heads * self.head_dim, \
            f"d_model ({self.d_model}) must equal n_heads * head_dim ({self.n_heads} * {self.head_dim})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        self.n_groups = self.n_heads // self.n_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta,
            config.rope_scaling,
        )

        # Attention dropout
        self.attn_dropout = nn.Dropout(self.dropout)

        # Flash Attention support
        self.use_flash_attention = config.use_flash_attention
        self.flash_attn_available = False
        if self.use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.flash_attn_available = True
            except ImportError:
                print("Warning: flash-attn not available, falling back to standard attention")
                self.flash_attn_available = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of Grouped Query Attention.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: Optional attention mask (batch, seq_len) or (batch, 1, seq_len, seq_len)
            position_ids: Optional position IDs (batch, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # (batch, seq, n_heads * head_dim)
        k = self.k_proj(hidden_states)  # (batch, seq, n_kv_heads * head_dim)
        v = self.v_proj(hidden_states)  # (batch, seq, n_kv_heads * head_dim)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k, position_ids)

        # Expand KV heads to match Q heads (for GQA)
        # Each KV head is replicated n_groups times
        k = self._expand_kv(k)  # (batch, seq, n_heads, head_dim)
        v = self._expand_kv(v)  # (batch, seq, n_heads, head_dim)

        # Compute attention
        if self.flash_attn_available and self.use_flash_attention:
            # Use Flash Attention
            output = self._flash_attention(q, k, v)
        else:
            # Use standard attention
            output = self._standard_attention(q, k, v, attention_mask)

        # Reshape and project output
        output = output.reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        output = self.o_proj(output)

        return output

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match the number of Q heads.

        Args:
            x: (batch, seq, n_kv_heads, head_dim)

        Returns:
            expanded: (batch, seq, n_heads, head_dim)
        """
        batch_size, seq_len, n_kv_heads, head_dim = x.shape

        # Repeat each KV head n_groups times
        # (batch, seq, n_kv_heads, head_dim) -> (batch, seq, n_kv_heads, n_groups, head_dim)
        x = x.unsqueeze(3).expand(batch_size, seq_len, n_kv_heads, self.n_groups, head_dim)

        # Reshape to (batch, seq, n_heads, head_dim)
        x = x.reshape(batch_size, seq_len, self.n_heads, head_dim)

        return x

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention with causal masking.

        Args:
            q: (batch, seq, n_heads, head_dim)
            k: (batch, seq, n_heads, head_dim)
            v: (batch, seq, n_heads, head_dim)
            attention_mask: Optional mask

        Returns:
            output: (batch, seq, n_heads, head_dim)
        """
        # Transpose to (batch, n_heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # (batch, n_heads, seq, head_dim) @ (batch, n_heads, head_dim, seq)
        # -> (batch, n_heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create causal mask
        batch_size, n_heads, seq_len, _ = scores.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )

        # Apply causal mask
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (batch, n_heads, seq, seq) @ (batch, n_heads, seq, head_dim)
        # -> (batch, n_heads, seq, head_dim)
        output = torch.matmul(attn_weights, v)

        # Transpose back to (batch, seq, n_heads, head_dim)
        output = output.transpose(1, 2)

        return output

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flash Attention 2 - memory-efficient attention.

        Args:
            q: (batch, seq, n_heads, head_dim)
            k: (batch, seq, n_heads, head_dim)
            v: (batch, seq, n_heads, head_dim)

        Returns:
            output: (batch, seq, n_heads, head_dim)
        """
        # Flash Attention expects: (batch, seq, n_heads, head_dim)
        # and applies causal masking automatically
        output = self.flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,  # Autoregressive causal masking
        )

        return output


if __name__ == "__main__":
    # Test attention module
    from .config import get_test_config

    config = get_test_config()
    config.use_flash_attention = False  # Test without Flash Attention first

    print(f"Testing GQA with config:")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}")
    print(f"  head_dim={config.head_dim}, n_groups={config.n_heads // config.n_kv_heads}")

    # Create attention module
    attn = GroupedQueryAttention(config)
    print(f"  Parameters: {sum(p.numel() for p in attn.parameters()):,}")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)

    print(f"\nInput shape: {hidden_states.shape}")

    output = attn(hidden_states)
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
    output_32 = attn(hidden_states_32)
    print(f"Output shape: {output_32.shape}")
    assert output_32.shape == hidden_states_32.shape

    # Compare KV cache sizes
    print("\nKV cache size comparison:")
    full_mha_kv_size = 2 * config.n_heads * config.head_dim * seq_len
    gqa_kv_size = 2 * config.n_kv_heads * config.head_dim * seq_len
    reduction = full_mha_kv_size / gqa_kv_size
    print(f"  Full MHA KV cache: {full_mha_kv_size:,} elements")
    print(f"  GQA KV cache: {gqa_kv_size:,} elements")
    print(f"  Reduction: {reduction:.1f}x smaller")

    print("\nAll attention tests passed!")
