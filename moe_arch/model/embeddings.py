"""
Token embeddings and Rotary Position Embeddings (RoPE) for the transformer model.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import AdvancedMoEConfig


class TokenEmbedding(nn.Module):
    """Standard token embedding layer."""

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        return self.embedding(input_ids)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Instead of adding positional information, RoPE rotates the query and key
    vectors by an angle proportional to their position.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling

        # Precompute frequencies
        self._setup_frequencies()

    def _setup_frequencies(self):
        """
        Setup the frequency tensor for RoPE.

        freq_i = 1 / (theta^(2i/dim)) for i in [0, dim/2)
        """
        # Compute inverse frequencies
        # inv_freq: (dim/2,)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for all positions
        # This is optional but can speed up forward pass
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (max_seq_len, dim/2)

        # Create emb by concatenating [freqs, freqs] to match full dimension
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor (batch, seq_len, n_heads, head_dim) or (batch, n_heads, seq_len, head_dim)
            k: Key tensor (same shape as q)
            position_ids: Optional position IDs (batch, seq_len). If None, use range(seq_len)

        Returns:
            q_rotated, k_rotated: Rotated query and key tensors (same shape as input)
        """
        # Handle different input formats
        # Expected: (batch, seq_len, n_heads, head_dim)
        batch_size, seq_len = q.shape[0], q.shape[1]

        # Get position IDs
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=q.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        # Get cos and sin for the positions
        # cos_cached, sin_cached: (max_seq_len, dim)
        cos = self.cos_cached[position_ids]  # (batch, seq_len, dim)
        sin = self.sin_cached[position_ids]  # (batch, seq_len, dim)

        # Reshape to match query/key dimensions
        # q, k: (batch, seq_len, n_heads, head_dim)
        # cos, sin: (batch, seq_len, head_dim)
        cos = cos.unsqueeze(2)  # (batch, seq_len, 1, head_dim)
        sin = sin.unsqueeze(2)  # (batch, seq_len, 1, head_dim)

        # Apply rotation
        q_rotated = self._apply_rotary_emb(q, cos, sin)
        k_rotated = self._apply_rotary_emb(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.

        Rotation formula:
            x_out = x * cos + rotate_half(x) * sin

        where rotate_half swaps and negates the first and second half of the last dimension.
        """
        # x: (batch, seq_len, n_heads, head_dim)
        # cos, sin: (batch, seq_len, 1, head_dim)

        # Split x into first and second half
        x1, x2 = x.chunk(2, dim=-1)  # Each: (batch, seq_len, n_heads, head_dim/2)

        # Split cos and sin similarly
        cos1, cos2 = cos.chunk(2, dim=-1)
        sin1, sin2 = sin.chunk(2, dim=-1)

        # Apply rotation
        # Formula: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        # But RoPE actually does: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        # Simplified: x * cos + rotate_half(x) * sin
        # where rotate_half([x1, x2]) = [-x2, x1]

        rotated = torch.cat(
            [
                x1 * cos1 - x2 * sin1,
                x2 * cos2 + x1 * sin2,
            ],
            dim=-1,
        )

        return rotated


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is simpler and faster than LayerNorm, normalizing only by the RMS
    and scaling with a learned parameter (no bias/shift).

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            normalized: (batch, seq_len, dim)
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_normalized = x / rms
        return self.weight * x_normalized


def get_norm_layer(config: AdvancedMoEConfig) -> nn.Module:
    """
    Get normalization layer based on config.

    Args:
        config: Model configuration

    Returns:
        Normalization layer (RMSNorm or LayerNorm)
    """
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model, eps=config.norm_eps)
    elif config.norm_type == "layernorm":
        return nn.LayerNorm(config.d_model, eps=config.norm_eps)
    else:
        raise ValueError(f"Unknown norm type: {config.norm_type}")


if __name__ == "__main__":
    # Test embeddings
    from .config import get_test_config

    config = get_test_config()
    print(f"Testing embeddings with config: d_model={config.d_model}, vocab_size={config.vocab_size}")

    # Test token embedding
    token_emb = TokenEmbedding(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))  # (batch=2, seq=16)
    embeddings = token_emb(input_ids)
    print(f"Token embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, 16, config.d_model)

    # Test RoPE
    rope = RotaryPositionalEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
    q = torch.randn(2, 16, 4, config.head_dim)  # (batch, seq, n_heads, head_dim)
    k = torch.randn(2, 16, 4, config.head_dim)
    q_rot, k_rot = rope(q, k)
    print(f"RoPE output shapes: q={q_rot.shape}, k={k_rot.shape}")
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # Test RMSNorm
    rms_norm = RMSNorm(config.d_model)
    x = torch.randn(2, 16, config.d_model)
    x_norm = rms_norm(x)
    print(f"RMSNorm output shape: {x_norm.shape}")
    assert x_norm.shape == x.shape

    # Verify normalization (should have RMS ≈ 1)
    rms = torch.sqrt(torch.mean(x_norm ** 2, dim=-1))
    print(f"RMS after normalization (should be ≈1): mean={rms.mean().item():.4f}, std={rms.std().item():.4f}")

    print("\nAll embedding tests passed!")
