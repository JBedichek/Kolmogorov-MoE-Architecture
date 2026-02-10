"""
Feed-Forward Network (FFN) modules with various activations.

Supports:
- SwiGLU (Swish-Gated Linear Unit) - used in LLaMA, PaLM
- Standard activations (GELU, SiLU, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import AdvancedMoEConfig


class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit.

    SwiGLU(x) = (xW1 ⊙ σ(xW2))W3

    where:
    - W1, W2: project x to intermediate dimension
    - ⊙: element-wise product
    - σ: SiLU (Swish) activation
    - W3: project back to d_model

    Reference: "GLU Variants Improve Transformer" (Google, 2020)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Gate and value projections
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # Value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # Down projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Gate path with SiLU activation
        gate = F.silu(self.w1(x))  # (batch, seq_len, d_ff)

        # Value path (linear)
        value = self.w2(x)  # (batch, seq_len, d_ff)

        # Gated combination
        hidden = gate * value  # (batch, seq_len, d_ff)

        # Down projection
        output = self.w3(hidden)  # (batch, seq_len, d_model)

        return self.dropout(output)


class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network with configurable activation.

    FFN(x) = activation(xW1)W2
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Select activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu" or activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        hidden = self.activation(self.w1(x))  # (batch, seq_len, d_ff)
        output = self.w2(hidden)  # (batch, seq_len, d_model)
        return self.dropout(output)


def get_ffn_module(
    d_model: int,
    d_ff: int,
    activation: str = "swiglu",
    dropout: float = 0.1,
) -> nn.Module:
    """
    Get FFN module based on activation type.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        activation: Activation type ("swiglu", "gelu", "silu", "relu")
        dropout: Dropout probability

    Returns:
        FFN module
    """
    if activation == "swiglu":
        return SwiGLU(d_model, d_ff, dropout)
    else:
        return StandardFFN(d_model, d_ff, activation, dropout)


class ExpertFFN(nn.Module):
    """
    Expert FFN for Mixture of Experts.

    This is a simple FFN used as an expert in MoE layers.
    Identical to StandardFFN but packaged separately for clarity.
    """

    def __init__(
        self,
        d_model: int,
        d_ff_expert: int,
        activation: str = "swiglu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn = get_ffn_module(d_model, d_ff_expert, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (n_tokens, d_model)

        Returns:
            output: same shape as input
        """
        return self.ffn(x)


if __name__ == "__main__":
    # Test FFN modules
    print("Testing FFN modules...")

    d_model = 256
    d_ff = 704
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Test SwiGLU
    print("\n1. Testing SwiGLU...")
    swiglu = SwiGLU(d_model, d_ff, dropout=0.0)
    output = swiglu(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape
    print(f"   Parameters: {sum(p.numel() for p in swiglu.parameters()):,}")

    # Test StandardFFN
    print("\n2. Testing StandardFFN (GELU)...")
    standard_ffn = StandardFFN(d_model, d_ff, activation="gelu", dropout=0.0)
    output = standard_ffn(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape
    print(f"   Parameters: {sum(p.numel() for p in standard_ffn.parameters()):,}")

    # Test get_ffn_module
    print("\n3. Testing get_ffn_module...")
    ffn = get_ffn_module(d_model, d_ff, activation="swiglu", dropout=0.0)
    output = ffn(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape

    # Test ExpertFFN
    print("\n4. Testing ExpertFFN...")
    expert = ExpertFFN(d_model, d_ff // 2, activation="swiglu", dropout=0.0)
    output = expert(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape
    print(f"   Parameters: {sum(p.numel() for p in expert.parameters()):,}")

    # Test backward pass
    print("\n5. Testing backward pass...")
    loss = output.sum()
    loss.backward()
    print("   Backward pass successful!")

    # Compare parameter counts
    print("\n6. Parameter comparison:")
    print(f"   SwiGLU: {sum(p.numel() for p in swiglu.parameters()):,}")
    print(f"   StandardFFN: {sum(p.numel() for p in standard_ffn.parameters()):,}")
    print(f"   Note: SwiGLU has ~1.5x more parameters (3 matrices vs 2)")

    print("\nAll FFN tests passed!")
