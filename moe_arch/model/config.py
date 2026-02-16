"""
Configuration dataclass for Advanced MoE Transformer.

This configuration defines all hyperparameters for the 5B parameter model
with MoE, MoD, Mamba blocks, and multi-token prediction.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class AdvancedMoEConfig:
    """Configuration for Advanced MoE Transformer."""

    # ===== Model Size =====
    vocab_size: int = 50000
    d_model: int = 2048
    n_layers: int = 32
    max_seq_len: int = 2048

    # ===== Attention Configuration =====
    n_heads: int = 16
    n_kv_heads: int = 4  # For Grouped Query Attention (GQA)
    head_dim: int = 128  # d_model // n_heads

    # ===== FFN Configuration =====
    d_ff: int = 5632  # 2.75x expansion for standard FFN
    d_ff_expert: int = 2816  # Smaller for MoE experts (half of d_ff)
    ffn_activation: str = "swiglu"  # "swiglu", "gelu", "silu"

    # ===== MoE Configuration =====
    n_experts: int = 16
    moe_top_k: int = 2  # Number of experts to route each token to
    moe_capacity_factor: float = 1.25  # Overflow handling capacity
    moe_load_balance_loss_weight: float = 0.01
    moe_router_z_loss_weight: float = 0.001
    # MoE implementation type:
    # - "batched": Padded batched bmm (default, fastest for small expert count)
    # - "sparse": True sparse computation (no padding waste, better for many experts)
    # - "expert_parallel": Distributed experts across GPUs (for multi-GPU training)
    moe_implementation: str = "batched"
    # MoE routing type:
    # - "token_choice": Tokens select top-k experts (standard, can collapse)
    # - "expert_choice": Experts select top-k tokens (perfect load balance, collapse-resistant)
    moe_routing: str = "token_choice"
    # Balanced routing: enforce capacity constraints per expert to prevent collapse
    # When True, uses balanced top-k assignment instead of pure top-k
    # Each expert can only handle (n_tokens * top_k / n_experts) tokens per batch
    moe_balanced_routing: bool = False
    # Layers that use MoE (0-indexed)
    # Pattern: layers 2, 3, 6, 7, 9, 10, 11, 14, 15, 17, 18, 19, 22, 23, 25, 26, 27, 30, 31
    # Note: Layers 9, 17, 25 are Mamba layers that will use RoutingMamba (RoM)
    moe_layers: Tuple[int, ...] = field(default_factory=lambda: tuple(
        i for i in range(32) if ((i % 4 == 2) or (i % 4 == 3) or (i in [9, 17, 25]))
    ))

    # ===== Mixture of Depths (MoD) Configuration =====
    mod_enabled: bool = True
    mod_capacity_factor: float = 0.75  # 75% of tokens selected, 25% skip
    mod_load_balance_loss_weight: float = 0.001
    mod_router_hidden_dim: int = 128

    # ===== Mamba SSM Configuration =====
    mamba_enabled: bool = True
    ssm_state_dim: int = 16  # State space dimension
    ssm_expansion: int = 2  # Inner dimension expansion
    # Layers that use Mamba instead of attention (0-indexed)
    # Pattern: layers 5, 9, 13, 17, 21, 25, 29 (every 4th layer, offset by 1)
    # Note: Layers 9, 17, 25 also use MoE -> will use RoutingMamba (RoM)
    # Layers 5, 13, 21, 29 use standard Mamba
    mamba_layers: Tuple[int, ...] = field(default_factory=lambda: tuple(
        i for i in range(32) if (i % 4 == 1) and i >= 5
    ))

    # ===== Multi-Token Prediction Configuration =====
    n_pred_tokens: int = 4  # Predict t+1, t+2, t+3, t+4
    aux_loss_weights: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.2)

    # ===== Positional Encoding =====
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None  # For context extension

    # ===== Regularization =====
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    # ===== Normalization =====
    norm_type: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    norm_eps: float = 1e-5

    # ===== Initialization =====
    init_std: float = 0.02

    # ===== Training Configuration =====
    use_gradient_checkpointing: bool = True
    gradient_checkpointing_segments: int = 4  # Checkpoint every N layers

    # ===== Flash Attention =====
    use_flash_attention: bool = True

    # ===== Weight Tying =====
    tie_word_embeddings: bool = True  # Tie lm_head[0] weights with token embeddings

    # ===== Device =====
    device: str = "cuda"
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"

    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Validate head dimensions
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        # Validate GQA configuration
        self.n_groups = self.n_heads // self.n_kv_heads

        # Validate layer indices
        assert all(0 <= i < self.n_layers for i in self.moe_layers), \
            "MoE layer indices must be within [0, n_layers)"
        assert all(0 <= i < self.n_layers for i in self.mamba_layers), \
            "Mamba layer indices must be within [0, n_layers)"

        # Check for conflicts (a layer can't be both Mamba and MoE)
        moe_set = set(self.moe_layers)
        mamba_set = set(self.mamba_layers)
        conflict = moe_set & mamba_set
        if conflict:
            # This is actually okay - Mamba can have MoE FFN
            # But let's log a warning
            pass

        # Validate multi-token prediction
        assert len(self.aux_loss_weights) == self.n_pred_tokens, \
            f"aux_loss_weights length ({len(self.aux_loss_weights)}) must match n_pred_tokens ({self.n_pred_tokens})"

        # Validate MoD
        assert 0.0 < self.mod_capacity_factor <= 1.0, \
            "mod_capacity_factor must be in (0, 1]"

    def get_layer_type(self, layer_idx: int) -> Tuple[str, bool]:
        """
        Get the type of layer for a given index.

        Returns:
            (attention_type, use_moe):
                - attention_type: "attention" or "mamba"
                - use_moe: whether this layer uses MoE FFN
        """
        attention_type = "mamba" if layer_idx in self.mamba_layers else "attention"
        use_moe = layer_idx in self.moe_layers
        return attention_type, use_moe

    def count_parameters(self) -> dict:
        """
        Estimate parameter counts for different components.

        Returns a dictionary with parameter counts for debugging/analysis.
        """
        # Embedding parameters
        token_embed = self.vocab_size * self.d_model

        # Attention parameters (per layer)
        # GQA: Q projection = d_model * n_heads * head_dim
        #      K/V projections = d_model * n_kv_heads * head_dim (each)
        #      O projection = n_heads * head_dim * d_model
        attn_per_layer = (
            self.d_model * self.n_heads * self.head_dim +  # Q
            2 * self.d_model * self.n_kv_heads * self.head_dim +  # K, V
            self.n_heads * self.head_dim * self.d_model  # O
        )

        # Standard FFN parameters (per layer)
        # SwiGLU: gate + up + down = 3 projections
        ffn_per_layer = (
            self.d_model * self.d_ff +  # gate
            self.d_model * self.d_ff +  # up
            self.d_ff * self.d_model    # down
        )

        # MoE FFN parameters (per layer)
        # n_experts * (gate + up + down)
        moe_ffn_per_layer = self.n_experts * (
            self.d_model * self.d_ff_expert +
            self.d_model * self.d_ff_expert +
            self.d_ff_expert * self.d_model
        )
        # Router: d_model -> 128 -> n_experts
        moe_router_per_layer = (
            self.d_model * 128 + 128 * self.n_experts
        )

        # Mamba parameters (per layer) - approximate
        # Mamba is complex, but roughly: 2 * d_model * (ssm_expansion * d_model)
        mamba_per_layer = 15_000_000  # Rough estimate ~15M per layer

        # MoD router (per layer)
        mod_router_per_layer = (
            self.d_model * self.mod_router_hidden_dim +
            self.mod_router_hidden_dim
        )

        # Count layers of each type
        n_standard_attn = sum(
            1 for i in range(self.n_layers)
            if i not in self.mamba_layers and i not in self.moe_layers
        )
        n_moe_attn = sum(
            1 for i in range(self.n_layers)
            if i not in self.mamba_layers and i in self.moe_layers
        )
        n_mamba_standard = sum(
            1 for i in range(self.n_layers)
            if i in self.mamba_layers and i not in self.moe_layers
        )
        n_mamba_moe = sum(
            1 for i in range(self.n_layers)
            if i in self.mamba_layers and i in self.moe_layers
        )

        # Total parameters
        total_attn = (n_standard_attn + n_moe_attn) * attn_per_layer
        total_mamba = (n_mamba_standard + n_mamba_moe) * mamba_per_layer
        total_standard_ffn = (n_standard_attn + n_mamba_standard) * ffn_per_layer
        total_moe_ffn = (n_moe_attn + n_mamba_moe) * (moe_ffn_per_layer + moe_router_per_layer)
        total_mod = self.n_layers * mod_router_per_layer if self.mod_enabled else 0
        total_norm = self.n_layers * 2 * self.d_model  # 2 norms per layer

        # Multi-token prediction heads
        multi_token_heads = (self.n_pred_tokens - 1) * self.vocab_size * self.d_model

        # LM head (shared with embeddings in most architectures, but counting separately)
        lm_head = self.vocab_size * self.d_model

        total = (
            token_embed + total_attn + total_mamba +
            total_standard_ffn + total_moe_ffn + total_mod + total_norm +
            multi_token_heads + lm_head
        )

        return {
            "token_embeddings": token_embed,
            "attention": total_attn,
            "mamba": total_mamba,
            "standard_ffn": total_standard_ffn,
            "moe_ffn": total_moe_ffn,
            "mod_routers": total_mod,
            "layer_norms": total_norm,
            "multi_token_heads": multi_token_heads,
            "lm_head": lm_head,
            "total": total,
            "total_billions": total / 1e9,
            # Layer counts for debugging
            "n_standard_attn_layers": n_standard_attn,
            "n_moe_attn_layers": n_moe_attn,
            "n_mamba_standard_layers": n_mamba_standard,
            "n_mamba_moe_layers": n_mamba_moe,
        }

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AdvancedMoEConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


# Predefined configurations

def get_5b_config() -> AdvancedMoEConfig:
    """Get the default 5B parameter configuration."""
    return AdvancedMoEConfig()


def get_3b_config() -> AdvancedMoEConfig:
    """
    Get a 3B parameter configuration with high sparsity.

    This config uses:
    - Smaller d_model (1792 vs 2048)
    - Fewer layers (24 vs 32)
    - RoutingMamba enabled in layers [9, 17, 21]
    - Standard Mamba in layers [5, 13]
    - Attention+MoE in layers [2,3,6,7,10,11,14,15,18,19,22,23]
    - 64 experts with top-2 selection (3.1% sparsity vs 12.5% with 16 experts)
    - MoD at 60% for compute reduction
    - ~0.7B active parameters (88% sparsity)

    Total params: ~9B, Active: ~0.7B (88% sparse)
    """
    # Calculate MoE layers: pattern 2,3,6,7,10,11,14,15,18,19,22,23
    # Plus include some Mamba layers: 9, 17, 21
    moe_layers = tuple(
        i for i in range(24)
        if ((i % 4 == 2) or (i % 4 == 3) or (i in [9, 17, 21]))
    )

    # Mamba layers: 5, 9, 13, 17, 21
    mamba_layers = tuple(
        i for i in range(24)
        if (i % 4 == 1) and i >= 5
    )

    return AdvancedMoEConfig(
        # Model size
        d_model=1792,
        n_layers=24,

        # Attention
        n_heads=14,
        n_kv_heads=2,
        head_dim=128,

        # FFN
        d_ff=4928,  # 2.75x expansion
        d_ff_expert=2464,  # Half for MoE experts

        # MoE - 64 experts with top-2 for high sparsity
        n_experts=64,  # Increased from 16
        moe_top_k=2,   # Keep top-2 for router training

        # MoD - 50% token selection for aggressive reduction
        mod_capacity_factor=0.5,  # Reduced from 0.75

        # Layer patterns
        moe_layers=moe_layers,
        mamba_layers=mamba_layers,
    )


def get_test_config() -> AdvancedMoEConfig:
    """Get a small configuration for testing."""
    return AdvancedMoEConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        d_ff=704,
        d_ff_expert=352,
        n_experts=4,
        moe_layers=(1, 2),
        mamba_layers=(1,),
        max_seq_len=128,
        n_pred_tokens=2,
        aux_loss_weights=(1.0, 0.5),
    )


if __name__ == "__main__":
    # Test configuration
    config = get_5b_config()
    print("5B Model Configuration:")
    print(f"  Vocabulary size: {config.vocab_size:,}")
    print(f"  Model dimension: {config.d_model:,}")
    print(f"  Number of layers: {config.n_layers}")
    print(f"  Number of experts: {config.n_experts}")
    print(f"  MoE layers: {len(config.moe_layers)}")
    print(f"  Mamba layers: {len(config.mamba_layers)}")
    print(f"  Max sequence length: {config.max_seq_len:,}")
    print(f"  Multi-token prediction heads: {config.n_pred_tokens}")

    print("\nParameter counts:")
    params = config.count_parameters()
    for key, value in params.items():
        if isinstance(value, int):
            if value > 1e6:
                print(f"  {key}: {value/1e6:.2f}M")
            else:
                print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")

    print(f"\nEstimated total: {params['total_billions']:.2f}B parameters")
