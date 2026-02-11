"""
Main MoETransformer model implementation.

Phase 1: Basic transformer with single-token prediction
Phase 2: MoE layers with expert routing
Phase 3: MoD, Mamba blocks, multi-token prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from .config import AdvancedMoEConfig
from .embeddings import TokenEmbedding, get_norm_layer
from .layers import TransformerBlock
from ..training.losses import MultiTokenPredictionLoss


class MoETransformer(nn.Module):
    """
    Advanced MoE Transformer Language Model.

    Features (Phase 1):
    - Token embeddings
    - Stack of Transformer blocks with GQA
    - RoPE positional embeddings
    - Pre-normalization with RMSNorm
    - Language modeling head

    Features (Phase 2):
    - MoE layers with expert routing
    - Load balancing auxiliary losses
    - Efficient grouped GEMM expert computation

    Features (Phase 3):
    - Mixture of Depths (MoD) - conditional layer skipping
    - Mamba SSM blocks for efficient sequence modeling
    - Routing Mamba (RoM) - MoE-style routing for Mamba
    - Multi-token prediction (predict t+1, t+2, t+3, t+4)

    Future phases will add:
    - Phase 4: Training infrastructure
    - Phase 5: Flash Attention, Muon optimizer
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False  # Enable later with .gradient_checkpointing_enable()

        # Token embeddings
        self.token_embedding = TokenEmbedding(config)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final normalization
        self.final_norm = get_norm_layer(config)

        # Language modeling heads for multi-token prediction
        # Main head (t+1) + auxiliary heads (t+2, t+3, t+4, ...)
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.d_model, config.vocab_size, bias=False)
            for _ in range(config.n_pred_tokens)
        ])

        # Optionally tie first head with token embeddings for weight sharing
        # self.lm_heads[0].weight = self.token_embedding.embedding.weight

        # Multi-token prediction loss module
        self.multitoken_loss = MultiTokenPredictionLoss(
            n_pred_tokens=config.n_pred_tokens,
            aux_loss_weights=config.aux_loss_weights,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Report model size
        n_params = sum(p.numel() for p in self.parameters())
        n_active = self.count_active_parameters()
        sparsity = 1.0 - (n_active / n_params) if n_params > 0 else 0.0
        print(f"Initialized MoETransformer with {n_params/1e9:.2f}B parameters")
        print(f"  Active parameters: {n_active/1e9:.2f}B ({sparsity:.1%} sparsity)")
        print(f"  Multi-token prediction: {config.n_pred_tokens} heads")

    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory (trades compute for memory)."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids: (batch, seq_len) - Input token IDs
            attention_mask: (batch, seq_len) - Optional attention mask
            position_ids: (batch, seq_len) - Optional position IDs
            labels: (batch, seq_len) - Optional labels for language modeling loss

        Returns:
            Dictionary containing:
                - logits: (batch, seq_len, vocab_size) - Main LM head logits (t+1)
                - logits_list: List of all prediction head logits
                - loss: Total loss (multitoken + auxiliary) if labels provided
                - lm_loss: Multi-token prediction loss
                - multitoken_loss_dict: Individual losses per prediction head
                - aux_loss: Auxiliary losses (MoE, MoD, etc.)
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        hidden_states = self.token_embedding(input_ids)  # (batch, seq, d_model)

        # Forward through all transformer blocks
        # Collect auxiliary losses from MoE layers
        total_aux_loss = 0.0
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                from torch.utils.checkpoint import checkpoint
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            # Accumulate auxiliary losses from MoE layers
            if self.training and hasattr(layer, 'aux_loss'):
                total_aux_loss += layer.aux_loss

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Multi-token prediction heads
        # Generate predictions for t+1, t+2, t+3, t+4
        logits_list = [
            lm_head(hidden_states)  # (batch, seq, vocab_size)
            for lm_head in self.lm_heads
        ]

        # Main logits (t+1 prediction) for generation
        logits = logits_list[0]

        # Compute loss if labels provided
        loss = None
        lm_loss = None
        multitoken_loss_dict = {}

        if labels is not None:
            # Compute multi-token prediction loss with attention mask
            multitoken_loss, multitoken_loss_dict = self.multitoken_loss(
                logits_list, labels, attention_mask=attention_mask
            )
            lm_loss = multitoken_loss

            # Total loss = Multi-token LM loss + auxiliary losses (MoE, MoD, etc.)
            if self.training and total_aux_loss > 0:
                loss = lm_loss + total_aux_loss
            else:
                loss = lm_loss

        return {
            "logits": logits,  # Main head logits for generation
            "logits_list": logits_list,  # All heads for analysis
            "loss": loss,
            "lm_loss": lm_loss,
            "multitoken_loss_dict": multitoken_loss_dict,
            "aux_loss": total_aux_loss if self.training else 0.0,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: (batch, seq_len) - Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Keep top tokens with cumulative probability >= p (nucleus sampling)

        Returns:
            generated: (batch, seq_len + max_new_tokens) - Generated token IDs
        """
        for _ in range(max_new_tokens):
            # Get predictions for the last position
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        embedding_params = sum(p.numel() for p in self.token_embedding.parameters())
        layer_params = sum(p.numel() for p in self.layers.parameters())
        norm_params = sum(p.numel() for p in self.final_norm.parameters())
        lm_head_params = sum(p.numel() for p in self.lm_heads.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        # Count MoE vs standard layers, Mamba vs attention layers
        moe_layer_params = 0
        standard_layer_params = 0
        n_moe_layers = 0
        n_standard_layers = 0
        n_mamba_layers = 0
        n_attention_layers = 0

        for layer in self.layers:
            layer_param_count = sum(p.numel() for p in layer.parameters())
            if layer.use_moe:
                moe_layer_params += layer_param_count
                n_moe_layers += 1
            else:
                standard_layer_params += layer_param_count
                n_standard_layers += 1

            if layer.use_mamba:
                n_mamba_layers += 1
            else:
                n_attention_layers += 1

        # Calculate active parameters (accounting for MoE sparsity)
        active_params = self.count_active_parameters()

        return {
            "embedding": embedding_params,
            "layers": layer_params,
            "moe_layers": moe_layer_params,
            "standard_layers": standard_layer_params,
            "n_moe_layers": n_moe_layers,
            "n_standard_layers": n_standard_layers,
            "n_mamba_layers": n_mamba_layers,
            "n_attention_layers": n_attention_layers,
            "final_norm": norm_params,
            "lm_heads": lm_head_params,
            "total": total_params,
            "total_billions": total_params / 1e9,
            "active": active_params,
            "active_billions": active_params / 1e9,
            "sparsity": 1.0 - (active_params / total_params),
        }

    def count_active_parameters(self) -> int:
        """
        Count active parameters per forward pass.

        For MoE models, only top-k experts are active per token, so we count:
        - All non-expert parameters (always active)
        - Only top-k/n_experts fraction of expert parameters

        Note: MoD reduces compute but doesn't affect parameter count
        (all parameters are still "active", just used on fewer tokens).
        """
        # Always active: embeddings, norms, LM heads
        active_params = 0
        active_params += sum(p.numel() for p in self.token_embedding.parameters())
        active_params += sum(p.numel() for p in self.final_norm.parameters())
        active_params += sum(p.numel() for p in self.lm_heads.parameters())

        # Layer parameters
        for layer in self.layers:
            # Sequence layer (attention or mamba) - always active
            active_params += sum(p.numel() for p in layer.seq_norm.parameters())
            active_params += sum(p.numel() for p in layer.seq_layer.parameters())

            # FFN layer
            active_params += sum(p.numel() for p in layer.ffn_norm.parameters())

            if layer.use_moe and hasattr(layer.ffn, 'experts'):
                # MoE layer: only top-k experts active per token
                # Router is always active
                if hasattr(layer.ffn, 'router'):
                    active_params += sum(p.numel() for p in layer.ffn.router.parameters())

                # Expert parameters: only top-k/n_experts fraction active
                expert_params = sum(p.numel() for p in layer.ffn.experts.parameters())
                active_expert_params = int(expert_params * (self.config.moe_top_k / self.config.n_experts))
                active_params += active_expert_params
            else:
                # Standard FFN: all parameters active
                active_params += sum(p.numel() for p in layer.ffn.parameters())

            # MoD routers (if enabled) - always active
            if layer.use_mod:
                if layer.seq_mod_router is not None:
                    active_params += sum(p.numel() for p in layer.seq_mod_router.parameters())
                if layer.ffn_mod_router is not None:
                    active_params += sum(p.numel() for p in layer.ffn_mod_router.parameters())

            # Dropout (no parameters)

        return active_params


if __name__ == "__main__":
    # Test MoETransformer
    from .config import get_test_config

    config = get_test_config()
    config.use_flash_attention = False
    config.n_layers = 4  # Small for testing

    print("Testing MoETransformer...")
    print(f"Config: {config.n_layers} layers, d_model={config.d_model}, vocab_size={config.vocab_size}")

    # Create model
    model = MoETransformer(config)

    # Count parameters
    params = model.count_parameters()
    print("\nParameter counts:")
    for key, value in params.items():
        if isinstance(value, int) and value > 1000:
            print(f"  {key}: {value/1e6:.2f}M")
        else:
            print(f"  {key}: {value}")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    outputs = model(input_ids)
    logits = outputs["logits"]
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # Test with labels (compute loss)
    print("\nTesting with labels...")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    print(f"Loss: {loss.item():.4f}")
    assert loss is not None

    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    print("Backward pass successful!")

    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    print(f"Prompt shape: {prompt.shape}")
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    print(f"Generated shape: {generated.shape}")
    assert generated.shape == (1, 30)  # 10 + 20

    print("\nAll MoETransformer tests passed!")
