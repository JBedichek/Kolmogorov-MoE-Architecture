"""
Sparse Distillation: Convert dense models to MoE by distilling FFN layers.

This module provides tools to convert a pretrained dense transformer into a
sparse Mixture-of-Experts model while preserving its learned representations.

Key idea:
- Each MoE layer learns to approximate the input-output mapping of the
  corresponding dense FFN layer
- Training is done via MSE loss: minimize ||MoE(x) - FFN_dense(x)||²
- Uses streaming approach: no disk caching, dense model generates targets on-the-fly

Training modes:
- Parallel: Train all layers simultaneously (default)
- Sequential: Train one layer at a time (low memory)
- Dynamic: Variable-sized experts with loss-based reallocation

Usage:
    from Experiments.sparse_distillation import SparseDistillationTrainer
    from Experiments.sparse_distillation.distill import DistillationConfig

    config = DistillationConfig(
        n_experts=16,
        dynamic_experts=True,  # Enable dynamic reallocation
    )
    trainer = SparseDistillationTrainer(
        dense_model=dense_model,
        config=config,
    )
    trainer.distill(dataloader, epochs=3)
"""

from .distill import SparseDistillationTrainer, DistillationConfig
from .hooks import FFNActivationCache

# Dynamic expert reallocation (optional)
try:
    from .dynamic_experts import (
        DynamicExpertLayer,
        DynamicExpertTracker,
        DynamicExpertConfig,
        interpolate_expert_weights,
    )
    __all__ = [
        'SparseDistillationTrainer',
        'DistillationConfig',
        'FFNActivationCache',
        'DynamicExpertLayer',
        'DynamicExpertTracker',
        'DynamicExpertConfig',
        'interpolate_expert_weights',
    ]
except ImportError:
    __all__ = ['SparseDistillationTrainer', 'DistillationConfig', 'FFNActivationCache']
