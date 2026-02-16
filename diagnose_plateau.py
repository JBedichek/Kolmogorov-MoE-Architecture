#!/usr/bin/env python3
"""
Diagnostic script to debug loss plateau issues.
Checks router behavior, expert utilization, gradient magnitudes, and loss components.

Usage:
    python diagnose_plateau.py --config configs/production_training.yaml
    python diagnose_plateau.py --config configs/production_training.yaml --checkpoint checkpoints/step_1000.pt
"""

import torch
import torch.nn as nn
import yaml
import argparse
from collections import defaultdict
import numpy as np

from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model_from_config(config_dict: dict) -> MoETransformer:
    """Create model from config dict."""
    model_config = config_dict.get('model', {})

    # Map YAML config to AdvancedMoEConfig
    config = AdvancedMoEConfig(
        vocab_size=model_config.get('vocab_size', 50000),
        d_model=model_config.get('d_model', 2048),
        n_layers=model_config.get('n_layers', 32),
        n_heads=model_config.get('n_heads', 16),
        n_kv_heads=model_config.get('n_kv_heads', 4),
        head_dim=model_config.get('head_dim', 128),
        d_ff=model_config.get('d_ff', 5632),
        d_ff_expert=model_config.get('d_ff_expert', 2816),
        max_seq_len=model_config.get('max_seq_len', 2048),
        n_experts=model_config.get('n_experts', 16),
        moe_top_k=model_config.get('moe_top_k', 2),
        moe_capacity_factor=model_config.get('moe_capacity_factor', 1.25),
        moe_load_balance_loss_weight=model_config.get('moe_load_balance_loss_weight', 0.01),
        moe_router_z_loss_weight=model_config.get('moe_router_z_loss_weight', 0.001),
        moe_layers=tuple(model_config.get('moe_layers', [])),
        moe_implementation=model_config.get('moe_implementation', 'batched'),
        mod_enabled=model_config.get('mod_enabled', True),
        mod_capacity_factor=model_config.get('mod_capacity_factor', 0.75),
        mod_router_hidden_dim=model_config.get('mod_router_hidden_dim', 128),
        mod_load_balance_loss_weight=model_config.get('mod_load_balance_loss_weight', 0.001),
        mamba_enabled=model_config.get('mamba_enabled', False),
        mamba_layers=tuple(model_config.get('mamba_layers', [])),
        n_pred_tokens=model_config.get('n_pred_tokens', 1),
        aux_loss_weights=tuple(model_config.get('aux_loss_weights', [1.0])),
        use_flash_attention=model_config.get('use_flash_attention', True),
        rope_theta=model_config.get('rope_theta', 10000.0),
        norm_type=model_config.get('norm_type', 'rmsnorm'),
        ffn_activation=model_config.get('ffn_activation', 'swiglu'),
        dropout=model_config.get('dropout', 0.1),
        attention_dropout=model_config.get('attention_dropout', 0.1),
        residual_dropout=model_config.get('residual_dropout', 0.1),
    )

    return MoETransformer(config), config


class RouterAnalyzer:
    """Hooks into routers to analyze their behavior."""

    def __init__(self):
        self.stats = defaultdict(dict)
        self.hooks = []

    def register_hooks(self, model, n_experts):
        """Register forward hooks on all routers."""
        self.n_experts = n_experts

        for name, module in model.named_modules():
            # Hook MoE routers - look for the router's output projection
            if 'moe' in name.lower() and 'router' in name.lower():
                self.hooks.append(
                    module.register_forward_hook(self._make_moe_hook(name))
                )
            # Hook MoD routers
            elif 'mod' in name.lower() and 'router' in name.lower():
                self.hooks.append(
                    module.register_forward_hook(self._make_mod_hook(name))
                )

    def _make_moe_hook(self, name):
        def hook(module, input, output):
            # Output could be various formats depending on implementation
            if isinstance(output, tuple):
                logits = output[0] if len(output) > 0 else None
            elif isinstance(output, torch.Tensor):
                logits = output
            else:
                return

            if logits is None or not isinstance(logits, torch.Tensor):
                return

            with torch.no_grad():
                # Router logits statistics
                self.stats[name]['logits_mean'] = logits.float().mean().item()
                self.stats[name]['logits_std'] = logits.float().std().item()
                self.stats[name]['logits_max'] = logits.float().max().item()
                self.stats[name]['logits_min'] = logits.float().min().item()

                # Entropy (higher = more uniform routing)
                probs = torch.softmax(logits.float(), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                max_entropy = np.log(self.n_experts)
                self.stats[name]['entropy'] = entropy.item()
                self.stats[name]['entropy_ratio'] = entropy.item() / max_entropy

                # Expert selection distribution
                top_k_indices = torch.topk(logits, k=2, dim=-1).indices
                expert_counts = torch.bincount(
                    top_k_indices.flatten(),
                    minlength=self.n_experts
                )
                self.stats[name]['experts_used'] = (expert_counts > 0).sum().item()
                self.stats[name]['expert_counts'] = expert_counts.tolist()

                # Check for collapsed routing (one expert dominates)
                expert_freq = expert_counts.float() / expert_counts.sum()
                max_freq = expert_freq.max().item()
                self.stats[name]['max_expert_freq'] = max_freq
                if max_freq > 0.5:
                    self.stats[name]['collapsed'] = True
        return hook

    def _make_mod_hook(self, name):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return

            with torch.no_grad():
                scores = output.float()
                self.stats[name]['scores_mean'] = scores.mean().item()
                self.stats[name]['scores_std'] = scores.std().item()

                # Selection rate (what fraction of tokens pass)
                if scores.dim() >= 2:
                    probs = torch.sigmoid(scores)
                    self.stats[name]['selection_rate'] = (probs > 0.5).float().mean().item()

                    # Check if router is learning (should have variance)
                    self.stats[name]['prob_std'] = probs.std().item()
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_stats(self):
        return dict(self.stats)


def analyze_gradients(model):
    """Analyze gradient statistics after backward pass."""
    grad_stats = defaultdict(lambda: defaultdict(list))

    for name, param in model.named_parameters():
        # Categorize parameter
        if 'router' in name.lower():
            if 'mod' in name.lower():
                category = 'mod_router'
            else:
                category = 'moe_router'
        elif 'expert' in name.lower():
            category = 'expert'
        elif 'lm_head' in name.lower():
            category = 'lm_head'
        elif 'embed' in name.lower():
            category = 'embedding'
        elif 'attention' in name.lower() or 'attn' in name.lower():
            category = 'attention'
        elif 'norm' in name.lower():
            category = 'norm'
        else:
            category = 'other'

        if param.grad is not None:
            grad = param.grad.float()
            grad_stats[category]['has_grad'].append(True)
            grad_stats[category]['norm'].append(grad.norm().item())
            grad_stats[category]['mean'].append(grad.mean().item())
            grad_stats[category]['max'].append(grad.abs().max().item())
            grad_stats[category]['has_nan'].append(torch.isnan(grad).any().item())
            grad_stats[category]['names'].append(name)
        else:
            grad_stats[category]['has_grad'].append(False)
            grad_stats[category]['norm'].append(0)
            grad_stats[category]['names'].append(name)

    return dict(grad_stats)


def main():
    parser = argparse.ArgumentParser(description='Diagnose training plateau')
    parser.add_argument('--config', type=str, default='configs/production_training.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to analyze')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING PLATEAU DIAGNOSTIC REPORT")
    print("=" * 70)

    # Load config
    config_dict = load_config(args.config)
    print(f"\nConfig: {args.config}")

    # Print key config values
    model_cfg = config_dict.get('model', {})
    training_cfg = config_dict.get('training', {})

    print("\n" + "-" * 70)
    print("KEY CONFIGURATION")
    print("-" * 70)
    print(f"  d_model:                    {model_cfg.get('d_model')}")
    print(f"  n_layers:                   {model_cfg.get('n_layers')}")
    print(f"  n_experts:                  {model_cfg.get('n_experts')}")
    print(f"  moe_top_k:                  {model_cfg.get('moe_top_k')}")
    print(f"  mod_enabled:                {model_cfg.get('mod_enabled')}")
    print(f"  mod_capacity_factor:        {model_cfg.get('mod_capacity_factor')}")
    print(f"  moe_load_balance_weight:    {model_cfg.get('moe_load_balance_loss_weight')}")
    print(f"  max_lr:                     {training_cfg.get('max_lr')}")
    print(f"  batch_size:                 {training_cfg.get('batch_size')}")
    print(f"  gradient_accumulation:      {training_cfg.get('gradient_accumulation_steps')}")
    effective_batch = training_cfg.get('batch_size', 1) * training_cfg.get('gradient_accumulation_steps', 1)
    print(f"  effective_batch_size:       {effective_batch}")

    # Flag potential issues
    print("\n" + "-" * 70)
    print("CONFIGURATION ISSUES")
    print("-" * 70)
    issues = []

    mod_cap = model_cfg.get('mod_capacity_factor', 1.0)
    if mod_cap < 0.5:
        issues.append(f"🚨 CRITICAL: mod_capacity_factor={mod_cap} is EXTREMELY aggressive!")
        issues.append(f"   Only {mod_cap*100:.0f}% of tokens processed per layer ({(1-mod_cap)*100:.0f}% skipped)")
        issues.append(f"   RECOMMENDATION: Start with mod_capacity_factor=0.75 or disable MoD")

    lr = training_cfg.get('max_lr', 0)
    if lr > 0.001:
        issues.append(f"⚠️  max_lr={lr} is high for transformers")
        issues.append(f"   RECOMMENDATION: Use 3e-4 to 6e-4")

    if effective_batch < 16:
        issues.append(f"⚠️  effective_batch_size={effective_batch} is small")
        issues.append(f"   RECOMMENDATION: Increase to 32+ for stable training")

    n_experts = model_cfg.get('n_experts', 16)
    top_k = model_cfg.get('moe_top_k', 2)
    active_ratio = top_k / n_experts
    if active_ratio < 0.05:
        issues.append(f"ℹ️  Expert sparsity: {active_ratio*100:.1f}% active ({top_k}/{n_experts})")
        issues.append(f"   This is very sparse - ensure routers are learning properly")

    if not issues:
        print("  ✓ No obvious configuration issues")
    else:
        for issue in issues:
            print(f"  {issue}")

    # Check GPU memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"\n  GPU memory: {free_mem/1e9:.1f}GB free / {total_mem/1e9:.1f}GB total")
        if free_mem < 20e9:
            print(f"  ⚠️  Low GPU memory - using reduced batch/seq for diagnostics")
            args.batch_size = 1
            args.seq_len = 256

    # Create model
    print("\n" + "-" * 70)
    print("CREATING MODEL")
    print("-" * 70)

    model, config = create_model_from_config(config_dict)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"  ✓ Loaded checkpoint")

    model = model.to(device)
    if device == 'cuda':
        model = model.to(torch.bfloat16)

    # Create dummy input
    vocab_size = model_cfg.get('vocab_size', 50000)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = input_ids.clone()

    # Analyze router behavior
    print("\n" + "-" * 70)
    print("ROUTER ANALYSIS")
    print("-" * 70)

    analyzer = RouterAnalyzer()
    analyzer.register_hooks(model, n_experts)

    model.eval()
    with torch.no_grad():
        try:
            _ = model(input_ids)
        except Exception as e:
            print(f"  Error during forward pass: {e}")
            analyzer.remove_hooks()
            return

    analyzer.remove_hooks()
    router_stats = analyzer.get_stats()

    # Report MoE router stats
    moe_stats = {k: v for k, v in router_stats.items() if 'moe' in k.lower() and 'mod' not in k.lower()}
    if moe_stats:
        print("\n  MoE Routers:")
        collapsed_count = 0
        low_entropy_count = 0

        for name, stats in moe_stats.items():
            short_name = name.split('.')[-2] if '.' in name else name
            print(f"\n    {short_name}:")

            if 'entropy_ratio' in stats:
                er = stats['entropy_ratio']
                print(f"      entropy_ratio: {er:.2f} (1.0 = uniform)")
                if er < 0.5:
                    print(f"      ⚠️  Low entropy - routing may be collapsing")
                    low_entropy_count += 1

            if 'experts_used' in stats:
                used = stats['experts_used']
                print(f"      experts_used: {used}/{n_experts}")
                if used < n_experts * 0.5:
                    print(f"      ⚠️  Many experts unused!")

            if 'max_expert_freq' in stats:
                freq = stats['max_expert_freq']
                print(f"      max_expert_freq: {freq:.1%}")
                if freq > 0.3:
                    print(f"      ⚠️  One expert receiving too many tokens")

            if stats.get('collapsed'):
                collapsed_count += 1
                print(f"      🚨 ROUTER COLLAPSED - one expert dominates!")

        if collapsed_count > 0:
            print(f"\n  🚨 {collapsed_count} routers have collapsed!")
            print(f"     RECOMMENDATION: Increase load_balance_loss_weight")
        if low_entropy_count > len(moe_stats) // 2:
            print(f"\n  ⚠️  {low_entropy_count}/{len(moe_stats)} routers have low entropy")
            print(f"     Routers may not be learning diverse routing")
    else:
        print("  No MoE router hooks captured (check model structure)")

    # Report MoD router stats
    mod_stats = {k: v for k, v in router_stats.items() if 'mod' in k.lower()}
    if mod_stats:
        print("\n  MoD Routers:")
        for name, stats in list(mod_stats.items())[:5]:
            short_name = name.split('.')[-2] if '.' in name else name
            print(f"\n    {short_name}:")

            if 'selection_rate' in stats:
                rate = stats['selection_rate']
                expected = mod_cap
                print(f"      selection_rate: {rate:.1%} (expected: {expected:.1%})")
                if abs(rate - expected) > 0.15:
                    print(f"      ⚠️  Selection rate differs significantly from capacity")

            if 'prob_std' in stats:
                std = stats['prob_std']
                print(f"      prob_std: {std:.3f}")
                if std < 0.1:
                    print(f"      ⚠️  Low variance - router may not be discriminating")

            if 'scores_std' in stats:
                print(f"      scores_std: {stats['scores_std']:.3f}")

    # Analyze loss components
    print("\n" + "-" * 70)
    print("LOSS ANALYSIS")
    print("-" * 70)

    model.train()
    try:
        output = model(input_ids, labels=labels)

        print(f"  total_loss:      {output['loss'].item():.4f}")
        if 'lm_loss' in output:
            print(f"  lm_loss:         {output['lm_loss'].item():.4f}")
        if 'aux_loss' in output and output['aux_loss'] is not None:
            aux = output['aux_loss'].item()
            total = output['loss'].item()
            print(f"  aux_loss:        {aux:.4f} ({aux/total*100:.1f}% of total)")
            if aux / total > 0.3:
                print(f"  ⚠️  Auxiliary loss is large fraction of total!")
                print(f"     LM learning may be hindered")
    except Exception as e:
        print(f"  Error computing loss: {e}")
        return

    # Analyze gradients
    print("\n" + "-" * 70)
    print("GRADIENT ANALYSIS")
    print("-" * 70)

    model.zero_grad()
    output = model(input_ids, labels=labels)
    output['loss'].backward()

    grad_stats = analyze_gradients(model)

    # Check each category
    categories_order = ['moe_router', 'mod_router', 'expert', 'lm_head', 'embedding', 'attention', 'norm', 'other']

    print("\n  Gradient statistics by component:")
    problems = []

    for cat in categories_order:
        if cat not in grad_stats:
            continue

        stats = grad_stats[cat]
        n_total = len(stats['has_grad'])
        n_with_grad = sum(stats['has_grad'])

        if n_total == 0:
            continue

        norms = [n for n, has in zip(stats['norm'], stats['has_grad']) if has]

        status = "✓" if n_with_grad == n_total else "⚠️"
        if n_with_grad == 0:
            status = "🚨"

        print(f"\n    {cat}:")
        print(f"      {status} {n_with_grad}/{n_total} parameters have gradients")

        if norms:
            mean_norm = np.mean(norms)
            max_norm = np.max(norms)
            min_norm = np.min(norms)
            print(f"      norm: mean={mean_norm:.2e}, max={max_norm:.2e}, min={min_norm:.2e}")

            if mean_norm < 1e-7:
                print(f"      🚨 Gradients are extremely small!")
                problems.append(f"{cat}: gradients too small")
            elif mean_norm < 1e-5:
                print(f"      ⚠️  Gradients are small")

        if n_with_grad < n_total:
            missing = [name for name, has in zip(stats['names'], stats['has_grad']) if not has]
            print(f"      Missing gradients for: {missing[:3]}")
            problems.append(f"{cat}: {n_total - n_with_grad} params missing gradients")

    # Check for NaN gradients
    has_nan = any(
        any(stats.get('has_nan', []))
        for stats in grad_stats.values()
    )
    if has_nan:
        print("\n  🚨 NaN gradients detected!")
        problems.append("NaN gradients")

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    if problems:
        print("\n  Issues found:")
        for p in problems:
            print(f"    - {p}")

    print("\n  Recommendations to fix plateau at loss ~8:")
    print()

    rec_num = 1

    if mod_cap < 0.5:
        print(f"  {rec_num}. 🔴 CRITICAL: Increase mod_capacity_factor")
        print(f"     Current: {mod_cap} (skipping {(1-mod_cap)*100:.0f}% of tokens)")
        print(f"     Try: 0.75 or disable MoD entirely (mod_enabled: false)")
        print(f"     This is likely the main cause of plateau!")
        rec_num += 1

    if lr > 0.001:
        print(f"  {rec_num}. Reduce learning rate")
        print(f"     Current: {lr}")
        print(f"     Try: 3e-4 or 6e-4")
        rec_num += 1

    if effective_batch < 32:
        print(f"  {rec_num}. Increase effective batch size")
        print(f"     Current: {effective_batch}")
        print(f"     Try: gradient_accumulation_steps: {max(32 // training_cfg.get('batch_size', 1), 8)}")
        rec_num += 1

    print(f"  {rec_num}. Test without MoD first")
    print(f"     Set mod_enabled: false and verify loss decreases")
    print(f"     If loss improves, gradually re-enable MoD with higher capacity")
    rec_num += 1

    print(f"  {rec_num}. Monitor router entropy during training")
    print(f"     If entropy drops rapidly, increase load_balance_loss_weight")
    rec_num += 1

    if 'moe_router' in grad_stats:
        router_grads = grad_stats['moe_router']
        if any(not g for g in router_grads['has_grad']):
            print(f"  {rec_num}. Check FSDP configuration")
            print(f"     Some router parameters not receiving gradients")
            print(f"     Consider using DDP instead with --use-ddp flag")
            rec_num += 1

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
