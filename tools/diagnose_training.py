#!/usr/bin/env python3
"""
Training Diagnostic Script for MoE Models

Detects:
1. Router collapse (experts not being used evenly)
2. Frozen/stuck parameters (weights not moving from initialization)
3. Gradient flow issues
4. Evaluates on real data to measure actual performance

Usage:
    python tools/diagnose_training.py --checkpoint checkpoints/checkpoint-5000.pt --config configs/1b_150m_active.yaml
"""

import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig


def create_model_config(config_dict: dict) -> AdvancedMoEConfig:
    """Create AdvancedMoEConfig from YAML config dictionary."""
    model_cfg = config_dict['model']

    # Convert moe_layers list to tuple if needed
    moe_layers = model_cfg.get('moe_layers')
    if moe_layers is not None:
        moe_layers = tuple(moe_layers)

    # Convert mamba_layers list to tuple if needed
    mamba_layers = model_cfg.get('mamba_layers', [])
    if mamba_layers is not None:
        mamba_layers = tuple(mamba_layers)

    # Convert aux_loss_weights to tuple if needed
    aux_loss_weights = model_cfg.get('aux_loss_weights', [1.0])
    if aux_loss_weights is not None:
        aux_loss_weights = tuple(aux_loss_weights)

    return AdvancedMoEConfig(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        n_layers=model_cfg['n_layers'],
        n_heads=model_cfg['n_heads'],
        n_kv_heads=model_cfg.get('n_kv_heads', model_cfg['n_heads']),
        head_dim=model_cfg.get('head_dim', model_cfg['d_model'] // model_cfg['n_heads']),
        d_ff=model_cfg.get('d_ff', model_cfg['d_model'] * 4),
        max_seq_len=model_cfg.get('max_seq_len', 2048),
        n_experts=model_cfg.get('n_experts', 1),
        moe_top_k=model_cfg.get('moe_top_k', 2),
        moe_capacity_factor=model_cfg.get('moe_capacity_factor', 1.25),
        d_ff_expert=model_cfg.get('d_ff_expert', model_cfg['d_model'] * 2),
        moe_load_balance_loss_weight=model_cfg.get('moe_load_balance_loss_weight', 0.01),
        moe_router_z_loss_weight=model_cfg.get('moe_router_z_loss_weight', 0.001),
        moe_layers=moe_layers,
        moe_implementation=model_cfg.get('moe_implementation', 'batched'),
        mod_enabled=model_cfg.get('mod_enabled', False),
        mod_capacity_factor=model_cfg.get('mod_capacity_factor', 0.5),
        mod_router_hidden_dim=model_cfg.get('mod_router_hidden_dim', 64),
        mod_load_balance_loss_weight=model_cfg.get('mod_load_balance_loss_weight', 0.01),
        mamba_enabled=model_cfg.get('mamba_enabled', False),
        mamba_layers=mamba_layers,
        n_pred_tokens=model_cfg.get('n_pred_tokens', 1),
        aux_loss_weights=aux_loss_weights,
        use_flash_attention=model_cfg.get('use_flash_attention', True),
        rope_theta=model_cfg.get('rope_theta', 10000.0),
        dropout=model_cfg.get('dropout', 0.0),
        attention_dropout=model_cfg.get('attention_dropout', 0.0),
        residual_dropout=model_cfg.get('residual_dropout', 0.0),
    )


def load_checkpoint(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load model from checkpoint."""
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model config
    model_config = create_model_config(config)

    # Create model
    model = MoETransformer(model_config).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config, checkpoint


def create_fresh_model(config: dict, device: str = 'cuda'):
    """Create a freshly initialized model for comparison."""
    model_config = create_model_config(config)
    model = MoETransformer(model_config).to(device)
    return model


def analyze_router_collapse(model, n_samples: int = 100, seq_len: int = 512, device: str = 'cuda'):
    """
    Analyze router behavior to detect collapse.

    Router collapse occurs when the router consistently selects the same experts,
    ignoring most of the expert capacity.
    """
    print("\n" + "="*60)
    print("ROUTER COLLAPSE ANALYSIS")
    print("="*60)

    model.eval()
    vocab_size = model.token_embedding.embedding.weight.shape[0]

    # Collect router decisions across multiple random inputs
    expert_counts = defaultdict(lambda: defaultdict(int))  # layer -> expert -> count
    expert_probs = defaultdict(list)  # layer -> list of prob distributions

    with torch.no_grad():
        for i in range(n_samples):
            # Random input
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

            # Get hidden states through the model
            hidden = model.token_embedding(input_ids)

            for layer_idx, layer in enumerate(model.layers):
                # Check if this is an MoE layer (layer.ffn is MoELayer)
                is_moe = hasattr(layer, 'use_moe') and layer.use_moe

                if is_moe and hasattr(layer.ffn, 'router') and layer.ffn.router is not None:
                    moe_layer = layer.ffn

                    # Apply pre-FFN norm
                    normed_hidden = layer.ffn_norm(hidden)

                    # Get router output
                    routing_weights, selected_experts, router_logits = moe_layer.router(normed_hidden)
                    # router_logits: (batch, seq, n_experts)

                    router_probs = F.softmax(router_logits, dim=-1)

                    # Get top-k selections (already in selected_experts)
                    top_k = moe_layer.top_k

                    # Count expert usage
                    for expert_idx in selected_experts.flatten().tolist():
                        expert_counts[layer_idx][expert_idx] += 1

                    # Store mean probability distribution
                    expert_probs[layer_idx].append(router_probs.mean(dim=(0, 1)).cpu().numpy())

                # Forward through layer to get next hidden state
                output = layer(hidden)
                hidden = output if isinstance(output, torch.Tensor) else output[0]

    # Analyze results - find first MoE layer to get n_experts/top_k
    n_experts = 1
    top_k = 1
    for layer in model.layers:
        if hasattr(layer, 'use_moe') and layer.use_moe and hasattr(layer.ffn, 'n_experts'):
            n_experts = layer.ffn.n_experts
            top_k = layer.ffn.top_k
            break

    total_selections = n_samples * seq_len * top_k
    expected_per_expert = total_selections / n_experts

    collapse_detected = False

    for layer_idx in sorted(expert_counts.keys()):
        counts = expert_counts[layer_idx]
        probs = np.mean(expert_probs[layer_idx], axis=0)

        # Sort by usage
        sorted_experts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Calculate metrics
        usage_fractions = [c / total_selections for _, c in sorted_experts]
        max_usage = max(usage_fractions)
        min_usage = min(usage_fractions) if len(usage_fractions) == n_experts else 0

        # Gini coefficient for inequality
        usage_array = np.array([counts.get(i, 0) for i in range(n_experts)])
        gini = calculate_gini(usage_array)

        # Entropy of probability distribution
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(n_experts)
        normalized_entropy = entropy / max_entropy

        # Detect collapse: if top expert gets >3x expected or Gini > 0.5
        is_collapsed = max_usage > 3 / n_experts or gini > 0.5 or normalized_entropy < 0.7

        status = "⚠️  COLLAPSED" if is_collapsed else "✓ Healthy"
        if is_collapsed:
            collapse_detected = True

        print(f"\nLayer {layer_idx}: {status}")
        print(f"  Expected per expert: {expected_per_expert:.0f} ({100/n_experts:.1f}%)")
        print(f"  Top 3 experts: {sorted_experts[:3]}")
        print(f"  Bottom 3 experts: {sorted_experts[-3:]}")
        print(f"  Max usage: {max_usage*100:.1f}% | Min usage: {min_usage*100:.1f}%")
        print(f"  Gini coefficient: {gini:.3f} (0=equal, 1=one expert)")
        print(f"  Router entropy: {normalized_entropy:.3f} (1=uniform, 0=collapsed)")
        print(f"  Mean probs: {probs[:4]}... (showing first 4)")

    print("\n" + "-"*60)
    if collapse_detected:
        print("⚠️  ROUTER COLLAPSE DETECTED!")
        print("   Some layers have highly imbalanced expert usage.")
        print("   This can cause loss plateaus and poor generalization.")
        print("\n   Potential fixes:")
        print("   1. Increase load balancing loss weight (moe_load_balance_loss_weight)")
        print("   2. Add router z-loss (moe_router_z_loss_weight)")
        print("   3. Lower learning rate for router")
        print("   4. Use auxiliary loss to encourage diversity")
    else:
        print("✓ Router appears healthy - experts are being used relatively evenly")

    return collapse_detected, expert_counts


def calculate_gini(array):
    """Calculate Gini coefficient for measuring inequality."""
    array = np.array(array, dtype=float)
    if array.sum() == 0:
        return 0
    array = np.sort(array)
    n = len(array)
    cumsum = np.cumsum(array)
    return (2 * np.sum((np.arange(1, n+1) * array)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def analyze_parameter_movement(trained_model, fresh_model, device: str = 'cuda'):
    """
    Compare trained model to freshly initialized model.
    Detect parameters that haven't moved from initialization.
    """
    print("\n" + "="*60)
    print("PARAMETER MOVEMENT ANALYSIS")
    print("="*60)

    trained_state = trained_model.state_dict()
    fresh_state = fresh_model.state_dict()

    stuck_params = []
    moved_params = []

    total_params = 0
    stuck_param_count = 0

    for name in trained_state.keys():
        if name not in fresh_state:
            continue

        trained_param = trained_state[name]
        fresh_param = fresh_state[name]

        if trained_param.shape != fresh_param.shape:
            continue

        param_count = trained_param.numel()
        total_params += param_count

        # Calculate relative change
        diff = (trained_param.float() - fresh_param.float()).abs()
        fresh_norm = fresh_param.float().abs().mean()

        if fresh_norm > 1e-8:
            relative_change = diff.mean() / fresh_norm
        else:
            relative_change = diff.mean()

        # Consider stuck if change is very small
        is_stuck = relative_change < 1e-6

        if is_stuck:
            stuck_params.append((name, param_count, relative_change.item()))
            stuck_param_count += param_count
        else:
            moved_params.append((name, param_count, relative_change.item()))

    # Report
    print(f"\nTotal parameters analyzed: {total_params:,}")
    print(f"Stuck parameters: {stuck_param_count:,} ({100*stuck_param_count/total_params:.2f}%)")
    print(f"Moved parameters: {total_params - stuck_param_count:,} ({100*(1-stuck_param_count/total_params):.2f}%)")

    if stuck_params:
        print(f"\n⚠️  STUCK PARAMETERS ({len(stuck_params)} tensors):")
        for name, count, change in stuck_params[:20]:  # Show first 20
            print(f"  {name}: {count:,} params, change={change:.2e}")
        if len(stuck_params) > 20:
            print(f"  ... and {len(stuck_params) - 20} more")

    # Show parameters with largest changes
    print(f"\n✓ MOST CHANGED PARAMETERS (top 10):")
    moved_params.sort(key=lambda x: x[2], reverse=True)
    for name, count, change in moved_params[:10]:
        print(f"  {name}: {count:,} params, change={change:.2e}")

    return stuck_params, moved_params


def analyze_trainability(model):
    """
    Check that all parameters are set to trainable and have proper gradient settings.
    """
    print("\n" + "="*60)
    print("TRAINABILITY ANALYSIS")
    print("="*60)

    trainable_count = 0
    frozen_count = 0
    trainable_params = 0
    frozen_params = 0

    frozen_list = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            trainable_params += param.numel()
        else:
            frozen_count += 1
            frozen_params += param.numel()
            frozen_list.append((name, param.numel()))

    print(f"\nTrainable tensors: {trainable_count} ({trainable_params:,} params)")
    print(f"Frozen tensors: {frozen_count} ({frozen_params:,} params)")

    if frozen_list:
        print(f"\n⚠️  FROZEN PARAMETERS:")
        for name, count in frozen_list:
            print(f"  {name}: {count:,} params")
    else:
        print("\n✓ All parameters are trainable")

    return frozen_list


def run_evaluation(model, config: dict, n_batches: int = 50, device: str = 'cuda'):
    """
    Run evaluation on real data to measure actual performance.
    """
    print("\n" + "="*60)
    print("EVALUATION ON REAL DATA")
    print("="*60)

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("⚠️  Cannot run evaluation: datasets or transformers not installed")
        return None

    data_cfg = config.get('data', {})
    tokenizer_name = data_cfg.get('tokenizer_name', 'meta-llama/Llama-2-7b-hf')

    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading validation data (FineWeb sample)...")
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            "sample-10BT",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"⚠️  Could not load FineWeb: {e}")
        print("    Falling back to wikitext...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=True)

    model.eval()
    max_seq_len = config['model'].get('max_seq_len', 2048)

    total_loss = 0
    total_tokens = 0
    losses = []

    print(f"\nRunning evaluation on {n_batches} batches...")

    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= n_batches:
                break

            text = example.get('text', example.get('content', ''))
            if not text or len(text) < 100:
                continue

            # Tokenize
            tokens = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_seq_len,
                padding=False
            )

            input_ids = tokens['input_ids'].to(device)

            if input_ids.shape[1] < 10:
                continue

            # Forward pass
            outputs = model(input_ids, labels=input_ids, return_logits=False)
            loss = outputs['loss'].item()

            if not np.isnan(loss) and not np.isinf(loss):
                losses.append(loss)
                total_loss += loss * input_ids.shape[1]
                total_tokens += input_ids.shape[1]

            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{n_batches}: loss={loss:.4f}")

    if not losses:
        print("⚠️  No valid batches processed")
        return None

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    print(f"\n{'='*40}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"  Batches processed: {len(losses)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Loss std: {np.std(losses):.4f}")
    print(f"  Loss range: [{min(losses):.4f}, {max(losses):.4f}]")

    # Assessment
    print(f"\n{'Assessment':}")
    if perplexity > 500:
        print("  ⚠️  Very high perplexity - model may not be learning")
    elif perplexity > 200:
        print("  ⚠️  High perplexity - limited learning")
    elif perplexity > 50:
        print("  ✓ Moderate perplexity - some learning occurring")
    else:
        print("  ✓ Good perplexity - model is learning")

    return {
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'n_batches': len(losses),
        'total_tokens': total_tokens
    }


def test_generation(model, config: dict, device: str = 'cuda'):
    """
    Test generation to see if model produces coherent output.
    """
    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("⚠️  Cannot run generation test: transformers not installed")
        return

    data_cfg = config.get('data', {})
    tokenizer_name = data_cfg.get('tokenizer_name', 'meta-llama/Llama-2-7b-hf')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "The quick brown fox",
        "In the year 2024,",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    model.eval()

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

        # Simple greedy generation
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(30):
                outputs = model(generated, return_logits=True)
                logits = outputs['logits']
                next_token = logits[0, -1, :].argmax()
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                if next_token == tokenizer.eos_token_id:
                    break

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: '{output_text}'")

        # Check for repetition
        tokens = generated[0].tolist()
        unique_tokens = len(set(tokens))
        repetition_ratio = 1 - (unique_tokens / len(tokens))

        if repetition_ratio > 0.5:
            print(f"  ⚠️  High repetition detected ({repetition_ratio:.1%})")
        elif len(set(output_text.split())) < 5:
            print(f"  ⚠️  Very limited vocabulary in output")


def check_gradient_flow(model, config: dict, device: str = 'cuda'):
    """
    Do a forward-backward pass and check gradient flow.
    """
    print("\n" + "="*60)
    print("GRADIENT FLOW ANALYSIS")
    print("="*60)

    model.train()
    vocab_size = model.token_embedding.embedding.weight.shape[0]
    max_seq_len = config['model'].get('max_seq_len', 2048)

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (1, 256), device=device)

    # Forward pass
    outputs = model(input_ids, labels=input_ids, return_logits=False)
    loss = outputs['loss']

    # Backward pass
    loss.backward()

    # Analyze gradients
    no_grad_params = []
    zero_grad_params = []
    normal_grad_params = []

    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad_params.append((name, param.numel()))
        elif param.grad.abs().max() == 0:
            zero_grad_params.append((name, param.numel()))
        else:
            grad_norm = param.grad.norm().item()
            normal_grad_params.append((name, param.numel(), grad_norm))

    print(f"\nParameters with gradients: {len(normal_grad_params)}")
    print(f"Parameters with zero gradients: {len(zero_grad_params)}")
    print(f"Parameters with no gradients: {len(no_grad_params)}")

    if no_grad_params:
        print(f"\n⚠️  NO GRADIENT FLOW to these parameters:")
        for name, count in no_grad_params[:10]:
            print(f"  {name}: {count:,} params")

    if zero_grad_params:
        print(f"\n⚠️  ZERO GRADIENTS for these parameters:")
        for name, count in zero_grad_params[:10]:
            print(f"  {name}: {count:,} params")

    # Show gradient statistics
    if normal_grad_params:
        print(f"\nGradient norm statistics:")
        norms = [g[2] for g in normal_grad_params]
        print(f"  Mean: {np.mean(norms):.2e}")
        print(f"  Std:  {np.std(norms):.2e}")
        print(f"  Max:  {max(norms):.2e}")
        print(f"  Min:  {min(norms):.2e}")

        # Show largest gradients
        normal_grad_params.sort(key=lambda x: x[2], reverse=True)
        print(f"\nLargest gradient norms:")
        for name, count, norm in normal_grad_params[:5]:
            print(f"  {name}: {norm:.2e}")

        print(f"\nSmallest gradient norms:")
        for name, count, norm in normal_grad_params[-5:]:
            print(f"  {name}: {norm:.2e}")

    # Clear gradients
    model.zero_grad()
    model.eval()

    return no_grad_params, zero_grad_params


def main():
    parser = argparse.ArgumentParser(description="Diagnose MoE training issues")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation on real data')
    parser.add_argument('--skip-generation', action='store_true', help='Skip generation test')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load models
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)

    trained_model, config, checkpoint = load_checkpoint(args.checkpoint, args.config, device)
    fresh_model = create_fresh_model(config, device)

    step = checkpoint.get('step', 'unknown')
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"\nCheckpoint info:")
    print(f"  Step: {step}")
    print(f"  Epoch: {epoch}")

    # Run diagnostics
    print("\n" + "#"*60)
    print("# RUNNING DIAGNOSTICS")
    print("#"*60)

    # 1. Router collapse
    collapse_detected, expert_counts = analyze_router_collapse(trained_model, device=device)

    # 2. Parameter movement
    stuck_params, moved_params = analyze_parameter_movement(trained_model, fresh_model, device)

    # 3. Trainability
    frozen_params = analyze_trainability(trained_model)

    # 4. Gradient flow
    no_grad, zero_grad = check_gradient_flow(trained_model, config, device)

    # 5. Evaluation (optional)
    if not args.skip_eval:
        eval_results = run_evaluation(trained_model, config, device=device)

    # 6. Generation test (optional)
    if not args.skip_generation:
        test_generation(trained_model, config, device)

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    issues = []

    if collapse_detected:
        issues.append("Router collapse detected - experts are not being used evenly")

    if stuck_params:
        issues.append(f"{len(stuck_params)} parameter tensors haven't moved from initialization")

    if frozen_params:
        issues.append(f"{len(frozen_params)} parameter tensors are frozen (requires_grad=False)")

    if no_grad:
        issues.append(f"{len(no_grad)} parameters received no gradients")

    if zero_grad:
        issues.append(f"{len(zero_grad)} parameters received zero gradients")

    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\n📋 RECOMMENDED ACTIONS:")
        if collapse_detected:
            print("  • Increase moe_load_balance_loss_weight (try 0.05)")
            print("  • Add/increase moe_router_z_loss_weight (try 0.01)")
            print("  • Consider using expert choice routing instead of token choice")
        if stuck_params or no_grad or zero_grad:
            print("  • Check that all model components are connected properly")
            print("  • Verify loss includes contributions from all experts")
            print("  • Check if aux_loss is being computed and added correctly")
    else:
        print("\n✓ No major issues detected")
        print("  If loss is still plateauing, consider:")
        print("  • Adjusting learning rate (try lower or use warmup)")
        print("  • Increasing model capacity")
        print("  • Checking data quality and diversity")


if __name__ == "__main__":
    main()
