#!/usr/bin/env python3
"""
Fine-tune and evaluate a sparse distilled model.

Usage:
    python Experiments/sparse_distillation/finetune_eval.py \
        --model meta-llama/Llama-3.2-1B \
        --checkpoint ./sparse_distillation_output/distilled_moe_layers.pt \
        --dataset wikitext \
        --epochs 1 \
        --eval_only  # Skip fine-tuning, just evaluate
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import math

# Increase recursion limit for very deep models (140+ layers)
sys.setrecursionlimit(5000)


def compute_load_balancing_loss(router_logits: torch.Tensor, expert_counts: torch.Tensor) -> torch.Tensor:
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
    if router_logits.dim() == 3:
        router_logits = router_logits.view(-1, router_logits.shape[-1])
    probs = F.softmax(router_logits, dim=-1)
    P = probs.mean(dim=0)

    # Load balancing loss
    n_experts = router_logits.shape[-1]
    lb_loss = n_experts * (f * P).sum()
    return lb_loss


def compute_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
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


def compute_routing_stats(moe_layers, verbose=True):
    """
    Compute and optionally print routing statistics for all MoE layers.

    Returns a dict with per-layer and aggregate statistics to detect:
    - Load imbalance (some experts used much more than others)
    - Collapse (routing converging to very few experts)

    Args:
        moe_layers: ModuleDict of MoE layers (keyed by layer index)
        verbose: If True, print detailed stats

    Returns:
        Dict with routing statistics
    """
    stats = {
        'per_layer': {},
        'aggregate': {},
    }

    all_entropies = []
    all_min_fracs = []
    all_max_fracs = []
    all_active_experts = []
    collapsed_layers = []

    for layer_idx, moe_layer in sorted(moe_layers.items(), key=lambda x: int(x[0])):
        layer_stats = {}
        n_experts = moe_layer.n_experts

        # Get expert counts from last forward pass
        if hasattr(moe_layer, '_last_expert_counts') and moe_layer._last_expert_counts is not None:
            expert_counts = moe_layer._last_expert_counts.float()
            total_tokens = expert_counts.sum()

            if total_tokens > 0:
                # Expert usage fractions
                fracs = expert_counts / total_tokens

                # Entropy (higher = more balanced, max = log(n_experts))
                # Use small epsilon to avoid log(0)
                probs = fracs + 1e-10
                entropy = -(probs * torch.log(probs)).sum().item()
                max_entropy = math.log(n_experts)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

                # Min/max usage
                min_frac = fracs.min().item()
                max_frac = fracs.max().item()

                # Count active experts (used at least once)
                active_experts = (expert_counts > 0).sum().item()

                # Detect collapse: if fewer than 50% of experts are meaningfully used
                # "Meaningful" = getting at least 10% of uniform share
                uniform_share = 1.0 / n_experts
                meaningful_threshold = 0.1 * uniform_share
                meaningful_experts = (fracs > meaningful_threshold).sum().item()

                layer_stats = {
                    'n_experts': n_experts,
                    'total_tokens': int(total_tokens.item()),
                    'expert_counts': expert_counts.cpu().tolist(),
                    'expert_fracs': fracs.cpu().tolist(),
                    'entropy': entropy,
                    'max_entropy': max_entropy,
                    'normalized_entropy': normalized_entropy,
                    'min_frac': min_frac,
                    'max_frac': max_frac,
                    'imbalance_ratio': max_frac / (min_frac + 1e-10),
                    'active_experts': active_experts,
                    'meaningful_experts': meaningful_experts,
                    'collapsed': meaningful_experts < n_experts * 0.5,
                }

                all_entropies.append(normalized_entropy)
                all_min_fracs.append(min_frac)
                all_max_fracs.append(max_frac)
                all_active_experts.append(active_experts / n_experts)

                if layer_stats['collapsed']:
                    collapsed_layers.append(int(layer_idx))
        else:
            layer_stats = {'n_experts': n_experts, 'no_data': True}

        stats['per_layer'][layer_idx] = layer_stats

    # Aggregate statistics
    if all_entropies:
        stats['aggregate'] = {
            'mean_normalized_entropy': sum(all_entropies) / len(all_entropies),
            'min_normalized_entropy': min(all_entropies),
            'mean_active_expert_frac': sum(all_active_experts) / len(all_active_experts),
            'worst_min_frac': min(all_min_fracs),
            'worst_max_frac': max(all_max_fracs),
            'collapsed_layers': collapsed_layers,
            'n_collapsed': len(collapsed_layers),
        }

    # Print stats if verbose
    if verbose and all_entropies:
        print("\n" + "=" * 70)
        print("ROUTING STATISTICS")
        print("=" * 70)

        # Per-layer summary table
        print(f"{'Layer':>6} | {'Experts':>8} | {'Active':>7} | {'Entropy':>8} | {'Min%':>6} | {'Max%':>6} | {'Imbal':>7} | {'Status':>10}")
        print("-" * 70)

        for layer_idx in sorted(stats['per_layer'].keys(), key=lambda x: int(x)):
            ls = stats['per_layer'][layer_idx]
            if 'no_data' in ls:
                print(f"{layer_idx:>6} | {'N/A':>8} | {'N/A':>7} | {'N/A':>8} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {'no data':>10}")
            else:
                status = "COLLAPSED" if ls['collapsed'] else "OK"
                if ls['normalized_entropy'] < 0.7:
                    status = "WARNING" if status == "OK" else status
                print(f"{layer_idx:>6} | {ls['n_experts']:>8} | {ls['active_experts']:>7} | {ls['normalized_entropy']:>8.3f} | {ls['min_frac']*100:>5.1f}% | {ls['max_frac']*100:>5.1f}% | {ls['imbalance_ratio']:>7.1f}x | {status:>10}")

        print("-" * 70)

        # Aggregate summary
        agg = stats['aggregate']
        print(f"\nAggregate Statistics:")
        print(f"  Mean normalized entropy: {agg['mean_normalized_entropy']:.3f} (1.0 = perfect balance)")
        print(f"  Min normalized entropy:  {agg['min_normalized_entropy']:.3f}")
        print(f"  Mean active expert %:    {agg['mean_active_expert_frac']*100:.1f}%")
        print(f"  Worst min expert usage:  {agg['worst_min_frac']*100:.2f}%")
        print(f"  Worst max expert usage:  {agg['worst_max_frac']*100:.1f}%")

        if agg['n_collapsed'] > 0:
            print(f"\n  ⚠️  WARNING: {agg['n_collapsed']} layers have COLLAPSED routing!")
            print(f"      Collapsed layers: {agg['collapsed_layers']}")
            print(f"      Consider increasing load_balance_weight or checking initialization.")
        elif agg['min_normalized_entropy'] < 0.5:
            print(f"\n  ⚠️  WARNING: Some layers have low entropy (poor balance).")
            print(f"      Consider increasing load_balance_weight.")
        else:
            print(f"\n  ✓ Routing appears healthy.")

        print("=" * 70 + "\n")

    return stats


def reset_routing_stats(moe_layers):
    """Reset routing statistics accumulators in MoE layers."""
    for moe_layer in moe_layers.values():
        if hasattr(moe_layer, '_last_expert_counts'):
            moe_layer._last_expert_counts = None
        if hasattr(moe_layer, '_last_router_logits'):
            moe_layer._last_router_logits = None
        # Reset accumulators
        if hasattr(moe_layer, '_accumulated_expert_counts'):
            moe_layer._accumulated_expert_counts = None


def start_routing_accumulation(moe_layers):
    """Start accumulating routing statistics across forward passes."""
    for moe_layer in moe_layers.values():
        moe_layer._accumulated_expert_counts = None
        moe_layer._accumulation_enabled = True


def stop_routing_accumulation(moe_layers):
    """Stop accumulating routing statistics."""
    for moe_layer in moe_layers.values():
        moe_layer._accumulation_enabled = False


def accumulate_routing_stats(moe_layers):
    """Accumulate current batch's routing stats into accumulators."""
    for moe_layer in moe_layers.values():
        if not getattr(moe_layer, '_accumulation_enabled', False):
            continue

        if hasattr(moe_layer, '_last_expert_counts') and moe_layer._last_expert_counts is not None:
            counts = moe_layer._last_expert_counts.detach().clone()
            if not hasattr(moe_layer, '_accumulated_expert_counts') or moe_layer._accumulated_expert_counts is None:
                moe_layer._accumulated_expert_counts = counts
            else:
                moe_layer._accumulated_expert_counts = moe_layer._accumulated_expert_counts + counts


def compute_routing_stats_accumulated(moe_layers, verbose=True):
    """
    Compute routing statistics from accumulated counts (across multiple batches).
    Falls back to last batch counts if no accumulation is available.
    """
    stats = {
        'per_layer': {},
        'aggregate': {},
    }

    all_entropies = []
    all_min_fracs = []
    all_max_fracs = []
    all_active_experts = []
    collapsed_layers = []

    for layer_idx, moe_layer in sorted(moe_layers.items(), key=lambda x: int(x[0])):
        layer_stats = {}
        n_experts = moe_layer.n_experts

        # Prefer accumulated counts, fall back to last batch
        expert_counts = None
        if hasattr(moe_layer, '_accumulated_expert_counts') and moe_layer._accumulated_expert_counts is not None:
            expert_counts = moe_layer._accumulated_expert_counts.float()
        elif hasattr(moe_layer, '_last_expert_counts') and moe_layer._last_expert_counts is not None:
            expert_counts = moe_layer._last_expert_counts.float()

        if expert_counts is not None:
            total_tokens = expert_counts.sum()

            if total_tokens > 0:
                # Expert usage fractions
                fracs = expert_counts / total_tokens

                # Entropy (higher = more balanced, max = log(n_experts))
                probs = fracs + 1e-10
                entropy = -(probs * torch.log(probs)).sum().item()
                max_entropy = math.log(n_experts)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

                # Min/max usage
                min_frac = fracs.min().item()
                max_frac = fracs.max().item()

                # Count active experts
                active_experts = (expert_counts > 0).sum().item()

                # Detect collapse
                uniform_share = 1.0 / n_experts
                meaningful_threshold = 0.1 * uniform_share
                meaningful_experts = (fracs > meaningful_threshold).sum().item()

                layer_stats = {
                    'n_experts': n_experts,
                    'total_tokens': int(total_tokens.item()),
                    'expert_counts': expert_counts.cpu().tolist(),
                    'expert_fracs': fracs.cpu().tolist(),
                    'entropy': entropy,
                    'max_entropy': max_entropy,
                    'normalized_entropy': normalized_entropy,
                    'min_frac': min_frac,
                    'max_frac': max_frac,
                    'imbalance_ratio': max_frac / (min_frac + 1e-10),
                    'active_experts': active_experts,
                    'meaningful_experts': meaningful_experts,
                    'collapsed': meaningful_experts < n_experts * 0.5,
                }

                all_entropies.append(normalized_entropy)
                all_min_fracs.append(min_frac)
                all_max_fracs.append(max_frac)
                all_active_experts.append(active_experts / n_experts)

                if layer_stats['collapsed']:
                    collapsed_layers.append(int(layer_idx))
        else:
            layer_stats = {'n_experts': n_experts, 'no_data': True}

        stats['per_layer'][layer_idx] = layer_stats

    # Aggregate statistics
    if all_entropies:
        stats['aggregate'] = {
            'mean_normalized_entropy': sum(all_entropies) / len(all_entropies),
            'min_normalized_entropy': min(all_entropies),
            'mean_active_expert_frac': sum(all_active_experts) / len(all_active_experts),
            'worst_min_frac': min(all_min_fracs),
            'worst_max_frac': max(all_max_fracs),
            'collapsed_layers': collapsed_layers,
            'n_collapsed': len(collapsed_layers),
        }

    # Print stats if verbose (same format as compute_routing_stats)
    if verbose and all_entropies:
        print("\n" + "=" * 70)
        print("ROUTING STATISTICS (accumulated)")
        print("=" * 70)

        print(f"{'Layer':>6} | {'Experts':>8} | {'Active':>7} | {'Entropy':>8} | {'Min%':>6} | {'Max%':>6} | {'Imbal':>7} | {'Status':>10}")
        print("-" * 70)

        for layer_idx in sorted(stats['per_layer'].keys(), key=lambda x: int(x)):
            ls = stats['per_layer'][layer_idx]
            if 'no_data' in ls:
                print(f"{layer_idx:>6} | {'N/A':>8} | {'N/A':>7} | {'N/A':>8} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {'no data':>10}")
            else:
                status = "COLLAPSED" if ls['collapsed'] else "OK"
                if ls['normalized_entropy'] < 0.7:
                    status = "WARNING" if status == "OK" else status
                print(f"{layer_idx:>6} | {ls['n_experts']:>8} | {ls['active_experts']:>7} | {ls['normalized_entropy']:>8.3f} | {ls['min_frac']*100:>5.1f}% | {ls['max_frac']*100:>5.1f}% | {ls['imbalance_ratio']:>7.1f}x | {status:>10}")

        print("-" * 70)

        agg = stats['aggregate']
        print(f"\nAggregate Statistics:")
        print(f"  Mean normalized entropy: {agg['mean_normalized_entropy']:.3f} (1.0 = perfect balance)")
        print(f"  Min normalized entropy:  {agg['min_normalized_entropy']:.3f}")
        print(f"  Mean active expert %:    {agg['mean_active_expert_frac']*100:.1f}%")
        print(f"  Worst min expert usage:  {agg['worst_min_frac']*100:.2f}%")
        print(f"  Worst max expert usage:  {agg['worst_max_frac']*100:.1f}%")

        if agg['n_collapsed'] > 0:
            print(f"\n  ⚠️  WARNING: {agg['n_collapsed']} layers have COLLAPSED routing!")
            print(f"      Collapsed layers: {agg['collapsed_layers']}")
        elif agg['min_normalized_entropy'] < 0.5:
            print(f"\n  ⚠️  WARNING: Some layers have low entropy (poor balance).")
        else:
            print(f"\n  ✓ Routing appears healthy.")

        print("=" * 70 + "\n")

    return stats


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def compute_perplexity(model, dataloader, device, pad_token_id=None, max_batches=None, accumulate_routing=True):
    """Compute perplexity on a dataset, optionally accumulating routing stats."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Start routing accumulation if model has MoE layers
    moe_layers = getattr(model, 'moe_layers', None)
    if accumulate_routing and moe_layers is not None:
        start_routing_accumulation(moe_layers)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and i >= max_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            else:
                input_ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                attention_mask = None

            # Shift for causal LM: predict next token
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1].contiguous()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            # Accumulate routing stats after each forward pass
            if accumulate_routing and moe_layers is not None:
                accumulate_routing_stats(moe_layers)

            # Mask out padding tokens in loss computation
            if attention_mask is not None:
                # Only compute loss on non-padded positions
                active_logits = logits[attention_mask.bool()]
                active_labels = labels[attention_mask.bool()]
                if active_labels.numel() == 0:
                    continue
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(active_logits, active_labels)
                n_tokens = active_labels.numel()
            elif pad_token_id is not None:
                # Create mask from pad token
                non_pad_mask = labels != pad_token_id
                active_logits = logits[non_pad_mask]
                active_labels = labels[non_pad_mask]
                if active_labels.numel() == 0:
                    continue
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(active_logits, active_labels)
                n_tokens = active_labels.numel()
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                n_tokens = labels.numel()

            total_loss += loss.item()
            total_tokens += n_tokens

    # Stop accumulation
    if accumulate_routing and moe_layers is not None:
        stop_routing_accumulation(moe_layers)

    if total_tokens == 0:
        return float('inf'), float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return perplexity, avg_loss


def freeze_non_moe_layers(model, moe_only=False):
    """
    Freeze non-MoE layers (attention, embeddings, etc.) for MoE-only training.

    Returns the parameters that should be trained.
    """
    if not moe_only:
        return list(model.parameters())

    trainable_params = []
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        # MoE layers contain 'experts' or 'router' in their names
        # Also keep layer norms trainable for stability
        is_moe = 'experts' in name.lower() or 'router' in name.lower()
        is_norm = 'norm' in name.lower() or 'ln' in name.lower()

        if is_moe or is_norm:
            param.requires_grad = True
            trainable_params.append(param)
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"MoE-only mode: {trainable_count/1e6:.1f}M trainable, {frozen_count/1e6:.1f}M frozen")
    return trainable_params


def finetune(model, train_dataloader, eval_dataloader, device, epochs=1, lr=1e-5,
             eval_every=500, max_steps=None, use_8bit_adam=True, pad_token_id=None,
             moe_only=False, load_balance_weight=0.01, z_loss_weight=0.001,
             resume_checkpoint=None, save_dir=None, use_lr_schedule=False):
    """Fine-tune the model with language modeling objective."""
    model.train()

    # Get trainable parameters (optionally freeze non-MoE layers)
    trainable_params = freeze_non_moe_layers(model, moe_only=moe_only)

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(trainable_params, lr=lr, weight_decay=0.01)
            print("Using 8-bit Adam optimizer")
        except ImportError:
            print("bitsandbytes not available, falling back to standard AdamW")
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Learning rate scheduler (optional)
    scheduler = None
    if use_lr_schedule:
        total_steps = len(train_dataloader) * epochs
        if max_steps:
            total_steps = min(total_steps, max_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        print(f"Using cosine LR schedule over {total_steps} steps")
    else:
        print(f"Using constant LR: {lr}")

    global_step = 0
    start_epoch = 0
    best_ppl = float('inf')

    # Resume from checkpoint
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)

        # Load MoE weights FIRST (this was missing!)
        if 'moe_layers' in ckpt and ckpt['moe_layers'] is not None:
            moe_layers = getattr(model, 'moe_layers', None)
            if moe_layers is not None:
                moe_layers.load_state_dict(ckpt['moe_layers'])
                print(f"  Loaded MoE weights from checkpoint")

        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'global_step' in ckpt:
            global_step = ckpt['global_step']
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch']
        if 'best_ppl' in ckpt:
            best_ppl = ckpt['best_ppl']
        print(f"  Resumed at step {global_step}, epoch {start_epoch}, best_ppl {best_ppl:.2f}")

    # Get MoE layers for auxiliary loss computation
    moe_layers = getattr(model, 'moe_layers', None)
    use_aux_loss = (load_balance_weight > 0 or z_loss_weight > 0) and moe_layers is not None

    if use_aux_loss:
        print(f"Auxiliary losses: load_balance={load_balance_weight}, z_loss={z_loss_weight}")

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_aux_loss = 0.0

        for batch in pbar:
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            else:
                input_ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                attention_mask = None

            # Shift for causal LM
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1].contiguous()

            # Forward
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            # LM Loss (ignore padding tokens)
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id if pad_token_id is not None else -100)
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Auxiliary losses from MoE layers
            aux_loss = torch.tensor(0.0, device=device)
            if use_aux_loss:
                for layer_idx, moe_layer in moe_layers.items():
                    # Get router logits by running input through router
                    # We need to get the FFN input, which we don't have directly
                    # Instead, use a hook or get_router_logits method if available
                    if hasattr(moe_layer, 'get_router_logits'):
                        # Need to get the input to this layer - approximate with input_ids embedding
                        # This is a limitation - for now we'll compute aux loss on a sample
                        pass

                # Simpler approach: compute aux loss on router logits from last forward
                # Most MoE implementations cache this
                for layer_idx, moe_layer in moe_layers.items():
                    if hasattr(moe_layer, '_last_router_logits'):
                        router_logits = moe_layer._last_router_logits
                        if load_balance_weight > 0 and hasattr(moe_layer, '_last_expert_counts'):
                            expert_counts = moe_layer._last_expert_counts
                            aux_loss = aux_loss + load_balance_weight * compute_load_balancing_loss(
                                router_logits, expert_counts
                            )
                        if z_loss_weight > 0:
                            aux_loss = aux_loss + z_loss_weight * compute_z_loss(router_logits)

            loss = lm_loss + aux_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            global_step += 1

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
            postfix = {
                'loss': f'{loss.item():.4f}',
                'lm': f'{lm_loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            }
            if use_aux_loss and aux_loss.item() > 0:
                postfix['aux'] = f'{aux_loss.item():.4f}'
            pbar.set_postfix(postfix)

            # Periodic evaluation
            if eval_dataloader and global_step % eval_every == 0:
                ppl, _ = compute_perplexity(model, eval_dataloader, device,
                                            pad_token_id=pad_token_id, max_batches=50,
                                            accumulate_routing=True)
                print(f"\n  Step {global_step}: eval_ppl = {ppl:.2f}")

                # Print routing stats (accumulated across eval batches)
                if moe_layers is not None:
                    compute_routing_stats_accumulated(moe_layers, verbose=True)

                if ppl < best_ppl:
                    best_ppl = ppl
                    # Save best checkpoint
                    if save_dir:
                        best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pt")
                        torch.save({
                            'moe_layers': model.moe_layers.state_dict() if hasattr(model, 'moe_layers') else None,
                            'global_step': global_step,
                            'epoch': epoch,
                            'best_ppl': best_ppl,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        }, best_ckpt_path)
                model.train()

            # Periodic checkpoint saving for resume
            if save_dir and global_step % (eval_every * 2) == 0:
                ckpt_path = os.path.join(save_dir, "latest_checkpoint.pt")
                torch.save({
                    'moe_layers': model.moe_layers.state_dict() if hasattr(model, 'moe_layers') else None,
                    'global_step': global_step,
                    'epoch': epoch,
                    'best_ppl': best_ppl,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                }, ckpt_path)

            if max_steps and global_step >= max_steps:
                break

        avg_loss = epoch_loss / len(train_dataloader)
        avg_lm = epoch_lm_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}, avg_lm = {avg_lm:.4f}")

        if max_steps and global_step >= max_steps:
            break

    return best_ppl


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate sparse model")

    # Model
    parser.add_argument('--model', type=str, required=True,
                        help='Base HuggingFace model (same as used for distillation)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to distilled MoE checkpoint')
    parser.add_argument('--layer_config', type=str, default=None,
                        help='Path to layer_config.json with per-layer n_experts/d_ff_expert')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])

    # Data
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1')
    parser.add_argument('--max_samples', type=int, default=20000)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)

    # Training
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--no_8bit_adam', action='store_true',
                        help='Disable 8-bit Adam (uses more memory)')
    parser.add_argument('--moe_only', action='store_true',
                        help='Freeze attention/embeddings, only train MoE layers and norms')
    parser.add_argument('--lr_schedule', action='store_true',
                        help='Enable cosine LR schedule (disabled by default for constant LR)')

    # Auxiliary losses
    parser.add_argument('--load_balance_weight', type=float, default=0.01,
                        help='Weight for load balancing loss (0 to disable)')
    parser.add_argument('--z_loss_weight', type=float, default=0.001,
                        help='Weight for z-loss (0 to disable)')

    # Balanced routing
    parser.add_argument('--balanced-routing', action='store_true',
                        help='Use balanced routing with capacity constraints to prevent collapse')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (includes optimizer state)')
    parser.add_argument('--load_finetuned', type=str, default=None,
                        help='Path to finetuned MoE checkpoint to load weights from (no optimizer state)')

    # Evaluation only
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip fine-tuning, just evaluate')
    parser.add_argument('--compare_dense', action='store_true',
                        help='Also evaluate the original dense model for comparison')

    # Control trial
    parser.add_argument('--random_init', action='store_true',
                        help='Control trial: use random MoE weights instead of distilled (to measure distillation value)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for saving fine-tuned checkpoint')

    # Device
    parser.add_argument('--device', type=str, default='cuda')

    # Compilation
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile with max-autotune for faster training')
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                        help='torch.compile mode (default: max-autotune)')

    args = parser.parse_args()

    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("SPARSE MODEL FINE-TUNING & EVALUATION")
    print("=" * 60)

    # Load base model
    print(f"\nLoading base model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dense_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
    )

    # Convert to sparse
    print(f"\nConverting to sparse model using checkpoint: {args.checkpoint}")
    if args.layer_config:
        print(f"Using layer config: {args.layer_config}")
    if args.random_init:
        print("*** CONTROL TRIAL: Using random MoE weights (not distilled) ***")
    from Experiments.sparse_distillation.convert import convert_to_sparse

    sparse_model = convert_to_sparse(
        dense_model,
        args.checkpoint,
        device=args.device,
        layer_config_path=args.layer_config,
        random_init=args.random_init,
    )

    # Enable balanced routing if requested
    if args.balanced_routing:
        print("\nEnabling balanced routing with capacity constraints...")
        enabled_count = 0
        for layer_idx, moe_layer in sparse_model.moe_layers.items():
            # Check both router and _base_router (for MLP router case)
            for router_attr in ['router', '_base_router']:
                if hasattr(moe_layer, router_attr):
                    router = getattr(moe_layer, router_attr)
                    if hasattr(router, 'balanced_routing'):
                        router.balanced_routing = True
                        enabled_count += 1
                        break  # Only count once per layer
        print(f"  Enabled balanced routing on {enabled_count} MoE layers")
        if enabled_count == 0:
            print("  WARNING: No routers with balanced_routing found (expert_choice routing doesn't need it)")

    # Optionally load finetuned MoE weights
    if args.load_finetuned:
        print(f"\nLoading finetuned MoE weights from: {args.load_finetuned}")
        finetuned_ckpt = torch.load(args.load_finetuned, map_location=args.device)

        print(f"  Checkpoint keys: {list(finetuned_ckpt.keys())}")

        if 'moe_layers' in finetuned_ckpt:
            saved_state = finetuned_ckpt['moe_layers']
            print(f"  Saved state dict keys (first 5): {list(saved_state.keys())[:5]}")
            print(f"  Model state dict keys (first 5): {list(sparse_model.moe_layers.state_dict().keys())[:5]}")

            # Check a sample weight before loading
            sample_key = list(saved_state.keys())[0]
            print(f"  Before load - {sample_key}: mean={sparse_model.moe_layers.state_dict()[sample_key].mean().item():.6f}")

            # Load weights
            sparse_model.moe_layers.load_state_dict(saved_state)

            # Verify after loading
            print(f"  After load - {sample_key}: mean={sparse_model.moe_layers.state_dict()[sample_key].mean().item():.6f}")
            print(f"  Saved value - {sample_key}: mean={saved_state[sample_key].mean().item():.6f}")

            # Verify the base_model also has the updated weights
            # Find the corresponding module in base_model
            for name, module in sparse_model.base_model.named_modules():
                if hasattr(module, 'experts') and hasattr(module.experts, 'w1'):
                    print(f"  Base model MoE {name} w1[0] mean: {module.experts.w1[0].mean().item():.6f}")
                    break

            print(f"  Loaded MoE layer weights successfully")
            if 'best_ppl' in finetuned_ckpt:
                print(f"  Previous best perplexity: {finetuned_ckpt['best_ppl']:.2f}")
        else:
            print(f"  WARNING: No 'moe_layers' key found in checkpoint")
            print(f"  Available keys: {list(finetuned_ckpt.keys())}")

    # Compile model for faster training/inference
    if args.compile:
        print(f"\nCompiling model with torch.compile (mode={args.compile_mode})...")
        try:
            # Compile the base_model (the actual transformer)
            sparse_model.base_model = torch.compile(
                sparse_model.base_model,
                mode=args.compile_mode,
                fullgraph=False,  # Allow graph breaks for complex MoE routing
            )
            print(f"  Model compiled successfully")
        except Exception as e:
            print(f"  WARNING: torch.compile failed: {e}")
            print(f"  Continuing without compilation...")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    from datasets import load_dataset

    if args.dataset == 'wikitext':
        train_dataset = load_dataset(args.dataset, args.dataset_config, split='train')
        eval_dataset = load_dataset(args.dataset, args.dataset_config, split='validation')
        text_column = 'text'
    else:
        train_dataset = load_dataset(args.dataset, split='train')
        eval_dataset = load_dataset(args.dataset, split='validation') if 'validation' in load_dataset(args.dataset).keys() else None
        text_column = 'text' if 'text' in train_dataset.column_names else train_dataset.column_names[0]

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=args.seq_len,
            padding='max_length',
            return_tensors='pt',
        )

    # Process train
    train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_samples)))
    train_dataset = train_dataset.filter(lambda x: len(x[text_column].strip()) > 0)
    train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    train_tokenized.set_format('torch')

    train_loader = DataLoader(train_tokenized, batch_size=args.batch_size, shuffle=True)

    # Process eval
    eval_loader = None
    if eval_dataset:
        eval_dataset = eval_dataset.filter(lambda x: len(x[text_column].strip()) > 0)
        eval_tokenized = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)
        eval_tokenized.set_format('torch')
        eval_loader = DataLoader(eval_tokenized, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_tokenized)}, Eval samples: {len(eval_tokenized) if eval_loader else 0}")

    # Optionally compare with dense model first
    if args.compare_dense:
        print("\n" + "-" * 40)
        print("DENSE MODEL BASELINE")
        print("-" * 40)

        # Reload fresh dense model
        dense_model_fresh = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=args.device,
        )

        if eval_loader:
            dense_ppl, dense_loss = compute_perplexity(
                dense_model_fresh, eval_loader, args.device,
                pad_token_id=tokenizer.pad_token_id
            )
            print(f"Dense model perplexity: {dense_ppl:.2f}")
            print(f"Dense model loss: {dense_loss:.4f}")

        del dense_model_fresh
        torch.cuda.empty_cache()

    # Evaluate sparse model before fine-tuning
    print("\n" + "-" * 40)
    print("SPARSE MODEL (before fine-tuning)")
    print("-" * 40)

    if eval_loader:
        sparse_ppl_before, sparse_loss_before = compute_perplexity(
            sparse_model, eval_loader, args.device,
            pad_token_id=tokenizer.pad_token_id,
            accumulate_routing=True
        )
        print(f"Sparse model perplexity: {sparse_ppl_before:.2f}")
        print(f"Sparse model loss: {sparse_loss_before:.4f}")

        # Print routing stats after initial evaluation (accumulated across all eval batches)
        if hasattr(sparse_model, 'moe_layers'):
            compute_routing_stats_accumulated(sparse_model.moe_layers, verbose=True)

    if not args.eval_only:
        # Fine-tune
        print("\n" + "-" * 40)
        print("FINE-TUNING")
        if args.moe_only:
            print("(MoE-only mode: freezing attention/embeddings)")
        print("-" * 40)

        # Setup output dir for checkpointing
        save_dir = None
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            save_dir = args.output

        best_ppl = finetune(
            sparse_model,
            train_loader,
            eval_loader,
            args.device,
            epochs=args.epochs,
            lr=args.lr,
            eval_every=args.eval_every,
            max_steps=args.max_steps,
            use_8bit_adam=not args.no_8bit_adam,
            pad_token_id=tokenizer.pad_token_id,
            moe_only=args.moe_only,
            load_balance_weight=args.load_balance_weight,
            z_loss_weight=args.z_loss_weight,
            resume_checkpoint=args.resume,
            save_dir=save_dir,
            use_lr_schedule=args.lr_schedule,
        )

        # Final evaluation
        print("\n" + "-" * 40)
        print("SPARSE MODEL (after fine-tuning)")
        print("-" * 40)

        if eval_loader:
            sparse_ppl_after, sparse_loss_after = compute_perplexity(
                sparse_model, eval_loader, args.device,
                pad_token_id=tokenizer.pad_token_id,
                accumulate_routing=True
            )
            print(f"Sparse model perplexity: {sparse_ppl_after:.2f}")
            print(f"Sparse model loss: {sparse_loss_after:.4f}")
            print(f"Improvement: {sparse_ppl_before:.2f} -> {sparse_ppl_after:.2f} ({100*(sparse_ppl_before-sparse_ppl_after)/sparse_ppl_before:.1f}%)")

            # Print routing stats after fine-tuning (accumulated across all eval batches)
            if hasattr(sparse_model, 'moe_layers'):
                compute_routing_stats_accumulated(sparse_model.moe_layers, verbose=True)

        # Save checkpoint
        if args.output:
            os.makedirs(args.output, exist_ok=True)

            # Save MoE layers only (more portable)
            moe_checkpoint_path = os.path.join(args.output, "finetuned_moe_layers.pt")
            moe_state = {
                'moe_layers': sparse_model.moe_layers.state_dict(),
                'layer_indices': sparse_model.layer_indices,
                'base_model': args.model,
                'original_checkpoint': args.checkpoint,
                'final_perplexity': sparse_ppl_after if eval_loader else None,
                'training_args': {
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'max_steps': args.max_steps,
                    'moe_only': args.moe_only,
                },
            }
            torch.save(moe_state, moe_checkpoint_path)
            print(f"\nSaved MoE checkpoint to: {moe_checkpoint_path}")

            # Also save full model state dict (larger but complete)
            full_checkpoint_path = os.path.join(args.output, "finetuned_full_model.pt")
            torch.save({
                'model_state_dict': sparse_model.base_model.state_dict(),
                'base_model': args.model,
                'final_perplexity': sparse_ppl_after if eval_loader else None,
            }, full_checkpoint_path)
            print(f"Saved full model checkpoint to: {full_checkpoint_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    params = sparse_model.count_parameters()
    print(f"Total parameters: {params['total']/1e9:.2f}B")
    print(f"MoE parameters: {params['moe']/1e9:.2f}B ({params['moe_percentage']:.1f}%)")
    print(f"Active parameters (top-2): {sparse_model.get_active_parameters()/1e9:.2f}B")

    if eval_loader:
        if args.eval_only:
            print(f"\nPerplexity: {sparse_ppl_before:.2f}")
        else:
            print(f"\nPerplexity: {sparse_ppl_before:.2f} -> {sparse_ppl_after:.2f}")


if __name__ == "__main__":
    main()
