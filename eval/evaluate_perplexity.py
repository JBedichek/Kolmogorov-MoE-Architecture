#!/usr/bin/env python3
"""
Perplexity evaluation script for MoE models.

Measures perplexity on multiple datasets:
- Wikipedia
- Reddit (OpenWebText as proxy)
- CC News
- Arxiv
- Stack Exchange
- BookCorpus (BookCorpus2 as proxy)

Each dataset uses 3M tokens with 2048 context length.
"""

import torch
import argparse
import yaml
import os
import time
import math
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoTokenizer

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig


@dataclass
class EvalDataset:
    """Configuration for an evaluation dataset."""
    name: str
    hf_name: str
    hf_config: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    description: str = ""


# Dataset configurations
EVAL_DATASETS = [
    EvalDataset(
        name="wikipedia",
        hf_name="wikipedia",
        hf_config="20220301.en",
        split="train",
        text_column="text",
        description="English Wikipedia (March 2022)"
    ),
    EvalDataset(
        name="reddit",
        hf_name="openwebtext",
        hf_config=None,
        split="train",
        text_column="text",
        description="OpenWebText (Reddit-sourced web content)"
    ),
    EvalDataset(
        name="cc_news",
        hf_name="cc_news",
        hf_config=None,
        split="train",
        text_column="text",
        description="Common Crawl News"
    ),
    EvalDataset(
        name="arxiv",
        hf_name="togethercomputer/RedPajama-Data-1T-Sample",
        hf_config=None,
        split="train",
        text_column="text",
        description="RedPajama Arxiv subset"
    ),
    EvalDataset(
        name="stackexchange",
        hf_name="HuggingFaceFW/fineweb-edu",
        hf_config="sample-10BT",
        split="train",
        text_column="text",
        description="FineWeb-Edu (educational web content)"
    ),
    EvalDataset(
        name="books",
        hf_name="emozilla/pg19-test",
        hf_config=None,
        split="test",
        text_column="text",
        description="PG-19 Books (Project Gutenberg)"
    ),
]


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Load config
    config_dict = load_config(config_path)
    model_cfg = config_dict['model']

    # Create model config
    config = AdvancedMoEConfig(
        vocab_size=model_cfg.get('vocab_size', 128256),
        d_model=model_cfg.get('d_model', 2048),
        n_layers=model_cfg.get('n_layers', 32),
        n_heads=model_cfg.get('n_heads', 16),
        n_kv_heads=model_cfg.get('n_kv_heads', 4),
        head_dim=model_cfg.get('head_dim', 128),
        d_ff=model_cfg.get('d_ff', 5632),
        d_ff_expert=model_cfg.get('d_ff_expert', 2816),
        max_seq_len=model_cfg.get('max_seq_len', 2048),
        n_experts=model_cfg.get('n_experts', 16),
        moe_top_k=model_cfg.get('moe_top_k', 2),
        moe_capacity_factor=model_cfg.get('moe_capacity_factor', 1.25),
        moe_load_balance_loss_weight=model_cfg.get('moe_load_balance_loss_weight', 0.01),
        moe_router_z_loss_weight=model_cfg.get('moe_router_z_loss_weight', 0.001),
        moe_layers=tuple(model_cfg.get('moe_layers', [])),
        moe_implementation=model_cfg.get('moe_implementation', 'batched'),
        mod_enabled=model_cfg.get('mod_enabled', False),
        mod_capacity_factor=model_cfg.get('mod_capacity_factor', 0.75),
        mod_router_hidden_dim=model_cfg.get('mod_router_hidden_dim', 128),
        mod_load_balance_loss_weight=model_cfg.get('mod_load_balance_loss_weight', 0.001),
        mamba_enabled=model_cfg.get('mamba_enabled', False),
        mamba_layers=tuple(model_cfg.get('mamba_layers', [])),
        n_pred_tokens=model_cfg.get('n_pred_tokens', 1),
        aux_loss_weights=tuple(model_cfg.get('aux_loss_weights', [1.0])),
        use_flash_attention=model_cfg.get('use_flash_attention', True),
        dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )

    # Create model
    print("  Creating model architecture...")
    model = MoETransformer(config)

    # Load checkpoint
    print("  Loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', 'unknown')
        print(f"  Loaded from step {step}")
    else:
        model.load_state_dict(checkpoint)

    # Move to device and eval mode
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    # Count parameters
    params = model.count_parameters()
    print(f"  Model: {params['total_billions']:.3f}B params ({params['active_billions']:.3f}B active)")

    return model, config


def prepare_eval_data(
    dataset_config: EvalDataset,
    tokenizer,
    target_tokens: int = 3_000_000,
    seq_len: int = 2048,
    stride: int = 512,
) -> List[torch.Tensor]:
    """Prepare evaluation sequences from a dataset."""
    print(f"\n  Loading {dataset_config.name}...")

    try:
        if dataset_config.hf_config:
            dataset = load_dataset(
                dataset_config.hf_name,
                dataset_config.hf_config,
                split=dataset_config.split,
                streaming=True,
            )
        else:
            dataset = load_dataset(
                dataset_config.hf_name,
                split=dataset_config.split,
                streaming=True,
            )
    except Exception as e:
        print(f"  ERROR loading {dataset_config.name}: {e}")
        return []

    # Collect text until we have enough tokens
    all_tokens = []
    total_tokens = 0

    print(f"  Tokenizing (target: {target_tokens:,} tokens)...")

    for example in tqdm(dataset, desc=f"  {dataset_config.name}", leave=False):
        text = example.get(dataset_config.text_column, "")
        if not text or len(text.strip()) < 100:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        total_tokens += len(tokens)

        if total_tokens >= target_tokens:
            break

    if total_tokens < seq_len:
        print(f"  WARNING: Only got {total_tokens:,} tokens (need at least {seq_len})")
        return []

    print(f"  Collected {total_tokens:,} tokens")

    # Create overlapping sequences for perplexity calculation
    sequences = []
    all_tokens = all_tokens[:target_tokens]  # Truncate to target

    for i in range(0, len(all_tokens) - seq_len, stride):
        seq = torch.tensor(all_tokens[i:i + seq_len], dtype=torch.long)
        sequences.append(seq)

    print(f"  Created {len(sequences)} sequences (stride={stride})")

    return sequences


@torch.no_grad()
def compute_perplexity(
    model,
    sequences: List[torch.Tensor],
    batch_size: int = 1,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute perplexity on a list of sequences."""
    if not sequences:
        return {"perplexity": float('nan'), "loss": float('nan'), "num_tokens": 0}

    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(sequences), batch_size), desc="  Evaluating", leave=False):
        batch = sequences[i:i + batch_size]
        input_ids = torch.stack(batch).to(device)
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        loss = outputs['lm_loss'] if 'lm_loss' in outputs else outputs['loss']

        # Count tokens (excluding first token which has no prediction)
        num_tokens = input_ids.numel() - input_ids.shape[0]

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "num_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE model perplexity")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/debug_no_mod.yaml', help='Path to model config')
    parser.add_argument('--tokens', type=int, default=3_000_000, help='Tokens per dataset')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length (must match model max_seq_len)')
    parser.add_argument('--stride', type=int, default=512, help='Stride for overlapping sequences')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific datasets to evaluate (default: all)')
    parser.add_argument('--output', type=str, default=None, help='Output file for results (.md or .json)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on (e.g., cuda:0, cuda:1, cpu)')
    args = parser.parse_args()

    # Set device
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print("=" * 70)
    print("PERPLEXITY EVALUATION")
    print("=" * 70)

    print(f"\nSettings:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  Tokens per dataset: {args.tokens:,}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Stride: {args.stride}")
    print(f"  Batch size: {args.batch_size}")

    # Load tokenizer from config
    print("\nLoading tokenizer...")
    config_dict = load_config(args.config)
    data_cfg = config_dict.get('data', {})
    tokenizer_name = data_cfg.get('tokenizer_name', 'meta-llama/Llama-2-7b-hf')
    print(f"  Tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer):,}")

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, args.config, device=device)

    # Filter datasets if specified
    datasets_to_eval = EVAL_DATASETS
    if args.datasets:
        datasets_to_eval = [d for d in EVAL_DATASETS if d.name in args.datasets]
        print(f"\nEvaluating on: {[d.name for d in datasets_to_eval]}")

    # Evaluate each dataset
    results = {}

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    for dataset_config in datasets_to_eval:
        print(f"\n{'─' * 70}")
        print(f"Dataset: {dataset_config.name}")
        print(f"  Source: {dataset_config.hf_name}")
        print(f"  Description: {dataset_config.description}")

        # Prepare data
        sequences = prepare_eval_data(
            dataset_config,
            tokenizer,
            target_tokens=args.tokens,
            seq_len=args.seq_len,
            stride=args.stride,
        )

        if not sequences:
            print(f"  SKIPPED: Could not load data")
            results[dataset_config.name] = {
                "perplexity": None,
                "loss": None,
                "error": "Could not load data"
            }
            continue

        # Compute perplexity
        start_time = time.time()
        metrics = compute_perplexity(
            model,
            sequences,
            batch_size=args.batch_size,
            device=device,
        )
        eval_time = time.time() - start_time

        results[dataset_config.name] = {
            "perplexity": metrics["perplexity"],
            "loss": metrics["loss"],
            "num_tokens": metrics["num_tokens"],
            "eval_time_s": eval_time,
        }

        print(f"\n  Results:")
        print(f"    Perplexity: {metrics['perplexity']:.2f}")
        print(f"    Loss: {metrics['loss']:.4f}")
        print(f"    Tokens evaluated: {metrics['num_tokens']:,}")
        print(f"    Time: {eval_time:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Dataset':<15} {'Perplexity':>12} {'Loss':>10}")
    print("-" * 40)

    valid_ppls = []
    for name, metrics in results.items():
        ppl = metrics.get("perplexity")
        loss = metrics.get("loss")

        if ppl is not None and not math.isnan(ppl) and not math.isinf(ppl):
            print(f"{name:<15} {ppl:>12.2f} {loss:>10.4f}")
            valid_ppls.append(ppl)
        else:
            print(f"{name:<15} {'N/A':>12} {'N/A':>10}")

    if valid_ppls:
        avg_ppl = sum(valid_ppls) / len(valid_ppls)
        print("-" * 40)
        print(f"{'AVERAGE':<15} {avg_ppl:>12.2f}")

    # Save results
    if args.output:
        if args.output.endswith('.md'):
            # Save as markdown
            with open(args.output, 'w') as f:
                f.write("# Perplexity Evaluation Results\n\n")
                f.write("## Settings\n\n")
                f.write(f"- **Checkpoint**: `{args.checkpoint}`\n")
                f.write(f"- **Config**: `{args.config}`\n")
                f.write(f"- **Device**: `{device}`\n")
                f.write(f"- **Tokens per dataset**: {args.tokens:,}\n")
                f.write(f"- **Sequence length**: {args.seq_len}\n")
                f.write(f"- **Stride**: {args.stride}\n\n")
                f.write("## Results\n\n")
                f.write("| Dataset | Perplexity | Loss | Tokens | Time (s) |\n")
                f.write("|---------|------------|------|--------|----------|\n")
                for name, metrics in results.items():
                    ppl = metrics.get("perplexity")
                    loss = metrics.get("loss")
                    num_tokens = metrics.get("num_tokens", 0)
                    eval_time = metrics.get("eval_time_s", 0)
                    if ppl is not None and not math.isnan(ppl) and not math.isinf(ppl):
                        f.write(f"| {name} | {ppl:.2f} | {loss:.4f} | {num_tokens:,} | {eval_time:.1f} |\n")
                    else:
                        error = metrics.get("error", "N/A")
                        f.write(f"| {name} | N/A | N/A | - | - | *{error}* |\n")
                if valid_ppls:
                    f.write(f"| **Average** | **{avg_ppl:.2f}** | - | - | - |\n")
        else:
            # Save as JSON
            import json
            with open(args.output, 'w') as f:
                json.dump({
                    "checkpoint": args.checkpoint,
                    "config": args.config,
                    "device": device,
                    "settings": {
                        "tokens_per_dataset": args.tokens,
                        "seq_len": args.seq_len,
                        "stride": args.stride,
                    },
                    "results": results,
                }, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
