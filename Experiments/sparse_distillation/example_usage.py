#!/usr/bin/env python3
"""
Example: Sparse Distillation of a HuggingFace Model

This example demonstrates how to convert a pretrained dense model (e.g., Llama-2-7B)
into a sparse MoE model using internal distillation.

Usage:
    python Experiments/sparse_distillation/example_usage.py \
        --model meta-llama/Llama-2-7b-hf \
        --dataset wikitext \
        --n_experts 16 \
        --epochs 3 \
        --output ./sparse_llama

Requirements:
    pip install transformers datasets
"""

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(description="Sparse Distillation Example")

    # Model
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='HuggingFace model name or path')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])

    # Data
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='Dataset name (wikitext, c4, etc.)')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum number of samples to use')
    parser.add_argument('--seq_len', type=int, default=2048,
                        help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1)

    # MoE config
    parser.add_argument('--n_experts', type=int, default=64,
                        help='Number of experts per layer')
    parser.add_argument('--moe_top_k', type=int, default=3,
                        help='Experts per token (token-choice) or ignored (expert-choice)')
    parser.add_argument('--capacity_factor', type=float, default=1.25,
                        help='Expert capacity factor (expert-choice: tokens_per_expert = seq * cap / n_experts)')
    parser.add_argument('--d_ff_expert', type=str, default=0.075,
                        help='Expert FFN dim: None=match dense, 0.5=half dense, 2048=absolute')
    # token_choice recommended for distillation (guarantees every token is processed)
    # expert_choice can leave tokens unprocessed, causing poor distillation
    parser.add_argument('--routing', type=str, default='token_choice',
                        choices=['expert_choice', 'token_choice'])
    parser.add_argument('--layer_config', type=str, default=None,
                        help='Path to per-layer config JSON (from generate_config.py)')

    # Training
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--sequential', action='store_true',
                        help='Train one layer at a time (much lower memory)')

    # Data-free distillation (recommended - faster and better coverage)
    parser.add_argument('--no_data_free', action='store_true',
                        help='Disable data-free mode (use real data instead)')
    parser.add_argument('--data_free_steps', type=int, default=20000,
                        help='Training steps per layer in data-free mode')
    parser.add_argument('--data_free_batch_size', type=int, default=4,
                        help='Batch size for data-free distillation')
    parser.add_argument('--data_free_seq_len', type=int, default=512,
                        help='Sequence length for data-free distillation')

    # Performance
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile')

    # Auxiliary losses
    parser.add_argument('--no_load_balance', action='store_true',
                        help='Disable load balancing loss')
    parser.add_argument('--load_balance_weight', type=float, default=0.01,
                        help='Weight for load balancing loss')
    parser.add_argument('--no_z_loss', action='store_true',
                        help='Disable z-loss')
    parser.add_argument('--z_loss_weight', type=float, default=0.001,
                        help='Weight for z-loss')

    # Supervised routing (cluster-based)
    parser.add_argument('--no_supervised_routing', action='store_true',
                        help='Disable supervised router training (use joint training instead)')
    parser.add_argument('--cluster_samples', type=int, default=1000000,
                        help='Number of samples to collect for clustering')
    parser.add_argument('--router_train_steps', type=int, default=10000,
                        help='Steps to train router per layer (supervised routing)')
    parser.add_argument('--analyze_clusters', action='store_true',
                        help='Run cluster quality analysis to find optimal n_experts, then exit')

    # Empirical routing (train router based on actual expert performance)
    parser.add_argument('--empirical_routing', action='store_true',
                        help='Use empirical routing: run all experts, train router on lowest-loss experts')
    parser.add_argument('--empirical_eval_interval', type=int, default=10,
                        help='Recompute empirical routing targets every N steps')
    parser.add_argument('--empirical_temperature', type=float, default=0.1,
                        help='Temperature for softmax over expert losses')
    parser.add_argument('--empirical_entropy_weight', type=float, default=0.01,
                        help='Entropy bonus weight to prevent routing collapse')

    # Curriculum balanced routing (prevents collapse by starting uniform, gradually specializing)
    parser.add_argument('--no_curriculum', action='store_true',
                        help='Disable curriculum balanced routing (enabled by default with empirical routing)')
    parser.add_argument('--curriculum_warmup_steps', type=int, default=5000,
                        help='Steps to transition from uniform to empirical routing targets')
    parser.add_argument('--curriculum_initial_balance', type=float, default=0.9,
                        help='Initial balance weight (0=pure empirical, 1=pure uniform)')
    parser.add_argument('--curriculum_final_balance', type=float, default=0.3,
                        help='Final balance weight after warmup (keeps some balance always)')

    # Gradient-based routing (route to experts with highest learning potential)
    parser.add_argument('--no_gradient_routing', action='store_true',
                        help='Use loss-based routing instead of gradient-based (default: gradient-based)')
    parser.add_argument('--gradient_routing_exact', action='store_true',
                        help='Use exact per-sample gradients (slow) instead of fast approximation')

    # Router architecture
    parser.add_argument('--router_mlp', action='store_true',
                        help='Use MLP router with non-linearities instead of linear')
    parser.add_argument('--router_hidden_dim', type=int, default=256,
                        help='Hidden dimension for MLP router')
    parser.add_argument('--router_n_layers', type=int, default=2,
                        help='Number of layers in MLP router')

    # Dynamic expert reallocation
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic expert reallocation based on per-expert loss')
    parser.add_argument('--reallocate_every', type=int, default=500,
                        help='Reallocate expert dimensions every N steps')
    parser.add_argument('--growth_factor', type=float, default=1.25,
                        help='Grow high-loss experts by this factor')
    parser.add_argument('--shrink_factor', type=float, default=0.8,
                        help='Shrink low-loss experts by this factor')
    parser.add_argument('--min_d_ff', type=int, default=256,
                        help='Minimum expert FFN dimension')
    parser.add_argument('--max_d_ff', type=int, default=8192,
                        help='Maximum expert FFN dimension')

    # Output
    parser.add_argument('--output', type=str, default='./sparse_distillation_output')

    # Device
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Set dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("SPARSE DISTILLATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"n_experts: {args.n_experts}")
    print(f"Routing: {args.routing}")
    print(f"Device: {args.device}")
    if not args.no_data_free:
        print(f"Mode: Data-free (random inputs)")
        print(f"  Steps per layer: {args.data_free_steps}")
        print(f"  Batch size: {args.data_free_batch_size}")
        if not args.no_supervised_routing:
            print(f"  Supervised routing: ON (cluster-based)")
            print(f"  Cluster samples: {args.cluster_samples}")
    elif args.sequential:
        print(f"Mode: Sequential (one layer at a time)")
    elif args.dynamic:
        print(f"Mode: Dynamic expert reallocation")
        print(f"  Reallocate every: {args.reallocate_every} steps")
        print(f"  Growth/shrink: {args.growth_factor:.2f}x / {args.shrink_factor:.2f}x")
    else:
        print(f"Mode: Parallel (all layers)")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    from datasets import load_dataset

    if args.dataset == 'wikitext':
        dataset = load_dataset(args.dataset, args.dataset_config, split='train')
        text_column = 'text'
    elif args.dataset == 'c4':
        dataset = load_dataset('c4', 'en', split='train', streaming=True)
        dataset = dataset.take(args.max_samples)
        text_column = 'text'
    elif args.dataset == 'fineweb-edu' or args.dataset == 'HuggingFaceFW/fineweb-edu':
        dataset = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
        dataset = dataset.take(args.max_samples)
        text_column = 'text'
    else:
        dataset = load_dataset(args.dataset, split='train')
        # Try to find text column
        text_column = 'text' if 'text' in dataset.column_names else dataset.column_names[0]

    # Tokenize and create dataloader
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=args.seq_len,
            padding='max_length',
            return_tensors='pt',
        )

    # Process dataset
    from datasets import IterableDataset
    is_streaming = isinstance(dataset, IterableDataset)

    if not is_streaming:
        # Non-streaming dataset
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
        )
        tokenized.set_format('torch')
    else:
        # Streaming - collect samples
        samples = []
        for sample in tqdm(dataset, total=args.max_samples, desc="Tokenizing"):
            tokens = tokenizer(
                sample[text_column],
                truncation=True,
                max_length=args.seq_len,
                padding='max_length',
                return_tensors='pt',
            )
            samples.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
            })
            if len(samples) >= args.max_samples:
                break

        from torch.utils.data import Dataset

        class SimpleDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]

        tokenized = SimpleDataset(samples)

    dataloader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(f"Dataset ready: {len(tokenized)} samples")

    # Create distillation trainer
    print("\nInitializing sparse distillation trainer...")
    from Experiments.sparse_distillation import SparseDistillationTrainer
    from Experiments.sparse_distillation.distill import DistillationConfig

    # Parse d_ff_expert: None, float (0-1), or int
    d_ff_expert = None
    if args.d_ff_expert is not None:
        try:
            val = float(args.d_ff_expert)
            if val <= 1.0:
                d_ff_expert = val  # Fraction of dense d_ff
            else:
                d_ff_expert = int(val)  # Absolute dimension
        except ValueError:
            pass  # Keep None

    config = DistillationConfig(
        n_experts=args.n_experts,
        moe_top_k=args.moe_top_k,
        moe_capacity_factor=args.capacity_factor,
        moe_routing=args.routing,
        d_ff_expert=d_ff_expert,
        layer_config_path=args.layer_config,
        learning_rate=args.lr,
        use_cosine_loss=True,
        init_from_dense=True,
        sequential_layers=args.sequential,
        # Data-free distillation (default: enabled)
        data_free=not args.no_data_free,
        data_free_steps=args.data_free_steps,
        data_free_batch_size=args.data_free_batch_size,
        data_free_seq_len=args.data_free_seq_len,
        # Performance
        use_compile=not args.no_compile,
        # Auxiliary losses
        use_load_balancing=not args.no_load_balance,
        load_balancing_weight=args.load_balance_weight,
        use_z_loss=not args.no_z_loss,
        z_loss_weight=args.z_loss_weight,
        # Supervised routing
        supervised_routing=not args.no_supervised_routing and not args.empirical_routing,
        cluster_samples=args.cluster_samples,
        router_train_steps=args.router_train_steps,
        # Empirical routing
        empirical_routing=args.empirical_routing,
        empirical_eval_interval=args.empirical_eval_interval,
        empirical_temperature=args.empirical_temperature,
        empirical_entropy_weight=args.empirical_entropy_weight,
        # Curriculum balanced routing
        curriculum_routing=not args.no_curriculum,
        curriculum_warmup_steps=args.curriculum_warmup_steps,
        curriculum_initial_balance=args.curriculum_initial_balance,
        curriculum_final_balance=args.curriculum_final_balance,
        # Gradient-based routing
        gradient_routing=not args.no_gradient_routing,
        gradient_routing_fast=not args.gradient_routing_exact,
        # Router architecture
        router_mlp=args.router_mlp,
        router_hidden_dim=args.router_hidden_dim,
        router_n_layers=args.router_n_layers,
        # Dynamic expert reallocation
        dynamic_experts=args.dynamic,
        reallocate_every_n_steps=args.reallocate_every,
        growth_factor=args.growth_factor,
        shrink_factor=args.shrink_factor,
        min_d_ff=args.min_d_ff,
        max_d_ff=args.max_d_ff,
    )

    trainer = SparseDistillationTrainer(
        dense_model=model,
        config=config,
        device=args.device,
    )

    # Optional: Run cluster analysis to find optimal n_experts
    if args.analyze_clusters:
        import json

        print("\nRunning cluster quality analysis on all layers...")
        k_values = list(range(4, 65, 2))  # 4, 6, 8, ..., 64

        all_results = {}
        layer_recommendations = {}

        for layer_idx in trainer.extractor.layer_indices:
            print(f"\n{'#'*60}")
            print(f"# LAYER {layer_idx}")
            print(f"{'#'*60}")

            ffn_inputs, ffn_outputs = trainer._collect_ffn_activations_from_data(
                dataloader, layer_idx,
                max_samples=min(args.cluster_samples, 100000),
            )

            # Run analysis on outputs
            print(f"\nAnalyzing FFN outputs (layer {layer_idx}):")
            results = trainer.analyze_cluster_quality(ffn_outputs, k_values)
            all_results[layer_idx] = results

            # Find best k by balance (primary metric for MoE expert count)
            best_k = max(results.keys(), key=lambda k: results[k]['balance'])

            # Convert results keys to int for JSON serialization
            results_serializable = {int(k): v for k, v in results.items()}

            layer_recommendations[layer_idx] = {
                'n_experts': best_k,
                'balance': results[best_k]['balance'],
                'silhouette': results[best_k]['silhouette'],
                'all_results': results_serializable,
            }

        # Summary across all layers
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL LAYERS (by best balance)")
        print(f"{'='*60}")
        print(f"\n{'Layer':>6} | {'Best k':>8} | {'Balance':>10} | {'Silhouette':>10}")
        print("-" * 50)

        for layer_idx in sorted(layer_recommendations.keys()):
            rec = layer_recommendations[layer_idx]
            print(f"{layer_idx:>6} | {rec['n_experts']:>8} | {rec['balance']:>10.4f} | {rec['silhouette']:>10.4f}")

        # Save to JSON (include all_results for generate_config.py to reanalyze if needed)
        output_data = {
            'd_model': trainer.d_model,
            'd_ff': trainer.d_ff,
            'layers': {
                str(layer_idx): {
                    'n_experts': rec['n_experts'],
                    'balance': rec['balance'],
                    'silhouette': rec['silhouette'],
                    'all_results': rec['all_results'],
                }
                for layer_idx, rec in layer_recommendations.items()
            }
        }

        analysis_path = os.path.join(args.output, 'cluster_analysis.json')
        os.makedirs(args.output, exist_ok=True)
        with open(analysis_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved analysis to: {analysis_path}")

        print("\nCluster analysis complete. Exiting.")
        print(f"Run: python Experiments/sparse_distillation/generate_config.py {analysis_path}")
        sys.exit(0)

    # Run distillation
    print("\nStarting distillation...")
    if config.data_free:
        # Data-free mode: no dataloader needed
        trainer.distill()
    else:
        trainer.distill(
            dataloader=dataloader,
            epochs=args.epochs,
            max_steps=args.max_steps,
        )

    # Evaluate on real data (optional, but useful to verify)
    print("\nEvaluating distillation quality on real data...")
    eval_dataloader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    trainer.evaluate(eval_dataloader, max_batches=100)

    # Save
    checkpoint_path = os.path.join(args.output, 'distilled_moe_layers.pt')
    trainer.save(checkpoint_path)

    # Convert to sparse model
    print("\nConverting to sparse model...")
    from Experiments.sparse_distillation.convert import convert_to_sparse

    sparse_model = convert_to_sparse(
        model,
        checkpoint_path,
        device=args.device,
    )

    # Save full sparse model
    sparse_path = os.path.join(args.output, 'sparse_model.pt')
    torch.save({
        'model_state_dict': sparse_model.state_dict(),
        'config': config,
    }, sparse_path)
    print(f"Saved sparse model to {sparse_path}")

    # Quick generation test
    print("\nTesting generation...")
    test_prompt = "The quick brown fox"
    inputs = tokenizer(test_prompt, return_tensors='pt').to(args.device)

    sparse_model.eval()
    with torch.no_grad():
        outputs = sparse_model.base_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated}")

    print("\n" + "=" * 60)
    print("DISTILLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
