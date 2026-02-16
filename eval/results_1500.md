# Perplexity Evaluation Results

## Settings

- **Checkpoint**: `checkpoints_quick/checkpoint-1500.pt`
- **Config**: `configs/debug_no_mod.yaml`
- **Device**: `cuda:1`
- **Tokens per dataset**: 2,000,000
- **Sequence length**: 1024
- **Stride**: 512

## Results

| Dataset | Perplexity | Loss | Tokens | Time (s) |
|---------|------------|------|--------|----------|
| wikipedia | N/A | N/A | - | - | *Could not load data* |
| reddit | 620.05 | 6.4298 | 3,994,815 | 48.7 |
| cc_news | 406.09 | 6.0066 | 3,994,815 | 50.9 |
| arxiv | N/A | N/A | - | - | *Could not load data* |
| stackexchange | 426.82 | 6.0564 | 3,994,815 | 47.3 |
| books | 632.50 | 6.4497 | 3,994,815 | 53.1 |
| **Average** | **521.37** | - | - | - |
