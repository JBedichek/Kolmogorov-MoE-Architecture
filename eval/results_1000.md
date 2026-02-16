# Perplexity Evaluation Results

## Settings

- **Checkpoint**: `checkpoints_quick/checkpoint-1000.pt`
- **Config**: `configs/debug_no_mod.yaml`
- **Device**: `cuda:1`
- **Tokens per dataset**: 2,000,000
- **Sequence length**: 1024
- **Stride**: 512

## Results

| Dataset | Perplexity | Loss | Tokens | Time (s) |
|---------|------------|------|--------|----------|
| wikipedia | N/A | N/A | - | - | *Could not load data* |
| reddit | 782.80 | 6.6629 | 3,994,815 | 48.8 |
| cc_news | 543.50 | 6.2980 | 3,994,815 | 49.2 |
| arxiv | N/A | N/A | - | - | *Could not load data* |
| stackexchange | 559.40 | 6.3269 | 3,994,815 | 48.9 |
| books | 917.03 | 6.8211 | 3,994,815 | 52.1 |
| **Average** | **700.68** | - | - | - |
