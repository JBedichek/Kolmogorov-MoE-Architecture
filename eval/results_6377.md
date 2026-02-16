# Perplexity Evaluation Results

## Settings

- **Checkpoint**: `checkpoints_quick/checkpoint-6377-interrupted.pt`
- **Config**: `configs/production_training.yaml`
- **Device**: `cuda:1`
- **Tokens per dataset**: 2,000,000
- **Sequence length**: 1024
- **Stride**: 512

## Results

| Dataset | Perplexity | Loss | Tokens | Time (s) |
|---------|------------|------|--------|----------|
| wikipedia | N/A | N/A | - | - | *Could not load data* |
| reddit | 270.40 | 5.5999 | 3,994,815 | 32.3 |
| cc_news | 202.87 | 5.3126 | 3,994,815 | 34.4 |
| arxiv | N/A | N/A | - | - | *Could not load data* |
| stackexchange | 257.02 | 5.5491 | 3,994,815 | 32.7 |
| books | 325.01 | 5.7838 | 3,994,815 | 35.7 |
| **Average** | **263.82** | - | - | - |
