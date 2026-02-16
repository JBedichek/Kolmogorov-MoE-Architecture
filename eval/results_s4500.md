# Perplexity Evaluation Results

## Settings

- **Checkpoint**: `checkpoints_quick/checkpoint-4500.pt`
- **Config**: `configs/production_training.yaml`
- **Device**: `cuda:1`
- **Tokens per dataset**: 2,000,000
- **Sequence length**: 1024
- **Stride**: 512

## Results

| Dataset | Perplexity | Loss | Tokens | Time (s) |
|---------|------------|------|--------|----------|
| wikipedia | N/A | N/A | - | - | *Could not load data* |
| reddit | 842.32 | 6.7362 | 3,994,815 | 50.2 |
| cc_news | 555.61 | 6.3201 | 3,994,815 | 53.1 |
| arxiv | N/A | N/A | - | - | *Could not load data* |
| stackexchange | 567.34 | 6.3410 | 3,994,815 | 50.0 |
| books | 947.60 | 6.8539 | 3,994,815 | 55.8 |
| **Average** | **728.22** | - | - | - |
