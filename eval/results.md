# Perplexity Evaluation Results

## Settings

- **Checkpoint**: `checkpoints_quick/checkpoint-4000.pt`
- **Config**: `configs/production_training.yaml`
- **Device**: `cuda:1`
- **Tokens per dataset**: 2,000,000
- **Sequence length**: 1024
- **Stride**: 512

## Results

| Dataset | Perplexity | Loss | Tokens | Time (s) |
|---------|------------|------|--------|----------|
| wikipedia | N/A | N/A | - | - | *Could not load data* |
| reddit | 901.47 | 6.8040 | 3,994,815 | 52.5 |
| cc_news | 598.15 | 6.3938 | 3,994,815 | 53.7 |
| arxiv | N/A | N/A | - | - | *Could not load data* |
| stackexchange | 615.65 | 6.4227 | 3,994,815 | 51.5 |
| books | 933.63 | 6.8391 | 3,994,815 | 57.7 |
| **Average** | **762.22** | - | - | - |
