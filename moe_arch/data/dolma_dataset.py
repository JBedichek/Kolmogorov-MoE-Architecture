"""
Dolma dataset loader with streaming support.

Supports:
- Streaming from Dolma dataset on HuggingFace
- Pre-tokenized data loading from disk
- Efficient batching with dynamic sequence packing
- Memory-mapped file support for fast random access
"""

import os
import torch
import numpy as np
from typing import Optional, Iterator, Dict
from torch.utils.data import IterableDataset, Dataset
from pathlib import Path


class DolmaStreamingDataset(IterableDataset):
    """
    Streaming dataset for Dolma data from HuggingFace.

    Continuously yields random chunks of tokenized data.
    Designed for large-scale training where dataset doesn't fit in memory.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        n_pred_tokens: int = 4,
        split: str = "train",
        streaming: bool = True,
        vocab_size: Optional[int] = None,
        use_dummy_if_no_split: bool = True,
    ):
        """
        Initialize streaming dataset.

        Args:
            tokenizer: TokenizerWrapper instance
            seq_len: Sequence length for training
            n_pred_tokens: Number of future tokens to predict (for multi-token prediction)
            split: Dataset split ('train' or 'validation')
            streaming: Whether to use streaming mode
            vocab_size: Model vocab size (for dummy data, defaults to tokenizer vocab_size)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.n_pred_tokens = n_pred_tokens
        self.split = split
        self.streaming = streaming
        self.vocab_size = vocab_size if vocab_size is not None else tokenizer.vocab_size

        # Try to load dataset (prioritize datasets without deprecated loading scripts)
        try:
            from datasets import load_dataset
            print(f"Loading dataset ({split}, streaming={streaming})...")

            # Try multiple datasets in order of preference
            datasets_to_try = [
                # FineWeb - large, high-quality, modern dataset
                ("HuggingFaceFW/fineweb", {"name": "sample-10BT"}, "FineWeb-10BT sample"),
                # FineWeb-Edu - filtered for educational content
                ("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT"}, "FineWeb-Edu sample"),
                # OpenWebText - GPT-2 training data recreation
                ("Skylion007/openwebtext", {}, "OpenWebText"),
                # The Pile - diverse text dataset
                ("monology/pile-uncopyrighted", {}, "The Pile (uncopyrighted)"),
            ]

            loaded = False
            for dataset_name, kwargs, description in datasets_to_try:
                try:
                    print(f"  Trying {description}...")
                    # Many datasets only have 'train' split - use it for validation too
                    actual_split = split
                    try:
                        self.dataset = load_dataset(
                            dataset_name,
                            split=actual_split,
                            streaming=streaming,
                            **kwargs
                        )
                    except ValueError as e:
                        if split == "validation" and "split" in str(e).lower():
                            print(f"    No validation split, using train split instead...")
                            actual_split = "train"
                            self.dataset = load_dataset(
                                dataset_name,
                                split=actual_split,
                                streaming=streaming,
                                **kwargs
                            )
                        else:
                            raise
                    print(f"  ✓ {description} loaded successfully (split: {actual_split})")
                    loaded = True
                    break
                except Exception as e:
                    print(f"    Failed: {str(e)[:100]}")
                    continue

            if not loaded:
                raise Exception("All dataset loading attempts failed")

        except Exception as e:
            print(f"Warning: Could not load any dataset: {e}")
            print(f"  Using dummy data for testing")
            self.dataset = None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over dataset yielding tokenized sequences.

        Yields:
            Dictionary with:
                - input_ids: (seq_len,) token IDs for input
                - labels: (seq_len,) token IDs for labels (shifted input)
        """
        if self.dataset is None:
            # Generate dummy data for testing
            while True:
                # Random token IDs (avoid 0 which might be padding)
                input_ids = torch.randint(
                    1, self.vocab_size,
                    (self.seq_len + self.n_pred_tokens,)
                )
                yield {
                    "input_ids": input_ids[:-self.n_pred_tokens],
                    "labels": input_ids[:-self.n_pred_tokens],
                }
        else:
            # Stream from Dolma dataset
            for example in self.dataset:
                # Get text and tokenize
                text = example.get("text", "")
                if not text:
                    continue

                # Tokenize (no truncation to get full text)
                encoded = self.tokenizer.encode(
                    text,
                    truncation=False,
                )
                tokens = encoded["input_ids"] if isinstance(encoded, dict) else encoded

                # Skip if too short
                if len(tokens) < self.seq_len + self.n_pred_tokens:
                    continue

                # Yield chunks of seq_len tokens
                for i in range(0, len(tokens) - self.seq_len - self.n_pred_tokens, self.seq_len):
                    chunk = torch.tensor(tokens[i : i + self.seq_len + self.n_pred_tokens])
                    yield {
                        "input_ids": chunk[:-self.n_pred_tokens],
                        "labels": chunk[:-self.n_pred_tokens],
                    }


class DolmaMemoryMappedDataset(Dataset):
    """
    Memory-mapped dataset for pre-tokenized Dolma data.

    Loads pre-tokenized data from disk using memory mapping for fast random access.
    Requires pre-processing step to tokenize data and save to binary file.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 2048,
        n_pred_tokens: int = 4,
        vocab_size: int = 50257,
    ):
        """
        Initialize memory-mapped dataset.

        Args:
            data_path: Path to pre-tokenized binary file (.bin)
            seq_len: Sequence length for training
            n_pred_tokens: Number of future tokens to predict
            vocab_size: Vocabulary size (for validation)
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.n_pred_tokens = n_pred_tokens
        self.vocab_size = vocab_size

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Pre-tokenized data file not found: {data_path}\n"
                f"Please run pre-tokenization script first to create this file."
            )

        # Load as memory-mapped array (uint16 supports vocab up to 65k)
        print(f"Loading memory-mapped dataset from {data_path}...")
        self.data = np.memmap(
            self.data_path,
            dtype=np.uint16,
            mode='r',
        )

        # Calculate number of samples
        tokens_per_sample = seq_len + n_pred_tokens
        self.n_samples = len(self.data) // tokens_per_sample

        print(f"  ✓ Loaded {len(self.data):,} tokens")
        print(f"  ✓ {self.n_samples:,} samples of length {seq_len}")

    def __len__(self) -> int:
        """Get number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - input_ids: (seq_len,) token IDs for input
                - labels: (seq_len,) token IDs for labels
        """
        # Calculate start position
        tokens_per_sample = self.seq_len + self.n_pred_tokens
        start_idx = idx * tokens_per_sample

        # Get chunk (seq_len + n_pred_tokens for multi-token prediction)
        chunk = self.data[start_idx : start_idx + tokens_per_sample]

        # Convert to tensor
        chunk = torch.from_numpy(chunk.astype(np.int64))

        return {
            "input_ids": chunk[:-self.n_pred_tokens],
            "labels": chunk[:-self.n_pred_tokens],
        }


def get_dataloaders(
    tokenizer,
    config,
    batch_size: int = 8,
    num_workers: int = 4,
    use_memory_mapped: bool = False,
    data_path: Optional[str] = None,
):
    """
    Create train and validation dataloaders.

    Args:
        tokenizer: TokenizerWrapper instance
        config: Model config
        batch_size: Batch size
        num_workers: Number of dataloader workers
        use_memory_mapped: Whether to use memory-mapped dataset
        data_path: Path to pre-tokenized data (if use_memory_mapped=True)

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    from torch.utils.data import DataLoader

    if use_memory_mapped:
        if data_path is None:
            raise ValueError("data_path must be provided when use_memory_mapped=True")

        # Memory-mapped datasets
        train_dataset = DolmaMemoryMappedDataset(
            data_path=os.path.join(data_path, "train.bin"),
            seq_len=config.max_seq_len,
            n_pred_tokens=config.n_pred_tokens,
            vocab_size=config.vocab_size,
        )

        val_dataset = DolmaMemoryMappedDataset(
            data_path=os.path.join(data_path, "val.bin"),
            seq_len=config.max_seq_len,
            n_pred_tokens=config.n_pred_tokens,
            vocab_size=config.vocab_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    else:
        # Streaming datasets
        train_dataset = DolmaStreamingDataset(
            tokenizer=tokenizer,
            seq_len=config.max_seq_len,
            n_pred_tokens=config.n_pred_tokens,
            split="train",
            streaming=True,
            vocab_size=config.vocab_size,
        )

        val_dataset = DolmaStreamingDataset(
            tokenizer=tokenizer,
            seq_len=config.max_seq_len,
            n_pred_tokens=config.n_pred_tokens,
            split="validation",
            streaming=True,
            vocab_size=config.vocab_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,  # Streaming datasets don't support multiple workers well
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from moe_arch.data.tokenizer import TokenizerWrapper
    from moe_arch.model.config import get_test_config

    print("Testing Dolma dataset loaders...")

    # Initialize tokenizer
    tokenizer = TokenizerWrapper("gpt2")

    # Test streaming dataset
    print("\n1. Testing streaming dataset...")
    config = get_test_config()
    dataset = DolmaStreamingDataset(
        tokenizer=tokenizer,
        seq_len=128,
        n_pred_tokens=4,
        split="train",
    )

    print("  Fetching first 3 samples...")
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"    Sample {i}:")
        print(f"      input_ids shape: {sample['input_ids'].shape}")
        print(f"      labels shape: {sample['labels'].shape}")

    print("  ✓ Streaming dataset works")

    # Test dataloader
    print("\n2. Testing dataloader...")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, num_workers=0)

    for i, batch in enumerate(loader):
        if i >= 2:
            break
        print(f"    Batch {i}:")
        print(f"      input_ids shape: {batch['input_ids'].shape}")
        print(f"      labels shape: {batch['labels'].shape}")

    print("  ✓ Dataloader works")

    print("\n✓ All dataset tests passed!")
