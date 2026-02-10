"""
Tokenizer wrapper for training.

Provides a unified interface around HuggingFace tokenizers.
Supports GPT-2 BPE tokenizer and custom tokenizers.
"""

from typing import List, Union, Optional
from transformers import AutoTokenizer


class TokenizerWrapper:
    """
    Wrapper around HuggingFace tokenizers.

    Provides consistent interface for tokenization with:
    - Automatic padding/truncation
    - Special token handling
    - Batch encoding
    - Vocab management
    """

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        vocab_size: Optional[int] = None,
        add_special_tokens: bool = True,
    ):
        """
        Initialize tokenizer.

        Args:
            tokenizer_name: Name of HuggingFace tokenizer or path to custom tokenizer
            vocab_size: Optional vocab size (for validation)
            add_special_tokens: Whether to add special tokens during encoding
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.add_special_tokens = add_special_tokens

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Validate vocab size if provided
        if vocab_size is not None:
            actual_vocab_size = len(self.tokenizer)
            if actual_vocab_size != vocab_size:
                print(f"Warning: Tokenizer vocab size ({actual_vocab_size}) "
                      f"differs from config vocab size ({vocab_size})")

        print(f"Initialized tokenizer: {tokenizer_name}")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print(f"  PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"  EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        print(f"  BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self.tokenizer.bos_token_id

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ):
        """
        Encode text to token IDs.

        Args:
            text: Single text string or list of strings
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            return_tensors: Format to return ('pt' for PyTorch tensors)

        Returns:
            Encoded token IDs (format depends on return_tensors)
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=self.add_special_tokens,
            return_tensors=return_tensors,
        )

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Single sequence or batch of token ID sequences
            skip_special_tokens: Whether to remove special tokens

        Returns:
            Decoded text string(s)
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def batch_decode(
        self,
        token_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode batch of token ID sequences to text.

        Args:
            token_ids: Batch of token ID sequences
            skip_special_tokens: Whether to remove special tokens

        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )


if __name__ == "__main__":
    # Test tokenizer wrapper
    print("Testing TokenizerWrapper...")

    # Initialize with GPT-2 tokenizer
    tokenizer = TokenizerWrapper("gpt2")

    # Test encoding
    text = "Hello, world! This is a test of the tokenizer."
    print(f"\nOriginal text: {text}")

    encoded = tokenizer.encode(text, return_tensors="pt")
    print(f"Encoded shape: {encoded['input_ids'].shape}")
    print(f"Token IDs: {encoded['input_ids'][0].tolist()}")

    # Test decoding
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"Decoded text: {decoded}")

    # Test batch encoding
    texts = [
        "First example sentence.",
        "Second example with more tokens to test padding.",
        "Third.",
    ]
    print(f"\nBatch encoding {len(texts)} texts...")

    batch_encoded = tokenizer.encode(
        texts,
        max_length=20,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    print(f"Batch shape: {batch_encoded['input_ids'].shape}")
    print(f"Attention mask shape: {batch_encoded['attention_mask'].shape}")

    # Test batch decoding
    batch_decoded = tokenizer.batch_decode(batch_encoded['input_ids'])
    print("\nBatch decoded:")
    for i, text in enumerate(batch_decoded):
        print(f"  {i}: {text}")

    print("\n✓ Tokenizer tests passed!")
