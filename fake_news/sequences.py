"""Shared token sequence encoding for LSTM training and inference."""

import numpy as np

from fake_news.config import MAX_LEN


def encode_sequences(tokenizer, texts, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode text with the same post padding and truncation in both paths."""
    sequences = tokenizer.texts_to_sequences(texts)
    encoded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for row, sequence in enumerate(sequences):
        length = min(len(sequence), max_len)
        encoded[row, :length] = sequence[:length]
    return encoded
