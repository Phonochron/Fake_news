"""Identical text preparation for training and inference."""

import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


@lru_cache(maxsize=1)
def _resources() -> tuple[set[str], PorterStemmer]:
    try:
        words = set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK stopwords are missing. Run: python -m nltk.downloader stopwords"
        ) from exc
    return words, PorterStemmer()


def clean_text(text: str) -> str:
    """Normalize article text consistently for training and inference."""
    stop_words, stemmer = _resources()
    normalized = str(text).lower()
    # The source marker is almost exclusive to the real class in this dataset.
    normalized = re.sub(r"\breuters\b", "", normalized)
    normalized = re.sub(r"http\S+", "", normalized)
    normalized = re.sub(r"\d+", "", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return " ".join(
        stemmer.stem(word) for word in normalized.split() if word not in stop_words
    )
