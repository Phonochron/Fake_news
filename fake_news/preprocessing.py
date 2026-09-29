"""Identical text preparation for training and inference."""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def _english_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            downloaded = nltk.download("stopwords", quiet=True)
        except OSError as exc:
            raise RuntimeError("Could not download NLTK stopwords.") from exc
        if not downloaded:
            raise RuntimeError("Could not download NLTK stopwords.")
        try:
            return set(stopwords.words("english"))
        except LookupError as exc:
            raise RuntimeError(
                "NLTK stopwords are unavailable after download."
            ) from exc


@lru_cache(maxsize=1)
def _resources() -> tuple[set[str], PorterStemmer]:
    return _english_stopwords(), PorterStemmer()


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
