"""Project paths and shared model settings."""

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_SELECTION_FILE = "selected_model.json"
TRAINING_MANIFEST_FILE = "training_manifest.json"

MAX_WORDS = 10_000
MAX_LEN = 100
MAX_INPUT_CHARACTERS = 50_000
MIN_RECOMMENDED_WORDS = 30
RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = RANDOM_STATE
    cv_folds: int = 5
    lstm_epochs: int = 10
    tfidf_max_features: int = 20_000
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    nb_alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0)
    logistic_max_iter: int = 2000
    svm_c: float = 1.0
    lstm_batch_size: int = 64
    lstm_embedding_dim: int = 128
    lstm_units: int = 64
    lstm_dense_units: int = 32
    lstm_dropout: float = 0.3

    def validate(self) -> None:
        if self.cv_folds < 2 or self.lstm_epochs < 1:
            raise ValueError("cv_folds must be >= 2 and lstm_epochs must be >= 1")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        if self.tfidf_max_features < 1 or self.tfidf_min_df < 1:
            raise ValueError("TF-IDF limits must be positive")
        if not 0 < self.tfidf_max_df <= 1 or self.svm_c <= 0:
            raise ValueError("tfidf_max_df must be in (0, 1] and svm_c positive")
        if not self.nb_alphas or any(alpha <= 0 for alpha in self.nb_alphas):
            raise ValueError("nb_alphas must contain positive values")
        if self.logistic_max_iter < 1 or self.lstm_batch_size < 1:
            raise ValueError("iteration and batch limits must be positive")
        if min(self.lstm_embedding_dim, self.lstm_units, self.lstm_dense_units) < 1:
            raise ValueError("LSTM layer sizes must be positive")
        if not 0 <= self.lstm_dropout < 1:
            raise ValueError("lstm_dropout must be in [0, 1)")


CLASSICAL_MODEL_NAMES = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "svm": "Linear SVM",
    "ensemble": "Voting Ensemble",
}

MODEL_FILES = {
    "vectorizer": "tfidf_vectorizer.pkl",
    "naive_bayes": "naive_bayes.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "svm": "svm.pkl",
    "ensemble": "ensemble.pkl",
    "tokenizer": "tokenizer.pkl",
    "lstm": "lstm_model.h5",
}
