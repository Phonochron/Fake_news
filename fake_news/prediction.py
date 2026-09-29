"""Loading saved artifacts and producing predictions."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from fake_news.config import (
    CLASSICAL_MODEL_NAMES,
    MAX_INPUT_CHARACTERS,
    MODEL_DIR,
    MODEL_FILES,
    MODEL_SELECTION_FILE,
)
from fake_news.explanations import linear_term_contributions
from fake_news.preprocessing import clean_text
from fake_news.sequences import encode_sequences


@dataclass(frozen=True)
class ModelBundle:
    vectorizer: Any
    naive_bayes: Any
    logistic_regression: Any
    svm: Any
    ensemble: Any
    tokenizer: Any
    lstm: Any
    selected_model: str | None = None


@dataclass(frozen=True)
class PredictionResult:
    predictions: dict[str, int]
    hoax_probability: dict[str, float]
    final_label: int
    selected_model: str | None = None
    term_contributions: tuple[tuple[str, float], ...] = ()


def load_models(model_dir: Path = MODEL_DIR) -> ModelBundle:
    missing = [
        name for name in MODEL_FILES.values() if not (model_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            artifacts = {
                key: joblib.load(model_dir / filename)
                for key, filename in MODEL_FILES.items()
                if key != "lstm"
            }
    except InconsistentVersionWarning as exc:
        raise RuntimeError(
            f"Saved models require scikit-learn {exc.original_sklearn_version}; "
            "install that version or retrain with the current environment."
        ) from exc
    # Keras is imported lazily so the Streamlit launcher can initialize
    # TensorFlow first on Windows.
    from keras.models import load_model

    artifacts["lstm"] = load_model(model_dir / MODEL_FILES["lstm"], compile=False)
    selection_file = model_dir / MODEL_SELECTION_FILE
    if selection_file.is_file():
        try:
            selection = json.loads(selection_file.read_text(encoding="utf-8"))
            selected_model = selection["model"]
            if selected_model not in (*CLASSICAL_MODEL_NAMES.values(), "LSTM"):
                raise ValueError(f"Unknown model: {selected_model}")
            if selection.get("strategy", "validation_selected_model") != (
                "validation_selected_model"
            ):
                raise ValueError("Unknown decision strategy")
            artifacts["selected_model"] = selected_model
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid model selection: {selection_file}") from exc
    return ModelBundle(**artifacts)


def predict_news(text: str, models: ModelBundle) -> PredictionResult:
    if not isinstance(text, str):
        raise ValueError("News text must be a string.")
    if len(text) > MAX_INPUT_CHARACTERS:
        raise ValueError(f"News text exceeds {MAX_INPUT_CHARACTERS:,} characters.")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("No words remain after preprocessing.")

    features = models.vectorizer.transform([cleaned])
    classifiers = {
        name: getattr(models, key) for key, name in CLASSICAL_MODEL_NAMES.items()
    }
    predictions = {
        name: int(model.predict(features)[0]) for name, model in classifiers.items()
    }
    hoax_probability = {
        name: float(classifiers[name].predict_proba(features)[0, 1])
        for name in ("Naive Bayes", "Logistic Regression")
    }
    encoded = encode_sequences(models.tokenizer, [cleaned])
    probability = float(models.lstm.predict(encoded, verbose=0)[0][0])
    predictions["LSTM"] = int(probability > 0.5)
    hoax_probability["LSTM"] = probability
    if models.selected_model is not None and models.selected_model not in predictions:
        raise ValueError(f"Unknown selected model: {models.selected_model}")
    term_contributions = linear_term_contributions(
        classifiers.get(models.selected_model), models.vectorizer, features
    )
    return PredictionResult(
        predictions=predictions,
        hoax_probability=hoax_probability,
        final_label=(
            predictions[models.selected_model]
            if models.selected_model is not None
            else int(sum(predictions.values()) >= 3)
        ),
        selected_model=models.selected_model,
        term_contributions=term_contributions,
    )
