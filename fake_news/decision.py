"""Compare final-decision rules using validation data only."""

import numpy as np
from sklearn.metrics import f1_score

from fake_news.config import CLASSICAL_MODEL_NAMES

PREDICTION_MODEL_NAMES = (*CLASSICAL_MODEL_NAMES.values(), "LSTM")


def five_model_majority(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Reproduce the legacy vote, including its correlated ensemble vote."""
    missing = set(PREDICTION_MODEL_NAMES) - predictions.keys()
    if missing:
        raise ValueError(
            f"Missing predictions for voting: {', '.join(sorted(missing))}"
        )
    votes = np.column_stack([predictions[name] for name in PREDICTION_MODEL_NAMES])
    return (votes.sum(axis=1) >= 3).astype(int)


def compare_decision_rules(
    validation_results: list[dict],
    validation_predictions: dict[str, np.ndarray],
    validation_labels,
    selected_model: str,
) -> dict:
    """Record why deployment uses the validation-selected model."""
    selected_f1 = next(
        row["F1"] for row in validation_results if row["Model"] == selected_model
    )
    majority_f1 = f1_score(
        validation_labels, five_model_majority(validation_predictions), zero_division=0
    )
    return {
        "strategy": "validation_selected_model",
        "selected_model": selected_model,
        "selected_validation_f1": float(selected_f1),
        "five_model_majority_validation_f1": float(majority_f1),
        "majority_has_correlated_vote": True,
        "note": (
            "Voting Ensemble already combines Naive Bayes, Logistic Regression, "
            "and Linear SVM. Counting it again with those models duplicates their "
            "influence. The deployed rule is the model selected on validation F1."
        ),
    }
