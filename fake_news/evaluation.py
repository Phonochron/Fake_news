"""Compute model metrics and select a model from validation results."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(name, actual, predicted, probability=None) -> dict:
    metrics = {
        "Model": name,
        "Accuracy": float(accuracy_score(actual, predicted)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(actual, probability))
        if probability is not None
        else np.nan,
    }
    print(f"\n{name}\n{classification_report(actual, predicted, zero_division=0)}")
    return metrics


def select_model(validation_results: list[dict]) -> str:
    """Choose once using validation F1, without consulting the test results."""
    return max(validation_results, key=lambda row: row["F1"])["Model"]
