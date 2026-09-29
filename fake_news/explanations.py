"""Local feature contributions for supported linear text classifiers."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def linear_term_contributions(model, vectorizer, features, limit: int = 5):
    """Return the largest signed TF-IDF contributions to a linear model score."""
    if not isinstance(model, (LinearSVC, LogisticRegression)):
        return ()

    row = features.getrow(0)
    weights = model.coef_[0, row.indices]
    contributions = row.data * weights
    terms = vectorizer.get_feature_names_out()
    ranked = np.argsort(np.abs(contributions))[::-1]
    return tuple(
        (str(terms[row.indices[index]]), float(contributions[index]))
        for index in ranked[:limit]
        if contributions[index] != 0
    )
