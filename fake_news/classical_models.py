"""Train and evaluate TF-IDF based classifiers."""

from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from fake_news.config import CLASSICAL_MODEL_NAMES, TrainingConfig
from fake_news.evaluation import evaluate_model


def make_vectorizer(config: TrainingConfig = TrainingConfig()) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.tfidf_max_features,
        ngram_range=(1, 2),
        min_df=config.tfidf_min_df,
        max_df=config.tfidf_max_df,
        sublinear_tf=True,
    )


def make_pipeline(estimator, config: TrainingConfig = TrainingConfig()) -> Pipeline:
    """Fit TF-IDF inside every cross-validation fold."""
    return Pipeline([("tfidf", make_vectorizer(config)), ("classifier", estimator)])


def train_classical(train_x, train_y, config: TrainingConfig):
    cv = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.seed
    )
    search = GridSearchCV(
        make_pipeline(MultinomialNB(), config),
        {"classifier__alpha": list(config.nb_alphas)},
        cv=cv,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )
    search.fit(train_x, train_y)
    vectorizer = search.best_estimator_.named_steps["tfidf"]
    features = vectorizer.transform(train_x)
    models = {
        "naive_bayes": search.best_estimator_.named_steps["classifier"],
        "logistic_regression": LogisticRegression(
            max_iter=config.logistic_max_iter, random_state=config.seed
        ),
        "svm": LinearSVC(C=config.svm_c, random_state=config.seed),
    }
    for key in ("logistic_regression", "svm"):
        models[key].fit(features, train_y)

    models["ensemble"] = VotingClassifier(
        estimators=[
            ("nb", MultinomialNB(alpha=search.best_params_["classifier__alpha"])),
            (
                "lr",
                LogisticRegression(
                    max_iter=config.logistic_max_iter, random_state=config.seed
                ),
            ),
            ("svm", LinearSVC(C=config.svm_c, random_state=config.seed)),
        ],
        voting="hard",
    )
    models["ensemble"].fit(features, train_y)

    cv_scores = {"Naive Bayes": float(search.best_score_)}
    for key in ("logistic_regression", "svm", "ensemble"):
        scores = cross_val_score(
            make_pipeline(models[key], config),
            train_x,
            train_y,
            cv=cv,
            scoring="f1",
            n_jobs=1,
        )
        cv_scores[CLASSICAL_MODEL_NAMES[key]] = float(scores.mean())
        print(f"{CLASSICAL_MODEL_NAMES[key]} train CV F1: {scores.mean():.4f}")
    print(f"Naive Bayes best alpha: {search.best_params_['classifier__alpha']}")
    return vectorizer, models, cv_scores


def evaluate_classical(models, vectorizer, text, labels):
    features = vectorizer.transform(text)
    results, predictions = [], {}
    for key, model in models.items():
        name = CLASSICAL_MODEL_NAMES[key]
        predicted = model.predict(features)
        probability = (
            model.predict_proba(features)[:, 1]
            if key in ("naive_bayes", "logistic_regression")
            else None
        )
        results.append(evaluate_model(name, labels, predicted, probability))
        predictions[name] = predicted
    return results, predictions
