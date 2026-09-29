"""Write evaluation tables, plots, and methodology notes."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from fake_news.config import OUTPUT_DIR, RANDOM_STATE


def save_reports(
    validation_results: list[dict],
    test_results: list[dict],
    cv_scores: dict[str, float],
    selected_model: str,
    test_predictions: dict,
    test_labels,
    vectorizer,
    naive_bayes,
    audit: dict,
    split_sizes: dict[str, int],
    output_dir: Path = OUTPUT_DIR,
    cv_folds: int = 5,
    seed: int = RANDOM_STATE,
    decision: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame(validation_results).sort_values("F1", ascending=False)
    test = pd.DataFrame(test_results).sort_values("F1", ascending=False)
    test["Selected_On_Validation"] = test["Model"].eq(selected_model)
    validation.to_csv(output_dir / "validation_metrics.csv", index=False)
    test.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [{"Model": name, "Train_CV_F1": score} for name, score in cv_scores.items()]
    ).to_csv(output_dir / "cv_metrics.csv", index=False)
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if decision is not None:
        (output_dir / "decision_policy.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
    pd.DataFrame(audit["subject_by_label"]).to_csv(
        output_dir / "subject_label_counts.csv", index=False
    )

    matrix = confusion_matrix(
        test_labels, test_predictions[selected_model], labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Real", "Hoax"],
        yticklabels=["Real", "Hoax"],
    )
    ax.set(
        title=f"Confusion Matrix - {selected_model}",
        xlabel="Predicted",
        ylabel="Actual",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    save_top_terms(vectorizer, naive_bayes, output_dir)
    _write_methodology(
        output_dir,
        audit,
        split_sizes,
        selected_model,
        validation,
        test,
        cv_folds,
        seed,
        decision,
    )


def save_top_terms(vectorizer, naive_bayes, output_dir: Path, limit: int = 50) -> None:
    """Rank terms by their Naive Bayes log likelihood difference."""
    terms = vectorizer.get_feature_names_out()
    log_odds = naive_bayes.feature_log_prob_[1] - naive_bayes.feature_log_prob_[0]
    top_indices = np.argsort(log_odds)[-limit:][::-1]
    pd.DataFrame(
        {
            "Term": terms[top_indices],
            "Naive_Bayes_Log_Odds": log_odds[top_indices],
        }
    ).to_csv(output_dir / "top_hoax_words.csv", index=False)


def _write_methodology(
    output_dir,
    audit,
    split_sizes,
    selected_model,
    validation,
    test,
    cv_folds,
    seed,
    decision,
):
    validation_f1 = validation.set_index("Model").loc[selected_model, "F1"]
    test_f1 = test.set_index("Model").loc[selected_model, "F1"]
    subject_is_proxy = all(
        not (row["real"] and row["fake"]) for row in audit["subject_by_label"]
    )
    lines = [
        "# Evaluation report",
        "",
        f"Selected model: **{selected_model}** (validation F1: {validation_f1:.4f}; "
        f"held-out test F1: {test_f1:.4f}).",
        "",
        "## Method",
        "",
        f"- Random seed: {seed}. Stratified split after text deduplication: "
        f"train {split_sizes['train']}, validation {split_sizes['validation']}, "
        f"test {split_sizes['test']}.",
        f"- TF-IDF is fitted inside each of the {cv_folds} classical-model "
        "cross-validation folds. "
        "Hyperparameters use training folds only.",
        "- The Naive Bayes CV score is the best score from its alpha search. "
        "The ensemble uses that selected alpha during CV. These CV scores are "
        "diagnostic, not unbiased nested-CV estimates.",
        "- LSTM early stopping uses a subset of the training split. "
        "The separate validation split selects one model by F1.",
        "- The test split is evaluated only after selection. All five models are "
        "reported for comparison; the test results do not change the selected model.",
        "- Duplicate normalized articles and conflicting-label articles are removed "
        "before splitting; the explicit Reuters token is removed during preprocessing.",
        "",
        "## Dataset audit",
        "",
        f"- Raw rows: {audit['raw_rows']}; unique usable rows: {audit['unique_rows']}; "
        f"duplicate normalized rows: {audit['duplicate_clean_text_rows']}; "
        f"conflicting-label rows: {audit['conflicting_label_rows']}.",
        "- Reuters marker rate before preprocessing: "
        f"real {audit['reuters_marker_rate']['0']:.1%}, "
        f"fake {audit['reuters_marker_rate']['1']:.1%}.",
        f"- Subject categories are exclusive to one label: {subject_is_proxy}. "
        "Subject and date are excluded from model features.",
        f"- Date range: {audit['date_min']} to {audit['date_max']}; "
        f"unparsed dates: {audit['unparsed_date_rows']}.",
        "",
        "## Interpretation",
        "",
        "The dataset has strong source/style cues even after removing the explicit "
        "Reuters token. These scores measure performance on this dataset's random "
        "holdout, not independent fact checking or generalization to unseen publishers. "
        "No reliable publisher identifier is available for a source-held-out evaluation.",
        "The top-term table ranks Naive Bayes log likelihood differences between "
        "HOAX and REAL classes. It is a dataset-level association, not an explanation "
        f"of an individual prediction or the selected {selected_model} model.",
        "",
        "See `data_audit.json`, `subject_label_counts.csv`, `cv_metrics.csv`, "
        "`validation_metrics.csv`, and `metrics.csv` for the underlying counts and scores.",
        "",
    ]
    if decision is not None:
        lines.extend(
            [
                "## Final decision rule",
                "",
                f"The selected model scored {decision['selected_validation_f1']:.4f} "
                "F1 on validation. The legacy five-model majority scored "
                f"{decision['five_model_majority_validation_f1']:.4f} F1 on the "
                "same validation split. The app uses the selected model. Voting "
                "Ensemble already combines three of the other votes, so a final "
                "majority counts those correlated models again.",
                "",
                "See `decision_policy.json` and `training_manifest.json` for the "
                "saved rule and provenance of this run.",
                "",
            ]
        )
    (output_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
