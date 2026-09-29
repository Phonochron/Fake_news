"""Orchestrate dataset preparation, model selection, and artifact output."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from fake_news.classical_models import evaluate_classical, train_classical
from fake_news.config import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_FILES,
    MODEL_SELECTION_FILE,
    OUTPUT_DIR,
    TrainingConfig,
)
from fake_news.data import load_dataset, split_dataset
from fake_news.decision import compare_decision_rules
from fake_news.evaluation import select_model
from fake_news.lstm_model import evaluate_lstm, train_lstm
from fake_news.provenance import create_manifest, verify_manifest_pair, write_manifest
from fake_news.reporting import save_reports


def save_models(model_dir: Path, vectorizer, models, tokenizer, lstm, selected: str):
    """Persist the fitted models and validation-selected decision rule."""
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {"vectorizer": vectorizer, "tokenizer": tokenizer, **models}
    for key, artifact in artifacts.items():
        joblib.dump(artifact, model_dir / MODEL_FILES[key])
    lstm.save(model_dir / MODEL_FILES["lstm"])
    (model_dir / MODEL_SELECTION_FILE).write_text(
        json.dumps(
            {"model": selected, "strategy": "validation_selected_model"}, indent=2
        ),
        encoding="utf-8",
    )


def train_and_evaluate(
    model_dir: Path, output_dir: Path, data_dir: Path, config: TrainingConfig
):
    config.validate()
    np.random.seed(config.seed)
    from tensorflow import random as tf_random

    tf_random.set_seed(config.seed)
    dataset, audit = load_dataset(data_dir, config.seed)
    train_x, val_x, test_x, train_y, val_y, test_y = split_dataset(dataset, config.seed)
    split_sizes = {"train": len(train_x), "validation": len(val_x), "test": len(test_x)}
    print(f"Prepared {audit['unique_rows']} unique articles; split: {split_sizes}")

    vectorizer, models, cv_scores = train_classical(train_x, train_y, config)
    tokenizer, lstm = train_lstm(train_x, train_y, config)

    validation, validation_predictions = evaluate_classical(
        models, vectorizer, val_x, val_y
    )
    lstm_validation, validation_predictions["LSTM"] = evaluate_lstm(
        lstm, tokenizer, val_x, val_y
    )
    validation.append(lstm_validation)
    selected = select_model(validation)
    decision = compare_decision_rules(
        validation, validation_predictions, val_y, selected
    )
    print(f"Selected on validation F1: {selected}")
    print(
        "Legacy five-model majority validation F1: "
        f"{decision['five_model_majority_validation_f1']:.4f}"
    )

    test, test_predictions = evaluate_classical(models, vectorizer, test_x, test_y)
    lstm_test, test_predictions["LSTM"] = evaluate_lstm(lstm, tokenizer, test_x, test_y)
    test.append(lstm_test)

    save_models(model_dir, vectorizer, models, tokenizer, lstm, selected)
    save_reports(
        validation,
        test,
        cv_scores,
        selected,
        test_predictions,
        test_y,
        vectorizer,
        models["naive_bayes"],
        audit,
        split_sizes,
        output_dir,
        cv_folds=config.cv_folds,
        seed=config.seed,
        decision=decision,
    )
    manifest = create_manifest(
        config, data_dir, model_dir, output_dir, audit, split_sizes, decision, test
    )
    write_manifest(manifest, model_dir, output_dir)
    verify_manifest_pair(model_dir, output_dir, check_files=True)
    print(f"Training complete; selected model: {selected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cv-folds", type=int, default=defaults.cv_folds)
    parser.add_argument("--lstm-epochs", type=int, default=defaults.lstm_epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--tfidf-max-features", type=int, default=defaults.tfidf_max_features
    )
    parser.add_argument("--svm-c", type=float, default=defaults.svm_c)
    parser.add_argument("--lstm-batch-size", type=int, default=defaults.lstm_batch_size)
    args = parser.parse_args()
    config = TrainingConfig(
        seed=args.seed,
        cv_folds=args.cv_folds,
        lstm_epochs=args.lstm_epochs,
        tfidf_max_features=args.tfidf_max_features,
        svm_c=args.svm_c,
        lstm_batch_size=args.lstm_batch_size,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    train_and_evaluate(args.model_dir, args.output_dir, args.data_dir, config)
