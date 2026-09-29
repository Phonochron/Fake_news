from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from fake_news.config import MAX_INPUT_CHARACTERS, MODEL_FILES, TrainingConfig
from fake_news.data import load_dataset, prepare_dataset, split_dataset
from fake_news.prediction import ModelBundle, load_models, predict_news
from fake_news.preprocessing import clean_text
from fake_news.sequences import encode_sequences


class PipelineTests(unittest.TestCase):
    def test_preprocessing_removes_urls_numbers_and_stopwords(self):
        self.assertEqual(
            clean_text("The NEWS at https://example.com costs 100 dollars!"),
            "news cost dollar",
        )

    def test_preprocessing_removes_source_marker(self):
        self.assertNotIn("reuter", clean_text("WASHINGTON (Reuters) - Officials spoke"))

    def test_dataset_labels_and_split(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            for filename in ("Fake.csv", "True.csv"):
                pd.DataFrame(
                    {
                        "title": [
                            f"{filename} headline token{chr(97 + i) * 2}"
                            for i in range(20)
                        ],
                        "text": ["News article" for _ in range(20)],
                        "subject": ["fake" if filename == "Fake.csv" else "real"] * 20,
                        "date": ["January 1, 2018"] * 20,
                    }
                ).to_csv(data_dir / filename, index=False)
            dataset, audit = load_dataset(data_dir)
        self.assertEqual(len(dataset), 40)
        self.assertEqual(audit["duplicate_clean_text_rows"], 0)
        self.assertEqual(dataset["label"].value_counts().to_dict(), {0: 20, 1: 20})
        split = split_dataset(dataset)
        self.assertEqual([len(part) for part in split], [28, 6, 6, 28, 6, 6])
        train_text, val_text, test_text = (set(part) for part in split[:3])
        self.assertFalse(train_text & val_text)
        self.assertFalse(train_text & test_text)
        self.assertFalse(val_text & test_text)

    def test_csv_missing_column_has_clear_error(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pd.DataFrame({"title": ["x"], "text": ["y"]}).to_csv(
                data_dir / "Fake.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "date, subject"):
                load_dataset(data_dir)

    def test_blank_article_is_removed_and_recorded(self):
        raw = pd.DataFrame(
            {
                "title": ["", "Real story", "Fake story"],
                "text": ["", "real article", "fake article"],
                "subject": ["", "news", "news"],
                "date": ["", "January 1, 2018", "January 1, 2018"],
                "label": [1, 0, 1],
            }
        )
        dataset, audit = prepare_dataset(raw)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(audit["blank_article_rows"], 1)
        self.assertEqual(audit["blank_subject_rows"], 1)

    def test_training_config_rejects_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "cv_folds"):
            TrainingConfig(cv_folds=1).validate()

    def test_deduplication_and_conflicting_labels_happen_before_split(self):
        raw = pd.DataFrame(
            {
                "title": [
                    "Unique fake",
                    "Unique fake",
                    "Unique real",
                    "Conflict",
                    "Conflict",
                ],
                "text": ["Story", "Story", "Reuters story", "Same story", "Same story"],
                "subject": ["fake", "fake", "real", "fake", "real"],
                "date": ["January 1, 2018"] * 5,
                "label": [1, 1, 0, 1, 0],
            }
        )
        dataset, audit = prepare_dataset(raw)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(audit["duplicate_clean_text_rows"], 1)
        self.assertEqual(audit["conflicting_label_rows"], 2)
        self.assertEqual(audit["reuters_marker_rate"]["0"], 0.5)

    def test_missing_artifacts_are_reported(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing model artifacts"):
                load_models(Path(directory))

    def test_incompatible_artifacts_report_required_version(self):
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            for name in MODEL_FILES.values():
                (model_dir / name).touch()
            mismatch = InconsistentVersionWarning(
                estimator_name="TfidfVectorizer",
                current_sklearn_version="1.4.2",
                original_sklearn_version="1.9.0",
            )
            with patch("fake_news.prediction.joblib.load", side_effect=mismatch):
                with self.assertRaisesRegex(RuntimeError, "scikit-learn 1.9.0"):
                    load_models(model_dir)

    def test_prediction_uses_post_padding_and_majority_vote(self):
        vectorizer = Mock()
        vectorizer.transform.return_value = "features"
        tokenizer = Mock()
        tokenizer.texts_to_sequences.return_value = [[7, 8, 9]]
        lstm = Mock()
        lstm.predict.return_value = np.array([[0.7]])

        def classifier(label, probability=None):
            model = Mock()
            model.predict.return_value = [label]
            if probability is not None:
                model.predict_proba.return_value = np.array([probability])
            return model

        models = ModelBundle(
            vectorizer=vectorizer,
            naive_bayes=classifier(1, [0.2, 0.8]),
            logistic_regression=classifier(0, [0.6, 0.4]),
            svm=classifier(1),
            ensemble=classifier(0),
            tokenizer=tokenizer,
            lstm=lstm,
        )
        result = predict_news("News article", models)
        self.assertEqual(result.final_label, 1)
        self.assertEqual(result.predictions["LSTM"], 1)
        self.assertEqual(
            result.hoax_probability,
            {"Naive Bayes": 0.8, "Logistic Regression": 0.4, "LSTM": 0.7},
        )
        padded = lstm.predict.call_args.args[0]
        self.assertEqual(padded.shape, (1, 100))
        self.assertEqual(padded[0, :4].tolist(), [7, 8, 9, 0])

        selected = predict_news(
            "News article", replace(models, selected_model="Logistic Regression")
        )
        self.assertEqual(selected.final_label, 0)
        self.assertEqual(selected.selected_model, "Logistic Regression")

    def test_prediction_rejects_only_stopwords(self):
        with self.assertRaisesRegex(ValueError, "No words remain"):
            predict_news("the and", Mock())

    def test_prediction_rejects_invalid_input_before_using_models(self):
        with self.assertRaisesRegex(ValueError, "must be a string"):
            predict_news(None, Mock())
        with self.assertRaisesRegex(ValueError, "exceeds"):
            predict_news("a" * (MAX_INPUT_CHARACTERS + 1), Mock())

    def test_sequence_encoding_post_pads_and_truncates(self):
        tokenizer = Mock()
        tokenizer.texts_to_sequences.return_value = [[1, 2], [3, 4, 5, 6]]
        encoded = encode_sequences(tokenizer, ["short", "long"], max_len=3)
        self.assertEqual(encoded.tolist(), [[1, 2, 0], [3, 4, 5]])
