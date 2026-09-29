"""Protect the validation selection and fold-local vectorization rules."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from fake_news.classical_models import make_pipeline, train_classical
from fake_news.config import TrainingConfig
from fake_news.decision import compare_decision_rules
from fake_news.evaluation import select_model
from fake_news.explanations import linear_term_contributions
from fake_news.reporting import save_reports, save_top_terms


class EvaluationTests(unittest.TestCase):
    def test_classical_cv_fits_tfidf_inside_pipeline(self):
        texts = pd.Series(
            [f"fabricated rumor claim item{chr(97 + i)}" for i in range(12)]
            + [f"verified official report item{chr(97 + i)}" for i in range(12)]
        )
        labels = pd.Series([1] * 12 + [0] * 12)
        pipeline = make_pipeline(MultinomialNB())
        self.assertEqual(list(pipeline.named_steps), ["tfidf", "classifier"])
        vectorizer, models, cv_scores = train_classical(
            texts, labels, TrainingConfig(cv_folds=3)
        )
        self.assertEqual(len(models), 4)
        self.assertEqual(len(cv_scores), 4)
        self.assertGreater(len(vectorizer.vocabulary_), 0)

    def test_report_keeps_validation_choice_even_if_other_test_f1_is_higher(self):
        validation = [
            {
                "Model": "Naive Bayes",
                "Accuracy": 0.8,
                "Precision": 0.8,
                "Recall": 0.8,
                "F1": 0.8,
                "ROC_AUC": 0.8,
            },
            {
                "Model": "LSTM",
                "Accuracy": 0.7,
                "Precision": 0.7,
                "Recall": 0.7,
                "F1": 0.7,
                "ROC_AUC": 0.7,
            },
        ]
        test = [
            {**validation[0], "F1": 0.6},
            {**validation[1], "F1": 0.9},
        ]
        selected = select_model(validation)
        self.assertEqual(selected, "Naive Bayes")
        vectorizer = Mock()
        vectorizer.get_feature_names_out.return_value = np.array(["claim", "report"])
        naive_bayes = Mock()
        naive_bayes.feature_log_prob_ = np.array([[-2.0, -1.0], [-1.0, -2.0]])
        audit = {
            "raw_rows": 10,
            "unique_rows": 8,
            "duplicate_clean_text_rows": 2,
            "conflicting_label_rows": 0,
            "reuters_marker_rate": {"0": 0.9, "1": 0.1},
            "subject_by_label": [{"subject": "news", "real": 4, "fake": 0}],
            "date_min": "2018-01-01",
            "date_max": "2018-01-02",
            "unparsed_date_rows": 0,
        }
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_reports(
                validation,
                test,
                {"Naive Bayes": 0.8},
                selected,
                {"Naive Bayes": [0, 1], "LSTM": [0, 1]},
                [0, 1],
                vectorizer,
                naive_bayes,
                audit,
                {"train": 6, "validation": 1, "test": 1},
                output_dir,
            )
            metrics = pd.read_csv(output_dir / "metrics.csv")
            chosen = metrics.loc[metrics["Selected_On_Validation"]]
            self.assertEqual(chosen.iloc[0]["Model"], "Naive Bayes")
            self.assertIn(
                "Selected model: **Naive Bayes**",
                (output_dir / "evaluation_report.md").read_text(),
            )

    def test_top_terms_use_class_log_odds(self):
        vectorizer = Mock()
        vectorizer.get_feature_names_out.return_value = np.array(["common", "signal"])
        naive_bayes = Mock()
        naive_bayes.feature_log_prob_ = np.array([[-0.1, -10.0], [-1.0, -2.0]])
        with TemporaryDirectory() as directory:
            save_top_terms(vectorizer, naive_bayes, Path(directory))
            terms = pd.read_csv(Path(directory) / "top_hoax_words.csv")
        self.assertEqual(terms.iloc[0]["Term"], "signal")
        self.assertEqual(terms.iloc[0]["Naive_Bayes_Log_Odds"], 8.0)

    def test_linear_explanation_uses_signed_local_contributions(self):
        model = LinearSVC()
        model.coef_ = np.array([[2.0, -3.0, 0.5]])
        vectorizer = Mock()
        vectorizer.get_feature_names_out.return_value = np.array(
            ["official", "rumor", "unused"]
        )
        features = csr_matrix([[0.5, 1.0, 0.0]])
        contributions = linear_term_contributions(model, vectorizer, features)
        self.assertEqual(contributions, (("rumor", -3.0), ("official", 1.0)))

    def test_selected_model_beats_correlated_legacy_vote_on_validation(self):
        labels = np.array([1, 1, 0, 0])
        predictions = {
            "Naive Bayes": np.array([0, 0, 0, 0]),
            "Logistic Regression": np.array([0, 1, 0, 0]),
            "Linear SVM": labels,
            "Voting Ensemble": np.array([0, 1, 0, 0]),
            "LSTM": labels,
        }
        decision = compare_decision_rules(
            [{"Model": "Linear SVM", "F1": 1.0}], predictions, labels, "Linear SVM"
        )
        self.assertEqual(decision["strategy"], "validation_selected_model")
        self.assertLess(decision["five_model_majority_validation_f1"], 1.0)
