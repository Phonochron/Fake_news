"""Ensure a model and its reports belong to the same training run."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fake_news.config import MODEL_SELECTION_FILE, TRAINING_MANIFEST_FILE
from fake_news.provenance import sha256_file, verify_manifest_pair, write_manifest


class ProvenanceTests(unittest.TestCase):
    def test_manifest_detects_changed_artifact_and_mismatched_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models, outputs = root / "models", root / "outputs"
            models.mkdir()
            outputs.mkdir()
            (models / MODEL_SELECTION_FILE).write_text(
                json.dumps({"model": "Linear SVM"}), encoding="utf-8"
            )
            (models / "svm.pkl").write_bytes(b"model")
            (outputs / "metrics.csv").write_text("F1\n0.9\n", encoding="utf-8")
            manifest = {
                "run_id": "same-run",
                "decision": {"selected_model": "Linear SVM"},
                "model_sha256": {"svm.pkl": sha256_file(models / "svm.pkl")},
                "report_sha256": {"metrics.csv": sha256_file(outputs / "metrics.csv")},
            }
            write_manifest(manifest, models, outputs)
            self.assertEqual(
                verify_manifest_pair(models, outputs, check_files=True)["run_id"],
                "same-run",
            )

            (outputs / "metrics.csv").write_text("F1\n0.1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs from manifest"):
                verify_manifest_pair(models, outputs, check_files=True)

            changed = {**manifest, "run_id": "different-run"}
            (outputs / TRAINING_MANIFEST_FILE).write_text(
                json.dumps(changed), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "different runs"):
                verify_manifest_pair(models, outputs)
