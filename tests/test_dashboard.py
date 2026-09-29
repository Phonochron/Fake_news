"""Exercise the user visible Streamlit prediction flow without loading model files."""

import unittest
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from fake_news.prediction import PredictionResult


class DashboardTests(unittest.TestCase):
    app_path = Path(__file__).resolve().parent.parent / "fake_news" / "dashboard.py"

    def test_dashboard_imports_package_when_run_as_nested_script(self):
        project_root = self.app_path.parent.parent
        script = """
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path = [path for path in sys.path if Path(path or '.').resolve() != root]
runpy.run_path(str(root / 'fake_news' / 'dashboard.py'), run_name='import_check')
"""
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script, str(project_root)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_report_image_uses_file_bytes(self):
        from fake_news.dashboard import show_report

        with TemporaryDirectory() as directory:
            image_bytes = b"\x89PNG\r\n\x1a\nexample"
            (Path(directory) / "plot.png").write_bytes(image_bytes)
            with (
                patch("fake_news.dashboard.OUTPUT_DIR", Path(directory)),
                patch("fake_news.dashboard.st.divider"),
                patch("fake_news.dashboard.st.subheader"),
                patch("fake_news.dashboard.st.image") as image,
            ):
                show_report("Plot", "plot.png", image=True)

        image.assert_called_once_with(image_bytes)

    def test_empty_input_requests_news(self):
        app = AppTest.from_file(str(self.app_path)).run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 1)

    def test_english_example_fills_text_area(self):
        app = AppTest.from_file(str(self.app_path)).run(timeout=30)
        app.selectbox[0].select("Fictional English sample (no truth label)").run(
            timeout=30
        )
        self.assertIn("Riverside City Library", app.text_area[0].value)

    def test_prediction_result_is_displayed(self):
        result = PredictionResult(
            predictions={
                "Naive Bayes": 1,
                "Logistic Regression": 1,
                "Linear SVM": 0,
                "Voting Ensemble": 1,
                "LSTM": 1,
            },
            hoax_probability={
                "Naive Bayes": 0.9,
                "Logistic Regression": 0.8,
                "LSTM": 0.7,
            },
            final_label=1,
        )
        with (
            patch("fake_news.prediction.load_models", return_value=object()),
            patch("fake_news.prediction.predict_news", return_value=result),
        ):
            app = AppTest.from_file(str(self.app_path)).run(timeout=30)
            app.text_area[0].input("A test article")
            app.button[0].click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        prediction_table = app.dataframe[0].value
        self.assertEqual(prediction_table.loc[0, "P(HOAX)"], "90.0%")
        self.assertEqual(prediction_table.loc[2, "P(HOAX)"], "N/A")
        self.assertTrue(
            any("MODEL PREDICTION: HOAX" in item.value for item in app.error)
        )
        self.assertTrue(any("shorter than" in item.value for item in app.warning))

    def test_selected_model_is_explained_with_final_result(self):
        result = PredictionResult(
            predictions={"Linear SVM": 0, "LSTM": 1},
            hoax_probability={},
            final_label=0,
            selected_model="Linear SVM",
            term_contributions=(("offici", 1.2), ("rumor", -0.8)),
        )
        with (
            patch("fake_news.prediction.load_models", return_value=object()),
            patch("fake_news.prediction.predict_news", return_value=result),
        ):
            app = AppTest.from_file(str(self.app_path)).run(timeout=30)
            app.text_area[0].input("A test article")
            app.button[0].click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("MODEL PREDICTION: REAL" in item.value for item in app.success)
        )
        self.assertTrue(any("Linear SVM" in item.value for item in app.caption))
        self.assertEqual(app.dataframe[1].value.loc[0, "Term"], "offici")

    def test_mismatched_training_run_is_shown_as_error(self):
        with patch(
            "fake_news.provenance.verify_manifest_pair",
            side_effect=RuntimeError(
                "Model and report manifests describe different runs"
            ),
        ):
            app = AppTest.from_file(str(self.app_path)).run(timeout=30)
            app.text_area[0].input("A test article")
            app.button[0].click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("different runs" in item.value for item in app.error))
