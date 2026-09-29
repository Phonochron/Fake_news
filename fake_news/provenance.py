"""Dataset fingerprints and training-run manifests."""

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from fake_news.config import (
    MAX_LEN,
    MAX_WORDS,
    MODEL_FILES,
    MODEL_SELECTION_FILE,
    TRAINING_MANIFEST_FILE,
    TrainingConfig,
)

LIBRARIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "keras",
    "nltk",
    "streamlit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(
    config: TrainingConfig,
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    audit: dict,
    split_sizes: dict[str, int],
    decision: dict,
    test_results: list[dict],
) -> dict:
    selected = decision["selected_model"]
    test_f1 = next(row["F1"] for row in test_results if row["Model"] == selected)
    return {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            name: {"sha256": sha256_file(data_dir / name)}
            for name in ("Fake.csv", "True.csv")
        },
        "dataset_rows": {
            "raw": audit["raw_rows"],
            "unique_usable": audit["unique_rows"],
        },
        "split_sizes": split_sizes,
        "training_config": {
            **asdict(config),
            "max_words": MAX_WORDS,
            "max_len": MAX_LEN,
        },
        "library_versions": {
            "python": platform.python_version(),
            **{library: version(library) for library in LIBRARIES},
        },
        "decision": decision,
        "selected_test_f1": float(test_f1),
        "model_sha256": {
            name: sha256_file(model_dir / name)
            for name in (*MODEL_FILES.values(), MODEL_SELECTION_FILE)
        },
        "report_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != TRAINING_MANIFEST_FILE
        },
    }


def write_manifest(manifest: dict, model_dir: Path, output_dir: Path) -> None:
    content = json.dumps(manifest, indent=2, ensure_ascii=False)
    for directory in (model_dir, output_dir):
        (directory / TRAINING_MANIFEST_FILE).write_text(content, encoding="utf-8")


def verify_manifest_pair(
    model_dir: Path, output_dir: Path, *, check_files: bool = False
) -> dict | None:
    paths = [
        directory / TRAINING_MANIFEST_FILE for directory in (model_dir, output_dir)
    ]
    if not any(path.exists() for path in paths):
        return None  # Legacy artifacts from before manifests were added.
    if not all(path.is_file() for path in paths):
        raise RuntimeError("Model and report manifests are incomplete")
    try:
        model_manifest, output_manifest = (
            json.loads(path.read_text(encoding="utf-8")) for path in paths
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("Training manifest could not be read") from exc
    if model_manifest != output_manifest:
        raise RuntimeError("Model and report manifests describe different runs")
    try:
        selection = json.loads(
            (model_dir / MODEL_SELECTION_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("Selected-model metadata could not be read") from exc
    if selection.get("model") != model_manifest["decision"]["selected_model"]:
        raise RuntimeError("Selected model differs from training manifest")
    if selection.get("strategy") not in (
        None,
        model_manifest["decision"].get("strategy"),
    ):
        raise RuntimeError("Decision strategy differs from training manifest")
    if check_files:
        for directory, key in (
            (model_dir, "model_sha256"),
            (output_dir, "report_sha256"),
        ):
            for name, expected in model_manifest[key].items():
                try:
                    actual = sha256_file(directory / name)
                except OSError as exc:
                    raise RuntimeError(f"Artifact is missing: {name}") from exc
                if actual != expected:
                    raise RuntimeError(f"Artifact differs from manifest: {name}")
    return model_manifest
