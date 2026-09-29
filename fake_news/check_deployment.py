"""Measure local startup, model size, and representative input behavior."""

import importlib
import json
from time import perf_counter


def _measure_prediction(text, models, predict_news) -> float:
    started = perf_counter()
    predict_news(text, models)
    return round(perf_counter() - started, 3)


def main() -> None:
    started = perf_counter()
    importlib.import_module("tensorflow")
    tensorflow_seconds = round(perf_counter() - started, 3)

    from fake_news.config import MAX_INPUT_CHARACTERS, MODEL_DIR, OUTPUT_DIR
    from fake_news.prediction import load_models, predict_news
    from fake_news.provenance import verify_manifest_pair

    manifest = verify_manifest_pair(MODEL_DIR, OUTPUT_DIR, check_files=True)
    started = perf_counter()
    models = load_models()
    load_seconds = round(perf_counter() - started, 3)

    short_text = "Council meeting announced."
    typical_text = "The city council published a report after its public meeting. " * 5
    long_text = (
        "The city council published a report after its public meeting. " * 1_000
    )[:MAX_INPUT_CHARACTERS]
    latencies = {
        "short": _measure_prediction(short_text, models, predict_news),
        "typical": _measure_prediction(typical_text, models, predict_news),
        "long": _measure_prediction(long_text, models, predict_news),
    }

    rejected = []
    for name, text in (
        ("empty", ""),
        ("over_limit", "x" * (MAX_INPUT_CHARACTERS + 1)),
    ):
        try:
            predict_news(text, models)
        except ValueError:
            rejected.append(name)
    if len(rejected) != 2:
        raise RuntimeError("Input boundary checks did not reject invalid text")

    sizes = {
        path.name: path.stat().st_size
        for path in sorted(MODEL_DIR.iterdir())
        if path.is_file()
    }
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"] if manifest else None,
                "tensorflow_import_seconds": tensorflow_seconds,
                "model_load_seconds": load_seconds,
                "prediction_seconds": latencies,
                "max_input_characters": MAX_INPUT_CHARACTERS,
                "invalid_inputs_rejected": rejected,
                "model_total_bytes": sum(sizes.values()),
                "model_file_bytes": sizes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
