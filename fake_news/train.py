"""Training entry point: python -m fake_news.train."""

import importlib


def main() -> None:
    # On the tested Windows stack TensorFlow must load before sklearn/matplotlib.
    importlib.import_module("tensorflow")
    from fake_news.training import main as run_training

    run_training()


if __name__ == "__main__":
    main()
