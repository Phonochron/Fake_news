"""Start Streamlit after loading TensorFlow on Windows.

The tested Windows dependency stack fails to initialize TensorFlow when
Streamlit is imported first. Keeping this order in one entry point makes the
runtime behavior explicit and leaves the dashboard focused on presentation.
"""

import importlib
import sys

from fake_news.config import ROOT_DIR


def main() -> None:
    importlib.import_module("tensorflow")
    from streamlit.web import cli

    dashboard = ROOT_DIR / "fake_news" / "dashboard.py"
    sys.argv = ["streamlit", "run", str(dashboard), *sys.argv[1:]]
    cli.main()


if __name__ == "__main__":
    main()
