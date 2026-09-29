"""Streamlit interface for the local launcher and Community Cloud entrypoint."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit executes this file directly and may only add its directory to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fake_news.config import (  # noqa: E402
    MAX_INPUT_CHARACTERS,
    MIN_RECOMMENDED_WORDS,
    MODEL_DIR,
    OUTPUT_DIR,
)
from fake_news.prediction import load_models, predict_news  # noqa: E402
from fake_news.provenance import verify_manifest_pair  # noqa: E402

EXAMPLE_ARTICLE = (
    "On Tuesday, the Riverside City Library announced that its reading room "
    "will open at 9 a.m. next month. The library said the schedule was approved "
    "at a public board meeting and posted on its website. Residents can check "
    "the official notice for changes."
)


@st.cache_resource
def cached_models():
    verify_manifest_pair(MODEL_DIR, OUTPUT_DIR, check_files=True)
    return load_models()


def show_report(title: str, filename: str, *, image: bool = False) -> None:
    st.divider()
    st.subheader(title)
    path = OUTPUT_DIR / filename
    if not path.is_file():
        st.info(
            f"{filename} is unavailable. Run python -m fake_news.train to generate it."
        )
    elif image:
        try:
            st.image(path.read_bytes())
        except OSError as exc:
            st.warning(f"Could not read {filename}: {exc}")
    else:
        try:
            report = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            st.warning(f"Could not read {filename}: {exc}")
            return
        st.dataframe(report, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Fake News Detection AI", page_icon="📰", layout="wide"
    )
    with st.sidebar:
        st.title("📰 Fake News AI")
        st.markdown(
            "Naive Bayes · Logistic Regression · Linear SVM · Voting Ensemble · LSTM"
        )
        st.divider()
        st.info("Built with Streamlit, Scikit-Learn, and TensorFlow")

    st.title("📰 Fake News Detection AI")
    st.markdown("Classify English news text patterns as **HOAX** or **REAL**.")
    st.info(
        "This is a text-pattern classifier, not a fact checker. Verify claims "
        "against reliable sources before drawing conclusions."
    )
    st.caption(
        "Trained on English-language articles. Results for other languages "
        "have not been validated."
    )
    example = st.selectbox(
        "Input example",
        ("Write or paste your own", "Fictional English sample (no truth label)"),
    )
    news_text = st.text_area(
        "Paste News Here",
        value=EXAMPLE_ARTICLE if example.startswith("Fictional") else "",
        height=250,
        max_chars=MAX_INPUT_CHARACTERS,
        placeholder="Paste article text...",
        key=example,
    )
    st.caption(f"Maximum {MAX_INPUT_CHARACTERS:,} characters per article.")
    if st.button("🚀 Analyze News", use_container_width=True):
        if not news_text.strip():
            st.warning("Please enter news text.")
        else:
            if len(news_text.split()) < MIN_RECOMMENDED_WORDS:
                st.warning(
                    f"This is shorter than {MIN_RECOMMENDED_WORDS} words; "
                    "the classification may be less reliable."
                )
            try:
                result = predict_news(news_text, cached_models())
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.subheader("Prediction Results")
                rows = [
                    {
                        "Model": name,
                        "Predicted class": "HOAX" if label else "REAL",
                        "P(HOAX)": (
                            f"{result.hoax_probability[name]:.1%}"
                            if name in result.hoax_probability
                            else "N/A"
                        ),
                        "Selected": "Yes" if name == result.selected_model else "",
                    }
                    for name, label in result.predictions.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                st.caption(
                    "P(HOAX) is an uncalibrated model output, not the probability "
                    "that a claim is false. SVM and Voting Ensemble do not provide it."
                )
                st.divider()
                if result.selected_model:
                    st.caption(
                        f"Final decision uses {result.selected_model}, "
                        "selected on validation F1."
                    )
                else:
                    st.caption("Final decision uses the majority vote of five models.")
                if result.term_contributions:
                    st.subheader("Terms affecting the selected model score")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Term": term,
                                    "Direction": "HOAX" if score > 0 else "REAL",
                                    "Score contribution": round(score, 3),
                                }
                                for term, score in result.term_contributions
                            ]
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "These are preprocessed TF-IDF terms affecting a linear "
                        "model score. They are not evidence about the article's truth."
                    )
                if result.final_label:
                    st.error("MODEL PREDICTION: HOAX")
                else:
                    st.success("MODEL PREDICTION: REAL")

    show_report("📊 Model Performance", "metrics.csv")
    show_report("📈 Confusion Matrix", "confusion_matrix.png", image=True)
    show_report("🔥 Naive Bayes Terms Associated with HOAX", "top_hoax_words.csv")
    st.divider()
    st.caption(
        "Fake News Detection AI · Powered by Streamlit, Scikit-Learn, TensorFlow"
    )


if __name__ == "__main__":
    main()
