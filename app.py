# =========================================================
# FAKE NEWS DETECTION AI
# STREAMLIT WEB APP
# =========================================================

import re
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="📰",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    nb_model = joblib.load(
        "models/naive_bayes.pkl"
    )

    lr_model = joblib.load(
        "models/logistic_regression.pkl"
    )

    svm_model = joblib.load(
        "models/svm.pkl"
    )

    ensemble_model = joblib.load(
        "models/ensemble.pkl"
    )

    tokenizer = joblib.load(
        "models/tokenizer.pkl"
    )

    lstm_model = load_model(
        "models/lstm_model.h5"
    )

    return (
        vectorizer,
        nb_model,
        lr_model,
        svm_model,
        ensemble_model,
        tokenizer,
        lstm_model
    )

(
    vectorizer,
    nb_model,
    lr_model,
    svm_model,
    ensemble_model,
    tokenizer,
    lstm_model
) = load_models()

# =========================================================
# CONSTANT
# =========================================================

MAX_LEN = 100

# =========================================================
# PREPROCESSING
# =========================================================

try:
    stop_words = set(
        stopwords.words("english")
    )
except LookupError:
    nltk.download(
        "stopwords",
        quiet=True
    )

    stop_words = set(
        stopwords.words("english")
    )

stemmer = PorterStemmer()

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📰 Fake News AI")

    st.markdown("""
    ### Features

    ✅ Naive Bayes

    ✅ Logistic Regression

    ✅ Linear SVM

    ✅ Voting Ensemble

    ✅ LSTM Neural Network

    ✅ Confusion Matrix

    ✅ Metrics Dashboard

    """)

    st.divider()

    st.info(
        "Built with Streamlit, "
        "Scikit-Learn, and TensorFlow"
    )

# =========================================================
# TITLE
# =========================================================

st.title(
    "📰 Fake News Detection AI"
)

st.markdown(
    """
    Detect whether a news article is
    **HOAX** or **REAL**
    using Machine Learning
    and Deep Learning.
    """
)

# =========================================================
# INPUT
# =========================================================

news_text = st.text_area(
    "Paste News Here",
    height=250,
    placeholder="Paste article text..."
)

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button(
    "🚀 Analyze News",
    use_container_width=True
):

    if news_text.strip() == "":

        st.warning(
            "Please enter news text."
        )

    else:

        cleaned = clean_text(
            news_text
        )

        vector_news = vectorizer.transform(
            [cleaned]
        )

        # =====================================
        # NAIVE BAYES
        # =====================================

        nb_pred = nb_model.predict(
            vector_news
        )[0]

        nb_conf = np.max(

            nb_model.predict_proba(
                vector_news
            )[0]

        )

        # =====================================
        # LOGISTIC REGRESSION
        # =====================================

        lr_pred = lr_model.predict(
            vector_news
        )[0]

        lr_conf = np.max(

            lr_model.predict_proba(
                vector_news
            )[0]

        )

        # =====================================
        # SVM
        # =====================================

        svm_pred = svm_model.predict(
            vector_news
        )[0]

        # =====================================
        # ENSEMBLE
        # =====================================

        ensemble_pred = ensemble_model.predict(
            vector_news
        )[0]

        # =====================================
        # LSTM
        # =====================================

        seq = tokenizer.texts_to_sequences(
            [cleaned]
        )

        pad = pad_sequences(
            seq,
            maxlen=MAX_LEN
        )

        lstm_prob = lstm_model.predict(
            pad,
            verbose=0
        )[0][0]

        lstm_pred = (
            1 if lstm_prob > 0.5
            else 0
        )

        # =====================================
        # RESULT TABLE
        # =====================================

        st.subheader(
            "Prediction Results"
        )

        result_df = pd.DataFrame({

            "Model":[

                "Naive Bayes",

                "Logistic Regression",

                "Linear SVM",

                "Voting Ensemble",

                "LSTM"

            ],

            "Prediction":[

                "HOAX"
                if nb_pred
                else "REAL",

                "HOAX"
                if lr_pred
                else "REAL",

                "HOAX"
                if svm_pred
                else "REAL",

                "HOAX"
                if ensemble_pred
                else "REAL",

                "HOAX"
                if lstm_pred
                else "REAL"

            ]

        })

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # =====================================
        # CONFIDENCE
        # =====================================

        st.subheader(
            "Confidence Score"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Naive Bayes",
                f"{nb_conf*100:.2f}%"
            )

        with col2:

            st.metric(
                "Logistic Regression",
                f"{lr_conf*100:.2f}%"
            )

        # =====================================
        # FINAL DECISION
        # =====================================

        votes = [

            nb_pred,

            lr_pred,

            svm_pred,

            ensemble_pred,

            lstm_pred

        ]

        final_result = (
            1
            if sum(votes) >= 3
            else 0
        )

        st.divider()

        if final_result == 1:

            st.error(
                "🚨 FINAL RESULT: HOAX NEWS"
            )

        else:

            st.success(
                "✅ FINAL RESULT: REAL NEWS"
            )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.subheader(
    "📊 Model Performance"
)

try:

    metrics_df = pd.read_csv(
        "outputs/metrics.csv"
    )

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

except:

    st.info(
        "Run train_model.py first."
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.divider()

st.subheader(
    "📈 Confusion Matrix"
)

try:

    st.image(
        "outputs/confusion_matrix.png",
        use_container_width=True
    )

except:

    st.warning(
        "Confusion matrix not found."
    )

# =========================================================
# TOP HOAX WORDS
# =========================================================

st.divider()

st.subheader(
    "🔥 Top Hoax Words"
)

try:

    top_words = pd.read_csv(
        "outputs/top_hoax_words.csv"
    )

    st.dataframe(
        top_words,
        use_container_width=True
    )

except:

    st.warning(
        "Top hoax words file not found."
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Fake News Detection AI • "
    "Powered by Streamlit, "
    "Scikit-Learn, TensorFlow"
)
