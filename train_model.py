# =========================================================
# FAKE NEWS DETECTION AI
# TRAINING PIPELINE
# PART 1
# =========================================================

# =========================================================
# IMPORT LIBRARY
# =========================================================

import os
import re
import joblib
import nltk
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.naive_bayes import (
    MultinomialNB
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.svm import (
    LinearSVC
)

from sklearn.ensemble import (
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# TENSORFLOW
# =========================================================

from tensorflow.keras.models import (
    Sequential
)

from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.text import (
    Tokenizer
)

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

# =========================================================
# DOWNLOAD NLTK
# =========================================================

nltk.download("stopwords")

# =========================================================
# CREATE FOLDER
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)

os.makedirs(
    "data",
    exist_ok=True
)

# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

fake_df = pd.read_csv(
    "data/Fake.csv"
)

true_df = pd.read_csv(
    "data/True.csv"
)

# =========================================================
# LABELING
# =========================================================

# Fake = 1
# Real = 0

fake_df["label"] = 1
true_df["label"] = 0

# =========================================================
# COMBINE TITLE + TEXT
# =========================================================

fake_df["text"] = (
    fake_df["title"].astype(str)
    + " "
    + fake_df["text"].astype(str)
)

true_df["text"] = (
    true_df["title"].astype(str)
    + " "
    + true_df["text"].astype(str)
)

# =========================================================
# KEEP IMPORTANT COLUMN
# =========================================================

fake_df = fake_df[
    ["text", "label"]
]

true_df = true_df[
    ["text", "label"]
]

# =========================================================
# MERGE DATASET
# =========================================================

df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)

# Shuffle

df = df.sample(
    frac=1,
    random_state=42
)

df = df.reset_index(
    drop=True
)

print(
    f"Total Dataset : {len(df)}"
)

print(
    df.head()
)

# =========================================================
# PREPROCESSING
# =========================================================

print("\nPreparing preprocessing...")

stop_words = set(
    stopwords.words("english")
)

stemmer = PorterStemmer()

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = str(text)

    # lowercase

    text = text.lower()

    # remove url

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    # remove number

    text = re.sub(
        r"\d+",
        "",
        text
    )

    # remove punctuation

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    # remove extra spaces

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # tokenize

    words = text.split()

    # remove stopwords

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # stemming

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)

# =========================================================
# APPLY PREPROCESSING
# =========================================================

print("Cleaning text...")

df["clean_text"] = (
    df["text"]
    .astype(str)
    .apply(clean_text)
)

print(
    df[["clean_text", "label"]]
    .head()
)

# =========================================================
# SPLIT DATASET
# =========================================================

X = df["clean_text"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTrain Size:", len(X_train))
print("Validation Size:", len(X_val))
print("Test Size :", len(X_test))

# =========================================================
# TF-IDF
# =========================================================

print("\nTF-IDF Vectorization...")

vectorizer = TfidfVectorizer(

    max_features=20000,

    ngram_range=(1, 2),

    min_df=2,

    max_df=0.95,

    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    "TF-IDF Shape:",
    X_train_tfidf.shape
)

# =========================================================
# RESULT STORAGE
# =========================================================

results = []

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_prob=None
):

    acc = accuracy_score(
        y_true,
        y_pred
    )

    prec = precision_score(
        y_true,
        y_pred
    )

    rec = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    auc = np.nan

    if y_prob is not None:

        try:

            auc = roc_auc_score(
                y_true,
                y_prob
            )

        except:

            pass

    results.append([

        model_name,

        acc,

        prec,

        rec,

        f1,

        auc

    ])

    print("\n" + "=" * 60)

    print(model_name)

    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    print(
        f"Accuracy : {acc:.4f}"
    )

    print(
        f"Precision: {prec:.4f}"
    )

    print(
        f"Recall   : {rec:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    if not np.isnan(auc):

        print(
            f"ROC-AUC  : {auc:.4f}"
        )

# =========================================================
# OPTIMIZED NAIVE BAYES
# =========================================================

print("\n" + "=" * 60)
print("TRAINING OPTIMIZED NAIVE BAYES")
print("=" * 60)

param_grid = {

    "alpha": [

        0.001,
        0.01,
        0.1,
        0.5,
        1.0

    ]
}

grid_nb = GridSearchCV(

    estimator=MultinomialNB(),

    param_grid=param_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1,

    verbose=1

)

grid_nb.fit(

    X_train_tfidf,
    y_train

)

nb_model = grid_nb.best_estimator_

print(
    "\nBest Parameters:",
    grid_nb.best_params_
)

# =========================================================
# CROSS VALIDATION
# =========================================================

cv_scores = cross_val_score(

    nb_model,

    X_train_tfidf,

    y_train,

    cv=5,

    scoring="f1"

)

print(
    "\nNB Cross Validation F1:",
    cv_scores.mean()
)

# =========================================================
# PREDICTION
# =========================================================

nb_pred = nb_model.predict(
    X_test_tfidf
)

nb_prob = nb_model.predict_proba(
    X_test_tfidf
)[:, 1]

# =========================================================
# EVALUATION
# =========================================================

evaluate_model(

    "Naive Bayes",

    y_test,

    nb_pred,

    nb_prob

)

# =========================================================
# TOP HOAX WORDS
# =========================================================

print("\nTop Important Hoax Words")

feature_names = (
    vectorizer.get_feature_names_out()
)

top_hoax_idx = np.argsort(

    nb_model.feature_log_prob_[1]

)[-20:]

for idx in reversed(top_hoax_idx):

    print(
        feature_names[idx]
    )

# =========================================================
# LOGISTIC REGRESSION
# =========================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(

    max_iter=2000,

    n_jobs=-1,

    random_state=42

)

lr_model.fit(

    X_train_tfidf,

    y_train

)

lr_pred = lr_model.predict(
    X_test_tfidf
)

lr_prob = lr_model.predict_proba(
    X_test_tfidf
)[:,1]

evaluate_model(

    "Logistic Regression",

    y_test,

    lr_pred,

    lr_prob

)

# =========================================================
# LOGISTIC REGRESSION CV
# =========================================================

lr_cv = cross_val_score(

    lr_model,

    X_train_tfidf,

    y_train,

    cv=5,

    scoring="f1"

)

print(
    "\nLR CV F1:",
    lr_cv.mean()
)

# =========================================================
# SVM
# =========================================================

print("\n" + "=" * 60)
print("TRAINING LINEAR SVM")
print("=" * 60)

svm_model = LinearSVC(

    C=1.0,

    random_state=42

)

svm_model.fit(

    X_train_tfidf,

    y_train

)

svm_pred = svm_model.predict(
    X_test_tfidf
)

evaluate_model(

    "Linear SVM",

    y_test,

    svm_pred

)

# =========================================================
# SVM CV
# =========================================================

svm_cv = cross_val_score(

    svm_model,

    X_train_tfidf,

    y_train,

    cv=5,

    scoring="f1"

)

print(
    "\nSVM CV F1:",
    svm_cv.mean()
)

# =========================================================
# VOTING ENSEMBLE
# =========================================================

print("\n" + "=" * 60)
print("TRAINING VOTING ENSEMBLE")
print("=" * 60)

ensemble_model = VotingClassifier(

    estimators=[

        (
            "nb",
            nb_model
        ),

        (
            "lr",
            lr_model
        ),

        (
            "svm",
            svm_model
        )

    ],

    voting="hard"

)

ensemble_model.fit(

    X_train_tfidf,

    y_train

)

ensemble_pred = ensemble_model.predict(
    X_test_tfidf
)

evaluate_model(

    "Voting Ensemble",

    y_test,

    ensemble_pred

)

# =========================================================
# ENSEMBLE CV
# =========================================================

ensemble_cv = cross_val_score(

    ensemble_model,

    X_train_tfidf,

    y_train,

    cv=5,

    scoring="f1"

)

print(
    "\nEnsemble CV F1:",
    ensemble_cv.mean()
)

# =========================================================
# BEST TRADITIONAL MODEL
# =========================================================

traditional_models = {

    "Naive Bayes": f1_score(
        y_test,
        nb_pred
    ),

    "Logistic Regression": f1_score(
        y_test,
        lr_pred
    ),

    "Linear SVM": f1_score(
        y_test,
        svm_pred
    ),

    "Voting Ensemble": f1_score(
        y_test,
        ensemble_pred
    )

}

best_traditional = max(

    traditional_models,

    key=traditional_models.get

)

print("\n" + "=" * 60)

print(
    "BEST TRADITIONAL MODEL:",
    best_traditional
)

print("=" * 60)

# =========================================================
# SAVE TEMP RESULTS
# =========================================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC_AUC"

    ]

)

print("\nCurrent Results")

print(results_df)

# =========================================================
# LSTM PREPARATION
# =========================================================

print("\n" + "=" * 60)
print("PREPARING LSTM")
print("=" * 60)

MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    X_train
)

# =========================================================
# TEXT TO SEQUENCE
# =========================================================

X_train_seq = tokenizer.texts_to_sequences(
    X_train
)

X_val_seq = tokenizer.texts_to_sequences(
    X_val
)

X_test_seq = tokenizer.texts_to_sequences(
    X_test
)

# =========================================================
# PADDING
# =========================================================

X_train_pad = pad_sequences(

    X_train_seq,

    maxlen=MAX_LEN,

    padding="post",

    truncating="post"

)

X_test_pad = pad_sequences(

    X_test_seq,

    maxlen=MAX_LEN,

    padding="post",

    truncating="post"

)

X_val_pad = pad_sequences(

    X_val_seq,

    maxlen=MAX_LEN,

    padding="post",

    truncating="post"

)

print(
    "Train Shape:",
    X_train_pad.shape
)

print(
    "Validation Shape:",
    X_val_pad.shape
)

print(
    "Test Shape:",
    X_test_pad.shape
)

# =========================================================
# BUILD LSTM MODEL
# =========================================================

print("\nBuilding LSTM Model...")

lstm_model = Sequential([

    Embedding(

        input_dim=MAX_WORDS,

        output_dim=128,

        input_length=MAX_LEN

    ),

    LSTM(

        64,

        dropout=0.3

    ),

    Dense(

        32,

        activation="relu"

    ),

    Dropout(

        0.3

    ),

    Dense(

        1,

        activation="sigmoid"

    )

])

# =========================================================
# COMPILE
# =========================================================

lstm_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

print(
    lstm_model.summary()
)

# =========================================================
# EARLY STOPPING
# =========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=2,

    restore_best_weights=True

)

# =========================================================
# TRAIN LSTM
# =========================================================

print("\nTraining LSTM...")

history = lstm_model.fit(

    X_train_pad,

    y_train,

    validation_data=(X_val_pad, y_val),

    epochs=10,

    batch_size=64,

    callbacks=[early_stop],

    verbose=1

)

# =========================================================
# LSTM PREDICTION
# =========================================================

lstm_prob = lstm_model.predict(

    X_test_pad,

    verbose=0

)

lstm_pred = (
    lstm_prob > 0.5
).astype(int)

lstm_pred = lstm_pred.flatten()

# =========================================================
# LSTM EVALUATION
# =========================================================

evaluate_model(

    "LSTM",

    y_test,

    lstm_pred,

    lstm_prob.flatten()

)

# =========================================================
# FINAL RESULT TABLE
# =========================================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC_AUC"

    ]

)

results_df = results_df.sort_values(

    by="F1",

    ascending=False

)

print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df)

# =========================================================
# SAVE METRICS
# =========================================================

results_df.to_csv(

    "outputs/metrics.csv",

    index=False

)

# =========================================================
# BEST MODEL
# =========================================================

best_model_name = results_df.iloc[0]["Model"]

print("\nBest Model :", best_model_name)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nSaving confusion matrix...")

if best_model_name == "Naive Bayes":

    best_pred = nb_pred

elif best_model_name == "Logistic Regression":

    best_pred = lr_pred

elif best_model_name == "Linear SVM":

    best_pred = svm_pred

elif best_model_name == "Voting Ensemble":

    best_pred = ensemble_pred

else:

    best_pred = lstm_pred

cm = confusion_matrix(

    y_test,

    best_pred

)

plt.figure(

    figsize=(7,6)

)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "Real",
        "Hoax"
    ],

    yticklabels=[
        "Real",
        "Hoax"
    ]

)

plt.title(

    f"Confusion Matrix - {best_model_name}"

)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(

    "outputs/confusion_matrix.png",

    dpi=300

)

plt.close()

# =========================================================
# SAVE MODELS
# =========================================================

print("\nSaving models...")

joblib.dump(

    vectorizer,

    "models/tfidf_vectorizer.pkl"

)

joblib.dump(

    nb_model,

    "models/naive_bayes.pkl"

)

joblib.dump(

    lr_model,

    "models/logistic_regression.pkl"

)

joblib.dump(

    svm_model,

    "models/svm.pkl"

)

joblib.dump(

    ensemble_model,

    "models/ensemble.pkl"

)

joblib.dump(

    tokenizer,

    "models/tokenizer.pkl"

)

lstm_model.save(

    "models/lstm_model.h5"

)

# =========================================================
# SAVE TOP HOAX WORDS
# =========================================================

feature_names = vectorizer.get_feature_names_out()

top_hoax_idx = np.argsort(

    nb_model.feature_log_prob_[1]

)[-50:]

top_words = []

for idx in reversed(top_hoax_idx):

    top_words.append(
        feature_names[idx]
    )

pd.DataFrame({

    "Top_Hoax_Words": top_words

}).to_csv(

    "outputs/top_hoax_words.csv",

    index=False

)

# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")

print("models/")
print("- tfidf_vectorizer.pkl")
print("- naive_bayes.pkl")
print("- logistic_regression.pkl")
print("- svm.pkl")
print("- ensemble.pkl")
print("- tokenizer.pkl")
print("- lstm_model.h5")

print("\noutputs/")
print("- metrics.csv")
print("- confusion_matrix.png")
print("- top_hoax_words.csv")
