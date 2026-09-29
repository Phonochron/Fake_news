"""Train and evaluate the LSTM classifier."""

import numpy as np
from sklearn.model_selection import train_test_split

from fake_news.config import MAX_LEN, MAX_WORDS, TrainingConfig
from fake_news.evaluation import evaluate_model
from fake_news.sequences import encode_sequences


def train_lstm(train_x, train_y, config: TrainingConfig):
    # The CLI preloads TensorFlow before this lazy Keras import on Windows.
    from keras import Input
    from keras.callbacks import EarlyStopping
    from keras.layers import LSTM, Dense, Dropout, Embedding
    from keras.models import Sequential
    from tensorflow.keras.preprocessing.text import Tokenizer

    fit_x, stop_x, fit_y, stop_y = train_test_split(
        train_x,
        train_y,
        test_size=0.10,
        random_state=config.seed,
        stratify=train_y,
    )
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(fit_x)
    model = Sequential(
        [
            Input(shape=(MAX_LEN,)),
            Embedding(input_dim=MAX_WORDS, output_dim=config.lstm_embedding_dim),
            LSTM(config.lstm_units, dropout=config.lstm_dropout),
            Dense(config.lstm_dense_units, activation="relu"),
            Dropout(config.lstm_dropout),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        encode_sequences(tokenizer, fit_x),
        np.asarray(fit_y, dtype=np.int32),
        validation_data=(
            encode_sequences(tokenizer, stop_x),
            np.asarray(stop_y, dtype=np.int32),
        ),
        epochs=config.lstm_epochs,
        batch_size=config.lstm_batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        ],
        verbose=2,
    )
    return tokenizer, model


def evaluate_lstm(model, tokenizer, text, labels):
    probability = model.predict(encode_sequences(tokenizer, text), verbose=0).ravel()
    predicted = (probability > 0.5).astype(int)
    return evaluate_model("LSTM", labels, predicted, probability), predicted
