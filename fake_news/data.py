"""Dataset preparation, leakage audit, and deterministic holdout splits."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fake_news.config import DATA_DIR, RANDOM_STATE
from fake_news.preprocessing import clean_text

REQUIRED_COLUMNS = ("title", "text", "subject", "date")


def _validate_columns(frame: pd.DataFrame, source: str) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def load_raw_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for filename, label in (("Fake.csv", 1), ("True.csv", 0)):
        path = data_dir / filename
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError(f"Could not read {path}: {exc}") from exc
        _validate_columns(frame, filename)
        if frame.empty:
            raise ValueError(f"{filename} contains no articles")
        frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        frame["label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_dataset(
    raw: pd.DataFrame, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, dict]:
    """Normalize articles and remove ambiguous or duplicate texts before splitting."""
    _validate_columns(raw, "Dataset")
    if "label" not in raw.columns:
        raise ValueError("Dataset is missing required column: label")
    prepared = raw.copy()
    prepared["article"] = (
        prepared["title"].fillna("").astype(str)
        + " "
        + prepared["text"].fillna("").astype(str)
    )
    prepared["clean_text"] = prepared["article"].map(clean_text)

    conflicts = prepared.groupby("clean_text")["label"].transform("nunique").gt(1)
    empty = prepared["clean_text"].eq("")
    usable = prepared.loc[~conflicts & ~empty]
    unique = usable.drop_duplicates(subset="clean_text", keep="first")
    if unique["label"].nunique() < 2:
        raise ValueError("Dataset needs usable articles from both labels")
    unique = unique.sample(frac=1, random_state=seed).reset_index(drop=True)

    marker = raw["text"].fillna("").str.contains(r"\breuters\b", case=False, regex=True)
    subject_counts = pd.crosstab(raw["subject"].fillna("<missing>"), raw["label"])
    subject_counts = subject_counts.reindex(columns=[0, 1], fill_value=0)
    parsed_dates = pd.to_datetime(raw["date"], errors="coerce", format="mixed")
    audit = {
        "raw_rows": int(len(raw)),
        "raw_label_counts": _label_counts(raw),
        "blank_title_rows": _blank_count(raw["title"]),
        "blank_text_rows": _blank_count(raw["text"]),
        "blank_subject_rows": _blank_count(raw["subject"]),
        "blank_date_rows": _blank_count(raw["date"]),
        "blank_article_rows": int(empty.sum()),
        "conflicting_label_rows": int(conflicts.sum()),
        "duplicate_clean_text_rows": int(len(usable) - len(unique)),
        "unique_rows": int(len(unique)),
        "unique_label_counts": _label_counts(unique),
        "reuters_marker_rate": {
            str(label): float(marker[raw["label"].eq(label)].mean()) for label in (0, 1)
        },
        "subject_by_label": [
            {"subject": str(subject), "real": int(counts[0]), "fake": int(counts[1])}
            for subject, counts in subject_counts.iterrows()
        ],
        "unparsed_date_rows": int(parsed_dates.isna().sum()),
        "date_min": str(parsed_dates.min().date())
        if parsed_dates.notna().any()
        else None,
        "date_max": str(parsed_dates.max().date())
        if parsed_dates.notna().any()
        else None,
    }
    return unique[["clean_text", "label"]], audit


def load_dataset(
    data_dir: Path = DATA_DIR, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, dict]:
    return prepare_dataset(load_raw_dataset(data_dir), seed)


def split_dataset(dataset: pd.DataFrame, seed: int = RANDOM_STATE):
    """Reserve 70% train, 15% validation, and 15% final test."""
    text = dataset["clean_text"]
    labels = dataset["label"]
    train_x, remaining_x, train_y, remaining_y = train_test_split(
        text, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_x, test_x, val_y, test_y = train_test_split(
        remaining_x,
        remaining_y,
        test_size=0.50,
        random_state=seed,
        stratify=remaining_y,
    )
    return train_x, val_x, test_x, train_y, val_y, test_y


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def _blank_count(values: pd.Series) -> int:
    return int(values.fillna("").astype(str).str.strip().eq("").sum())
