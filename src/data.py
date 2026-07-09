"""Fetch the Titanic dataset and split it into train/val/test.

Kaggle's train.csv (891 labeled rows) is the full dataset; its test.csv is
unlabeled, so we split train.csv ourselves, stratified on Survived.
Fetch from Kaggle via kagglehub.

Run `python -m src.data` to cache the data, write the splits, and the sample CSV.
"""
from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_DIR / "raw" / "train.csv"
TARGET = "Survived"
SEED = 42


def fetch_raw_data(force: bool = False) -> pd.DataFrame:
    """Return the full labeled dataset, cached to data/raw/train.csv.
    force=True re-downloads even if cached."""

    # If data/raw/train.csv is already cached, read it
    if RAW.exists() and not force:
        return pd.read_csv(RAW)
    
    # Otherwise, fetch it, cache it to disk, and return it
    df = _from_kaggle()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)
    return df


def _from_kaggle() -> pd.DataFrame:
    """Download train.csv directly from Kaggle via kagglehub.

    Needs a Kaggle API token (~/.kaggle/access_token). Reviewers normally use
    the committed data/raw/train.csv cache and never reach this path.
    """
    try:
        path = kagglehub.competition_download("titanic")
        return pd.read_csv(Path(path) / "train.csv")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not fetch from Kaggle and no cached data/raw/train.csv found. "
            "Set up a Kaggle API token or restore the committed CSV."
        ) from exc


def split(df: pd.DataFrame, val: float = 0.15, test: float = 0.15, seed: int = SEED):
    """Stratified train/val/test split (deterministic for a given seed)."""
    train_val, test_df = train_test_split(df, test_size=test, stratify=df[TARGET], random_state=seed)
    train_df, val_df = train_test_split(
        train_val, test_size=val / (1 - test), stratify=train_val[TARGET], random_state=seed
    )
    return train_df, val_df, test_df


def main() -> None:
    df = fetch_raw_data()
    parts = dict(zip(("train", "val", "test"), split(df)))
    for name, part in parts.items():
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"[data] {name:>5}: {len(part):>3} rows, survival rate {part[TARGET].mean():.3f}")
    # Small labeled sample committed to the repo for the app demo.
    parts["test"].sample(40, random_state=SEED).to_csv(DATA_DIR / "sample_test.csv", index=False)
    print(f"[data] Wrote data/sample_test.csv")


if __name__ == "__main__":
    main()
