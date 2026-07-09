"""Titanic preprocessing: feature engineering, imputation, encoding and scaling.

`TitanicPreprocessor` implements the decisions from the EDA (Section 7 of
`notebooks/eda.ipynb`) as a single, picklable object shared by `train.py` and the
Streamlit app, so training and inference apply identical transforms.

Run `python -m src.preprocess` to fit on the train split and print a report.
"""
from __future__ import annotations

import re
from collections import namedtuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import TARGET, fetch_raw_data, split


# =============================================================================
# Configuration & constants
# =============================================================================
# Raw columns every input frame must contain (Survived/PassengerId are optional).
RAW_REQUIRED = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare",
                "Cabin", "Embarked"]

# Titles to extract from Name (longest first so "Mrs" matches before "Mr"), Dr is resolved by Sex below.
_TITLES = ["Mrs", "Mr", "Master", "Miss", "Major", "Rev", "Dr", "Ms", "Mlle", "Col",
           "Capt", "Mme", "Countess", "Don", "Jonkheer", "Lady", "Sir"]
_TITLE_PATTERN = "|".join(map(re.escape, sorted(_TITLES, key=len, reverse=True)))
_TITLE_MAP = {
    "Don": "Mr", "Major": "Mr", "Capt": "Mr", "Jonkheer": "Mr", "Rev": "Mr", "Col": "Mr",
    "Sir": "Mr", "Countess": "Mrs", "Mme": "Mrs", "Lady": "Mrs", "Mlle": "Miss", "Ms": "Miss",
}


# =============================================================================
# Feature registry
# =============================================================================
# A model feature: its name, encoding role ("onehot" | "scale" | "pass"), and a
# builder(pp, base) -> Series that reads from the imputed `base` frame.
Feat = namedtuple("Feat", "name role build")


def _family_bin(pp, b):
    return pd.cut(b.FamilySize, [0, 1, pp.family_boundary_, np.inf],
                  labels=["Single", "SmallFamily", "LargeFamily"])


def _deck(pp, b):
    return b.Cabin.str[0].fillna("Unknown")


# The enabled subset of this list defines the model's feature set.
FEATURES = [
    Feat("Sex",             "onehot", lambda pp, b: b.Sex),
    Feat("Pclass",          "onehot", lambda pp, b: b.Pclass),
    Feat("Embarked",        "onehot", lambda pp, b: b.Embarked),
    Feat("Title",           "onehot", lambda pp, b: b.Title),
    Feat("FamilyBin",       "onehot", _family_bin),
    Feat("Deck",            "onehot", _deck),
    Feat("Age",             "scale",  lambda pp, b: b.Age),
    Feat("FareLog",         "scale",  lambda pp, b: np.log1p(b.Fare)),
    Feat("FamilySize",      "scale",  lambda pp, b: b.FamilySize),
    Feat("FarePerPerson",   "scale",  lambda pp, b: b.Fare / b.TicketFrequency),
    Feat("TicketFrequency", "scale",  lambda pp, b: b.TicketFrequency),
    Feat("IsAlone",         "pass",   lambda pp, b: (b.FamilySize == 1).astype(int)),
    Feat("CabinKnown",      "pass",   lambda pp, b: b.Cabin.notna().astype(int)),
    Feat("AgeMissing",      "pass",   lambda pp, b: b.AgeMissing),
]
_BY_NAME = {f.name: f for f in FEATURES}


# =============================================================================
# Preprocessor
# =============================================================================
class TitanicPreprocessor:
    """Fit imputation/encoding on train data. transform any Titanic frame to a
    numeric feature matrix ready for PyTorch.

    include/exclude select which engineered features are used (for ablation);
    by default all of `FEATURES` are enabled.
    """

    # --- construction & feature selection ------------------------------------
    def __init__(self, include: list[str] | None = None, exclude: list[str] | None = None):
        names = list(_BY_NAME) if include is None else include
        exclude = set(exclude or [])
        unknown = (set(names) | exclude) - set(_BY_NAME)
        if unknown:
            raise ValueError(f"Unknown feature(s): {sorted(unknown)}")
        self.feature_order_ = [n for n in names if n not in exclude]

    @property
    def features(self) -> list:
        """Enabled feature specs, resolved from the registry. Kept out of the
        pickled state so save/load never needs to pickle builder functions."""
        return [_BY_NAME[n] for n in self.feature_order_]

    @property
    def n_features(self) -> int:
        """Number of encoded output columns (after one-hot expansion)."""
        return len(self.feature_names_)

    # --- public API: fit / transform -----------------------------------------
    def fit(self, df: pd.DataFrame) -> "TitanicPreprocessor":
        """Learn imputation statistics and the encoder from a training frame."""
        self._validate(df)
        # Imputation statistics (train only).
        self.fare_medians_ = df.groupby(["Pclass", "Embarked"]).Fare.median()
        self.fare_global_ = df.Fare.median()
        titled = self._add_title(df)
        self.age_medians_ = titled.groupby(["Title", "Sex", "Pclass"]).Age.median()
        self.age_global_ = df.Age.median()
        # FamilyBin Small/Large boundary: 75th percentile of non-singles (data-driven).
        fs = df.SibSp + df.Parch + 1
        self.family_boundary_ = fs[fs >= 2].quantile(0.75)

        # Encoder over the enabled features, grouped by role (skip empty roles).
        feat_df = self._build_features(df)
        encoders = [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scale", StandardScaler()), ("pass", "passthrough")]
        transformers = [(role, enc, cols) for role, enc in encoders
                        if (cols := [f.name for f in self.features if f.role == role])]
        self.encoder_ = ColumnTransformer(transformers)
        self.encoder_.fit(feat_df)
        self.feature_names_ = list(self.encoder_.get_feature_names_out())
        return self

    def transform(self, df: pd.DataFrame):
        """Return (X, y): X is float32 [n, n_features]; y is float32, or None when
        the `Survived` column is absent."""
        self._validate(df)
        X = self.encoder_.transform(self._build_features(df)).astype(np.float32)
        y = df[TARGET].to_numpy(np.float32) if TARGET in df.columns else None
        return X, y

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TitanicPreprocessor":
        return joblib.load(path)

    # --- feature construction ------------------------------------------------
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every enabled builder on the imputed base frame."""
        base = self._base(df)
        return pd.DataFrame({f.name: f.build(self, base) for f in self.features})

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputed base frame + shared intermediates (imputation always applied).
        Order matters: Title before Age, Embarked before Fare."""
        b = self._add_title(df)
        b["Embarked"] = b.apply(self._impute_embarked, axis=1)
        b["AgeMissing"] = b.Age.isna().astype(int)
        b["Age"] = b.apply(self._impute_age, axis=1)
        b["Fare"] = b.apply(self._impute_fare, axis=1)
        b["FamilySize"] = b.SibSp + b.Parch + 1
        b["TicketFrequency"] = b.groupby("Ticket").Ticket.transform("size")
        return b

    @staticmethod
    def _add_title(df: pd.DataFrame) -> pd.DataFrame:
        """Extract Title from Name and fold it into Mr / Mrs / Miss / Master.
        Dr resolves by Sex; any unknown/unmatched title falls back to 'Rare'."""
        raw = df.Name.str.extract(r",\s*(" + _TITLE_PATTERN + r")\.", expand=False)
        title = raw.replace(_TITLE_MAP)
        sex = df.Sex.str.lower()
        title = title.mask((raw == "Dr") & (sex == "male"), "Mr")
        title = title.mask((raw == "Dr") & (sex == "female"), "Mrs")
        title = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
        return df.assign(Title=title)

    # --- imputation (per-row; uses learned statistics) -----------------------
    def _impute_embarked(self, row):
        """Missing Embarked -> the same-class port whose median Fare is closest."""
        if pd.notna(row.Embarked):
            return row.Embarked
        return (self.fare_medians_[row.Pclass] - row.Fare).abs().idxmin()

    def _impute_fare(self, row):
        """Missing Fare -> median for its (Pclass, Embarked), else global median."""
        if pd.notna(row.Fare):
            return row.Fare
        return self.fare_medians_.get((row.Pclass, row.Embarked), self.fare_global_)

    def _impute_age(self, row):
        """Missing Age -> median for its (Title, Sex, Pclass), else global median."""
        if pd.notna(row.Age):
            return row.Age
        return self.age_medians_.get((row.Title, row.Sex, row.Pclass), self.age_global_)

    # --- validation ----------------------------------------------------------
    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in RAW_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")


# =============================================================================
# CLI self-check / report
# =============================================================================
def _self_check() -> None:
    """Fit on train, transform every split, and print a readable report."""
    tr, te = split(fetch_raw_data())
    pp = TitanicPreprocessor().fit(tr)

    def cols(role):
        return [f.name for f in pp.features if f.role == role]

    bar = "=" * 60
    print(f"{bar}\nTitanicPreprocessor — fit on train, transform all splits\n{bar}")
    print(f"Raw columns ({len(tr.columns)})  ->  {len(pp.features)} features  ->  "
          f"{pp.n_features} encoded columns\n")

    print(f"{'split':<6}{'rows':>6}{'columns':>9}{'survival':>10}{'NaNs':>7}")
    for name, d in [("train", tr), ("test", te)]:
        X, y = pp.transform(d)
        print(f"{name:<6}{len(d):>6}{X.shape[1]:>9}{y.mean():>10.3f}{int(np.isnan(X).sum()):>7}")

    print("\nImputation (missing in raw train -> filled; always on):")
    for col in ["Age", "Embarked", "Fare", "Cabin"]:
        print(f"  {col:<9}{int(tr[col].isna().sum()):>4} missing")
    print(f"  learned: FamilyBin cut={pp.family_boundary_:.0f}, "
          f"Age fallback={pp.age_global_:.1f}, Fare fallback={pp.fare_global_:.2f}")

    print("\nFeature roles (enabled):")
    print(f"  one-hot     : {', '.join(cols('onehot'))}")
    print(f"  scaled      : {', '.join(cols('scale'))}")
    print(f"  passthrough : {', '.join(cols('pass'))}")

    scaler = pp.encoder_.named_transformers_["scale"]
    print("\nScaling learned (mean / std):")
    for col, m, s in zip(cols("scale"), scaler.mean_, scaler.scale_):
        print(f"  {col:<16}mean={m:>8.3f}   std={s:>8.3f}")
    print(bar)


if __name__ == "__main__":
    _self_check()
