"""Standalone training script: fetch -> preprocess -> train -> save artifacts.

Trains the configuration chosen by the search harness (experiments/model_search.py):
the one-standard-error pick — `core` features + a single 32-unit MLP — on the train
split only, leaving the test split sealed for later evaluation. Saves the fitted
preprocessor and model weights to models/ for the Streamlit app to load.

Run:  python train.py
"""
from __future__ import annotations

from pathlib import Path

from src.data import fetch_raw_data, split
from src.model import TorchClassifier
from src.preprocess import TitanicPreprocessor

MODELS_DIR = Path(__file__).resolve().parent / "models"

CORE_FEATURES = ["Sex", "Pclass", "Age", "Title", "FareLog", "FamilyBin", "Embarked"]
MODEL_CFG = dict(hidden_dims=(32,), dropout=0.3, batchnorm=True, activation="relu",
                 optimizer="adam", lr=5e-4, weight_decay=1e-5)


def main() -> None:
    train, _ = split(fetch_raw_data())
    pp = TitanicPreprocessor(include=CORE_FEATURES).fit(train)
    X, y = pp.transform(train)
    clf = TorchClassifier(pp.n_features, **MODEL_CFG).fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    pp_path = MODELS_DIR / "preprocessor.joblib"
    model_path = MODELS_DIR / "model.pt"
    pp.save(pp_path)
    clf.save(model_path)

    print(f"Trained on {len(train)} rows -> {pp.n_features} features")
    print(f"Config: {CORE_FEATURES}\n        {MODEL_CFG}")
    print(f"Saved preprocessor -> {pp_path}")
    print(f"Saved model        -> {model_path}")


if __name__ == "__main__":
    main()
