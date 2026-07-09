"""Model / hyperparameter / feature-ablation search via stratified k-fold CV.

Throwaway harness (not a shipped deliverable) to pick a model + config before
writing `train.py`. Selection runs on train+val; the test split stays sealed.

Everything is driven by three registries — extend by appending one line:
  * MODELS        -- (name, build(n_features) -> estimator with .fit/.predict_proba)
  * FEATURE_SETS  -- (name, kwargs for TitanicPreprocessor)
  * METRICS       -- name -> fn(y_true, y_prob)

Run:  python -m experiments.model_search
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from src.data import SEED, TARGET, fetch_raw_data, split
from src.model import TorchClassifier
from src.preprocess import TitanicPreprocessor

N_FOLDS = 5
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"


# --- registries (the extension points) ---------------------------------------
# Add a model: append (name, build). build(n) -> anything with .fit/.predict_proba.
MODELS = [
    # sklearn references (not deliverable candidates)
    ("logreg",           lambda n: LogisticRegression(max_iter=1000)),
    ("rf",               lambda n: RandomForestClassifier(n_estimators=300, random_state=SEED)),
    # width / depth
    ("mlp[32]",          lambda n: TorchClassifier(n, [32])),
    ("mlp[64]",          lambda n: TorchClassifier(n, [64])),
    ("mlp[32,16]",       lambda n: TorchClassifier(n, [32, 16])),
    ("mlp[64,32]",       lambda n: TorchClassifier(n, [64, 32])),
    ("mlp[64,32,16]",    lambda n: TorchClassifier(n, [64, 32, 16])),
    # regularization
    ("mlp[64,32] d.2",   lambda n: TorchClassifier(n, [64, 32], dropout=0.2)),
    ("mlp[64,32] d.5",   lambda n: TorchClassifier(n, [64, 32], dropout=0.5)),
    ("mlp[64,32] bn",    lambda n: TorchClassifier(n, [64, 32], batchnorm=True)),
    # optimization
    ("mlp[64,32] lr5e4", lambda n: TorchClassifier(n, [64, 32], lr=5e-4)),
    ("mlp[64,32] lr3e3", lambda n: TorchClassifier(n, [64, 32], lr=3e-3)),
    ("mlp[64,32] e200",  lambda n: TorchClassifier(n, [64, 32], epochs=200)),
]

# Add an ablation: append (name, kwargs) passed to TitanicPreprocessor(**kwargs).
FEATURE_SETS = [
    ("all",     {}),
    ("no_deck", {"exclude": ["Deck"]}),
    ("minimal", {"include": ["Sex", "Pclass", "Age", "Title", "FareLog"]}),
]

# Add a metric: name -> fn(y_true, y_prob). Class metrics threshold probs at 0.5.
METRICS = {
    "accuracy":  lambda y, p: accuracy_score(y, p >= 0.5),
    "precision": lambda y, p: precision_score(y, p >= 0.5),
    "recall":    lambda y, p: recall_score(y, p >= 0.5),
    "f1":        lambda y, p: f1_score(y, p >= 0.5),
    "roc_auc":   lambda y, p: roc_auc_score(y, p),
}


def cross_validate(dev: pd.DataFrame, build, pp_kwargs: dict) -> dict:
    """Mean CV score per metric for one (model, feature_set), fitting the
    preprocessor inside each fold to avoid leakage."""
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    folds = {m: [] for m in METRICS}
    for tr_idx, va_idx in skf.split(dev, dev[TARGET]):
        raw_tr, raw_va = dev.iloc[tr_idx], dev.iloc[va_idx]
        pp = TitanicPreprocessor(**pp_kwargs).fit(raw_tr)
        Xtr, ytr = pp.transform(raw_tr)
        Xva, yva = pp.transform(raw_va)
        est = build(pp.n_features).fit(Xtr, ytr)
        prob = np.asarray(est.predict_proba(Xva))
        prob = prob[:, 1] if prob.ndim == 2 else prob  # sklearn returns 2 cols
        for name, fn in METRICS.items():
            folds[name].append(fn(yva, prob))
    return {m: (np.mean(v), np.std(v)) for m, v in folds.items()}


def main() -> None:
    tr, va, te = split(fetch_raw_data())
    dev = pd.concat([tr, va], ignore_index=True)  # test (te) stays sealed
    print(f"Dev set: {len(dev)} rows, {N_FOLDS}-fold CV "
          f"({len(MODELS)} models x {len(FEATURE_SETS)} feature sets)\n")

    rows = []
    for m_name, build in MODELS:
        for fs_name, kwargs in FEATURE_SETS:
            scores = cross_validate(dev, build, kwargs)
            rows.append({"model": m_name, "features": fs_name,
                         **{m: mean for m, (mean, _) in scores.items()},
                         "roc_auc_std": scores["roc_auc"][1]})

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200, "display.max_rows", None)
    print(df.round(4).to_string())
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {RESULTS_CSV.relative_to(RESULTS_CSV.parents[1])}")
    best = df.iloc[0]
    print(f"Top by ROC-AUC: {best.model} / {best.features}  "
          f"(auc={best.roc_auc:.4f}, acc={best.accuracy:.4f}, f1={best.f1:.4f})")


if __name__ == "__main__":
    main()
