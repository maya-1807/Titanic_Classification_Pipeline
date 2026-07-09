"""Random search over MLP hyperparameters x feature sets, via stratified k-fold CV.

Throwaway harness (not a shipped deliverable) to pick an MLP config before writing
`train.py`. The deliverable is a PyTorch model, so the search estimator IS the MLP
(logreg/RF appear only as reference yardsticks, not candidates).

Why random search, not a full grid: with ~750 rows the CV score wobbles by ~0.03
AUC, so evaluating thousands of grid points and taking the max mostly selects
noise. A fixed budget of random draws over the axes that matter finds the good
region cheaply, and we pick with the one-standard-error rule (simplest model within
1 std of the best) rather than chasing the single highest — a noisy — score.

`epochs` is absent on purpose: TorchClassifier early-stops, so run length is learned.

Run:  python -m experiments.model_search [n_configs]
"""
from __future__ import annotations

import random
import sys
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
N_CONFIGS = 50
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"


# --- search space (the extension points) --------------------------------------
# Named feature sets: kwargs passed to TitanicPreprocessor (feature ablation axis).
# Chosen from the correlation structure (see experiments/feature_search.py):
#   Sex~Title r.89 · FamilyBin~FamilySize~IsAlone r.98 · FareLog~FarePerPerson~Pclass r.7+
#   CabinKnown>Deck for signal · Age has near-zero univariate signal.
FEATURE_SETS = {
    "all":          {},
    "no_deck":      {"exclude": ["Deck"]},                       # Deck weak, ~ CabinKnown
    "no_age":       {"exclude": ["Age", "AgeMissing"]},          # tests Age's ~0 signal
    "core":         {"include": ["Sex", "Pclass", "Age", "Title", "FareLog", "FamilyBin", "Embarked"]},
    "no_redundant": {"include": ["Title", "Pclass", "Embarked", "IsAlone", "Age", "FareLog", "CabinKnown", "AgeMissing"]},
    "decorrelated": {"include": ["Title", "Pclass", "FareLog", "IsAlone", "CabinKnown", "Age", "Embarked"]},
    "top_signal":   {"include": ["Sex", "Title", "FareLog", "Pclass", "FarePerPerson", "CabinKnown"]},
    "minimal":      {"include": ["Sex", "Pclass", "Age", "Title", "FareLog"]},
    "strong":       {"include": ["Title", "Pclass", "FareLog"]},  # only the top-signal few
}

# Hyperparameter axes sampled per draw. activation/optimizer carry a low-impact
# alternative on purpose, so the results show whether they matter (they shouldn't).
SEARCH_SPACE = {
    "features":     list(FEATURE_SETS),
    "hidden_dims":  [(32,), (64,), (64, 32), (128, 64), (64, 32, 16)],
    "dropout":      [0.0, 0.2, 0.3, 0.5],
    "batchnorm":    [False, True],
    "activation":   ["relu"],
    "lr":           [3e-4, 5e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "optimizer":    ["adam"],
}
MODEL_KEYS = [k for k in SEARCH_SPACE if k != "features"]

# Reference yardsticks (not candidates): run once per feature set for comparison.
REFERENCES = {
    "logreg": lambda n: LogisticRegression(max_iter=1000),
    "rf":     lambda n: RandomForestClassifier(n_estimators=300, random_state=SEED),
}

METRICS = {
    "accuracy":  lambda y, p: accuracy_score(y, p >= 0.5),
    "precision": lambda y, p: precision_score(y, p >= 0.5),
    "recall":    lambda y, p: recall_score(y, p >= 0.5),
    "f1":        lambda y, p: f1_score(y, p >= 0.5),
    "roc_auc":   lambda y, p: roc_auc_score(y, p),
}


def sample_configs(n: int, seed: int) -> list[dict]:
    """n distinct random draws from SEARCH_SPACE (seeded, deduplicated)."""
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        key = tuple(cfg.items())
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def cross_validate(dev: pd.DataFrame, pp_kwargs: dict, build) -> dict:
    """Mean/std CV score per metric for one (feature set, model), refitting the
    preprocessor inside each fold to avoid leakage. build(n_features) -> estimator."""
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


def _row(model: str, features: str, scores: dict, **cfg) -> dict:
    means = {m: mean for m, (mean, _) in scores.items()}
    return {"model": model, "features": features, **cfg, **means,
            "roc_auc_std": scores["roc_auc"][1]}


def main() -> None:
    n_configs = int(sys.argv[1]) if len(sys.argv) > 1 else N_CONFIGS
    tr, va, te = split(fetch_raw_data())
    dev = pd.concat([tr, va], ignore_index=True)  # test (te) stays sealed
    configs = sample_configs(n_configs, SEED)
    print(f"Dev set: {len(dev)} rows, {N_FOLDS}-fold CV\n"
          f"Random search: {len(configs)} MLP configs + {len(REFERENCES)} references "
          f"x {len(FEATURE_SETS)} feature sets\n")

    rows = []
    for i, cfg in enumerate(configs, 1):
        model_cfg = {k: cfg[k] for k in MODEL_KEYS}
        build = lambda n, c=model_cfg: TorchClassifier(n, **c)
        scores = cross_validate(dev, FEATURE_SETS[cfg["features"]], build)
        rows.append(_row("mlp", cfg["features"], scores, **model_cfg))
        print(f"  [{i:>2}/{len(configs)}] auc={scores['roc_auc'][0]:.4f} "
              f"+-{scores['roc_auc'][1]:.3f}  {cfg['features']:<8} {cfg['hidden_dims']}")

    # Reference baselines across every feature set (yardsticks, not candidates).
    for ref_name, ref_build in REFERENCES.items():
        for fs_name, kwargs in FEATURE_SETS.items():
            scores = cross_validate(dev, kwargs, ref_build)
            rows.append(_row(ref_name, fs_name, scores))

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df.to_csv(RESULTS_CSV, index=False)

    pd.set_option("display.width", 220, "display.max_rows", None)
    print("\n" + df.round(4).to_string())
    print(f"\nSaved {RESULTS_CSV.relative_to(RESULTS_CSV.parents[1])}")
    _report_pick(df)


def _report_pick(df: pd.DataFrame) -> None:
    """Top MLP by AUC, plus the one-standard-error pick: the simplest MLP whose
    mean AUC is within 1 std of the best (favours regularization over raw score)."""
    mlp = df[df.model == "mlp"].copy()
    best = mlp.iloc[0]
    within = mlp[mlp.roc_auc >= best.roc_auc - best.roc_auc_std].copy()
    within["capacity"] = within.hidden_dims.map(sum)
    onese = within.sort_values(["capacity", "dropout"], ascending=[True, False]).iloc[0]

    def line(r):
        return (f"{r.features} / {r.hidden_dims} do={r.dropout} bn={r.batchnorm} "
                f"{r.activation} {r.optimizer} lr={r.lr} wd={r.weight_decay}\n"
                f"    auc={r.roc_auc:.4f}+-{r.roc_auc_std:.3f}  acc={r.accuracy:.4f}  f1={r.f1:.4f}")

    print(f"\nTop by ROC-AUC:\n  {line(best)}")
    print(f"\nOne-standard-error pick (simplest within 1 std of best):\n  {line(onese)}")


if __name__ == "__main__":
    main()
