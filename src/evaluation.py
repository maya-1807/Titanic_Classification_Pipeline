"""Evaluation on labelled Titanic data: metrics + plots, reused by the Streamlit app.

Plot helpers return matplotlib `Figure` objects (they never call `plt.show`), so the
app can `st.pyplot(fig)` and the `__main__` runner below can `savefig` — one implementation,
identical output. `load_artifacts` / `predict_frame` cover the load-and-infer flow the app
also needs.

Run `python -m src.evaluation` to evaluate the held-out test split and save the plots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.data import TARGET, fetch_raw_data, split
from src.model import TorchClassifier
from src.preprocess import TitanicPreprocessor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


# --- load & infer (shared with the app) --------------------------------------
def load_artifacts(models_dir: str | Path = MODELS_DIR):
    """Load the matched (preprocessor, classifier) pair written by train.py."""
    models_dir = Path(models_dir)
    pp = TitanicPreprocessor.load(models_dir / "preprocessor.joblib")
    clf = TorchClassifier.load(models_dir / "model.pt")
    return pp, clf


def predict_frame(pp: TitanicPreprocessor, clf: TorchClassifier, df: pd.DataFrame):
    """Transform `df` and return (y_true, y_prob); y_true is None if unlabelled."""
    X, y = pp.transform(df)
    return y, clf.predict_proba(X)


# --- metrics & plots ----------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Threshold class metrics at `threshold`; ROC-AUC uses the probabilities."""
    y_pred = np.asarray(y_prob) >= threshold
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def confusion_figure(y_true, y_prob, threshold: float = 0.5) -> Figure:
    """Confusion-matrix heatmap (Died / Survived)."""
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, np.asarray(y_prob) >= threshold, display_labels=["Died", "Survived"],
        cmap="Blues", colorbar=False)
    disp.ax_.set_title("Confusion matrix")
    return disp.figure_


def roc_figure(y_true, y_prob) -> Figure:
    """ROC curve with AUC in the legend."""
    disp = RocCurveDisplay.from_predictions(y_true, y_prob, name="MLP")
    disp.ax_.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)  # chance line
    disp.ax_.set_title("ROC curve")
    return disp.figure_


def _main() -> None:
    pp, clf = load_artifacts()
    test = split(fetch_raw_data())[1]  # the sealed held-out split
    y_true, y_prob = predict_frame(pp, clf, test)
    if y_true is None:
        raise SystemExit("Test data has no 'Survived' column — cannot evaluate.")

    print(f"Held-out test: {len(test)} rows")
    for name, val in compute_metrics(y_true, y_prob).items():
        print(f"  {name:<10} {val:.4f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    for fig, fname in [(confusion_figure(y_true, y_prob), "confusion_matrix.png"),
                       (roc_figure(y_true, y_prob), "roc_curve.png")]:
        path = REPORTS_DIR / fname
        fig.savefig(path, bbox_inches="tight", dpi=120)
        print(f"Saved {path}")


if __name__ == "__main__":
    _main()
