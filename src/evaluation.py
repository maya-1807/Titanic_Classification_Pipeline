"""Evaluation on labelled Titanic data: metrics + plots, reused by the Streamlit app.

Plot helpers return matplotlib `Figure` objects (they never call `plt.show`), so the
app can `st.pyplot(fig)` and the `__main__` runner below can `savefig` — one implementation,
identical output. `load_artifacts` / `predict_frame` cover the load-and-infer flow the app
also needs.

Run `python -m src.evaluation` to evaluate the held-out test split and save the plots.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from src.data import TARGET, fetch_raw_data, split
from src.model import TorchClassifier
from src.preprocess import TitanicPreprocessor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


# --- load & infer (shared with the app) --------------------------------------
def load_artifacts(
    models_dir: str | Path = MODELS_DIR,
    preprocessor_path: str | Path | None = None,
    model_path: str | Path | None = None,
):
    """Load the matched (preprocessor, classifier) pair written by train.py."""
    models_dir = Path(models_dir)
    preprocessor_path = preprocessor_path or models_dir / "preprocessor.joblib"
    model_path = model_path or models_dir / "model.pt"
    pp = TitanicPreprocessor.load(preprocessor_path)
    clf = TorchClassifier.load(model_path)
    return pp, clf


def predict_frame(pp: TitanicPreprocessor, clf: TorchClassifier, df: pd.DataFrame):
    """Transform `df` and return (y_true, y_prob); y_true is None if unlabelled."""
    X, y = pp.transform(df)
    return y, clf.predict_proba(X)


# --- metrics & plots ----------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Threshold class metrics at `threshold`; ROC-AUC uses the probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_prob) >= threshold
    roc_auc = np.nan
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc,
    }


def confusion_counts(y_true, y_prob, threshold: float = 0.5) -> dict[str, int]:
    """Return confusion-matrix cells with positive class = Survived."""
    y_pred = np.asarray(y_prob) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def prediction_table(df: pd.DataFrame, y_prob, threshold: float = 0.5) -> pd.DataFrame:
    """Passenger-level predictions for the Streamlit report table."""
    out = pd.DataFrame({
        "PassengerId": df["PassengerId"] if "PassengerId" in df else np.arange(1, len(df) + 1),
        "Passenger": df["Name"] if "Name" in df else "",
        "Class": df["Pclass"] if "Pclass" in df else np.nan,
        "Sex": df["Sex"] if "Sex" in df else "",
        "P(survived)": np.asarray(y_prob),
    })
    out["Predicted"] = np.where(out["P(survived)"] >= threshold, "Survived", "Died")
    if TARGET in df.columns:
        out["Actual"] = np.where(df[TARGET].to_numpy() == 1, "Survived", "Died")
        out["Correct"] = out["Predicted"].eq(out["Actual"])
    return out.sort_values("P(survived)", ascending=False).reset_index(drop=True)


def confusion_figure(y_true, y_prob, threshold: float = 0.5) -> Figure:
    """Confusion-matrix heatmap (Died / Survived)."""
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, np.asarray(y_prob) >= threshold, display_labels=["Died", "Survived"],
        cmap="Blues", colorbar=False)
    disp.ax_.set_title("Confusion matrix")
    return disp.figure_


def roc_figure(y_true, y_prob) -> Figure:
    """ROC curve with AUC in the legend."""
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC curve requires both classes in y_true.")
    disp = RocCurveDisplay.from_predictions(y_true, y_prob, name="MLP")
    disp.ax_.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)  # chance line
    disp.ax_.set_title("ROC curve")
    return disp.figure_


def probability_figure(y_prob, y_true=None, threshold: float = 0.5) -> Figure:
    """Histogram of predicted survival probabilities, optionally split by label."""
    import matplotlib.pyplot as plt

    y_prob = np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bins = np.linspace(0, 1, 11)
    if y_true is None:
        ax.hist(y_prob, bins=bins, color="#9c4c34", alpha=0.8, edgecolor="#fdfcfa")
    else:
        y_true = np.asarray(y_true)
        ax.hist(
            [y_prob[y_true == 0], y_prob[y_true == 1]],
            bins=bins,
            stacked=True,
            label=["Died", "Survived"],
            color=["#ddd6ca", "#9c4c34"],
            edgecolor="#fdfcfa",
        )
        ax.legend(frameon=False)
    ax.axvline(threshold, color="#26221c", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability of survival")
    ax.set_ylabel("Passengers")
    ax.set_title("Predicted probability")
    fig.tight_layout()
    return fig


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
