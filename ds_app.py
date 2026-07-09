"""Streamlit inference app for the Titanic PyTorch classifier.

Run:
    streamlit run ds_app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data import TARGET
from src.evaluation import (compute_metrics, confusion_counts, confusion_figure,
                            load_artifacts, predict_frame, prediction_table,
                            probability_figure, roc_figure)


def _fmt_pct(value: float) -> str:
    return "--" if pd.isna(value) else f"{value * 100:.1f}%"


def _fmt_num(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


def _model_summary(pp, clf) -> str:
    dims = ", ".join(str(d) for d in clf.hidden_dims)
    return (
        f"Model loaded - MLP [{dims}], dropout {clf.dropout}, "
        f"{pp.n_features} encoded features, device {clf.device}."
    )


def _init_state() -> None:
    st.session_state.setdefault("step", 1)
    st.session_state.setdefault("dataset_source", None)
    st.session_state.setdefault("model_source", None)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("pp", None)
    st.session_state.setdefault("clf", None)
    st.session_state.setdefault("y_true", None)
    st.session_state.setdefault("y_prob", None)
    st.session_state.setdefault("error", None)


def _style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #fdfcfa;
            --page: #eae8e3;
            --ink: #26221c;
            --muted: #8a8378;
            --line: #e7e2d8;
            --accent: #9c4c34;
            --ok: #3f7a55;
        }
        .stApp { background: var(--page); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { display: none; }
        .block-container {
            max-width: 900px;
            padding: 0 32px 30px;
            margin-top: 38px;
            margin-bottom: 44px;
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 13px;
            color: var(--ink);
            box-shadow: 0 18px 45px rgba(38, 34, 28, 0.06);
        }
        .report-top { padding: 24px 0 0; }
        .kick {
            color: var(--accent);
            font: 700 11px system-ui, sans-serif;
            letter-spacing: .15em;
            text-transform: uppercase;
        }
        .title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 30px;
            font-weight: 500;
            margin: 7px 0 3px;
            letter-spacing: -.01em;
        }
        .meta { color: var(--muted); font: 400 13px system-ui, sans-serif; }
        .rule { height: 1px; background: var(--line); margin: 18px 0; }
        .stepper {
            display: flex;
            gap: 26px;
            padding: 0 0 12px;
            border-bottom: 1px solid var(--line);
        }
        .step {
            font: 700 12px system-ui, sans-serif;
            color: #b5ad9f;
            padding-bottom: 12px;
            border-bottom: 2px solid transparent;
        }
        .step.on { color: var(--ink); border-color: var(--accent); }
        .step.done { color: var(--muted); }
        .step span { color: var(--accent); margin-right: 7px; }
        .body { padding: 22px 0 0; }
        .section-label {
            color: var(--muted);
            font: 700 11px system-ui, sans-serif;
            letter-spacing: .07em;
            text-transform: uppercase;
            margin: 6px 0 8px;
        }
        .note { color: var(--muted); font: 13px/1.6 system-ui, sans-serif; margin: 10px 0 0; }
        .ok { color: var(--ok); font: 14px system-ui, sans-serif; margin-top: 14px; }
        .kv {
            display: flex;
            justify-content: space-between;
            padding: 11px 0;
            border-bottom: 1px solid #ece7dd;
            font: 14px system-ui, sans-serif;
        }
        .kv span { color: var(--muted); }
        .summary { color: var(--muted); font: 13px system-ui, sans-serif; margin-bottom: 16px; }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            border: 1px solid var(--line);
            margin-bottom: 24px;
        }
        .stat {
            padding: 15px 16px;
            border-right: 1px solid var(--line);
        }
        .stat:last-child { border-right: 0; }
        .stat-value {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 29px;
            font-weight: 500;
        }
        .stat-label {
            color: var(--muted);
            font: 700 10px system-ui, sans-serif;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-top: 5px;
        }
        div[data-testid="stTextInput"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stFileUploader"] label {
            color: var(--muted);
            font: 700 11px system-ui, sans-serif;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        .stTextInput input {
            background: #fff;
            border: 1px solid #d8d2c6;
            border-radius: 2px;
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
        }
        .stButton > button, .stDownloadButton > button {
            background: var(--ink);
            color: var(--paper);
            border: none;
            border-radius: 2px;
            font: 700 13px system-ui, sans-serif;
            padding: 0.55rem 1.15rem;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: var(--accent);
            color: var(--paper);
            border: none;
        }
        [data-testid="stDataFrame"] {
            border-top: 1.5px solid var(--ink);
        }
        @media (max-width: 760px) {
            .block-container { padding-left: 18px; padding-right: 18px; }
            .stepper { gap: 10px; }
            .stat-grid { grid-template-columns: 1fr 1fr; }
            .stat { border-bottom: 1px solid var(--line); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _open_card() -> None:
    st.markdown(
        """
        <div class="report-top">
            <div class="kick">Model Evaluation</div>
            <div class="title">Titanic Survival - Inference Report</div>
            <div class="rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _close_card() -> None:
    return None


def _stepper() -> None:
    labels = ["Dataset", "Model", "Inference", "Results"]
    html = ['<div class="stepper">']
    for idx, label in enumerate(labels, start=1):
        cls = "step"
        if idx == st.session_state.step:
            cls += " on"
        elif idx < st.session_state.step:
            cls += " done"
        html.append(f'<div class="{cls}"><span>0{idx}</span>{label}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _body_open() -> None:
    st.markdown('<div class="body">', unsafe_allow_html=True)


def _body_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _load_dataset() -> None:
    try:
        upload = st.session_state.get("csv_upload")
        if upload is None:
            raise ValueError("Upload a test dataset CSV.")
        upload.seek(0)
        df = pd.read_csv(upload)
        st.session_state.df = df
        st.session_state.dataset_source = upload.name
        st.session_state.y_true = None
        st.session_state.y_prob = None
        st.session_state.error = None
        st.session_state.step = 2
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = f"Could not read dataset: {exc}"


def _load_model() -> None:
    try:
        model_upload = st.session_state.get("model_upload")
        preprocessor_upload = st.session_state.get("preprocessor_upload")
        if model_upload is None or preprocessor_upload is None:
            raise ValueError("Upload both model.pt and preprocessor.joblib.")
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model_path = tmpdir / "model.pt"
            preprocessor_path = tmpdir / "preprocessor.joblib"
            model_path.write_bytes(model_upload.getvalue())
            preprocessor_path.write_bytes(preprocessor_upload.getvalue())
            pp, clf = load_artifacts(
                preprocessor_path=preprocessor_path,
                model_path=model_path,
            )
        st.session_state.pp = pp
        st.session_state.clf = clf
        st.session_state.model_source = f"{model_upload.name} + {preprocessor_upload.name}"
        st.session_state.y_true = None
        st.session_state.y_prob = None
        st.session_state.error = None
        st.session_state.step = 3
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = f"Could not load model artifacts: {exc}"


def _run_inference() -> None:
    try:
        y_true, y_prob = predict_frame(
            st.session_state.pp,
            st.session_state.clf,
            st.session_state.df,
        )
        st.session_state.y_true = y_true
        st.session_state.y_prob = y_prob
        st.session_state.error = None
        st.session_state.step = 4
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = f"Could not run inference: {exc}"


def _dataset_step() -> None:
    st.markdown('<div class="section-label">Test dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Choose a test dataset (CSV) from disk.</div>',
        unsafe_allow_html=True,
    )
    df = st.session_state.df
    if df is not None:
        target_note = (
            f'with a labeled "{TARGET}" target'
            if TARGET in df.columns else f'without a "{TARGET}" target'
        )
        st.markdown(
            f'<div class="note">Loaded {st.session_state.dataset_source}. Detected '
            f"{len(df)} rows, {len(df.columns)} columns, "
            f"{target_note}.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="note">The CSV may be labeled with "Survived" for evaluation, '
            "or unlabeled for inference-only predictions.</div>",
            unsafe_allow_html=True,
        )
    st.file_uploader("", type=["csv"], key="csv_upload")
    st.button("Continue", on_click=_load_dataset)


def _model_step() -> None:
    st.markdown('<div class="section-label">Model artifacts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Upload both artifacts produced by train.py.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.file_uploader("Upload model.pt", type=["pt"], key="model_upload")
    with right:
        st.file_uploader("Upload preprocessor.joblib", type=["joblib"], key="preprocessor_upload")


    if st.session_state.pp is not None and st.session_state.clf is not None:
        st.markdown(
            f'<div class="note">Source: {st.session_state.model_source}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="ok">{_model_summary(st.session_state.pp, st.session_state.clf)}</div>',
                    unsafe_allow_html=True)
        st.button("Continue", on_click=lambda: st.session_state.update(step=3))
    else:
        st.button("Load model", on_click=_load_model)


def _inference_step() -> None:
    df = st.session_state.df
    pp = st.session_state.pp
    st.markdown(
        '<div class="note">A single forward pass scores P(survived) for every passenger; '
        "the decision threshold defaults to 0.50.</div>",
        unsafe_allow_html=True,
    )
    if df is not None and pp is not None:
        st.markdown(
            f'<div class="note">{len(df)} rows will be transformed into {pp.n_features} '
            "encoded model features.</div>",
            unsafe_allow_html=True,
        )
    st.button("Run inference", on_click=_run_inference)


def _metrics_html(metrics: dict[str, float]) -> None:
    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat"><div class="stat-value">{_fmt_pct(metrics["accuracy"])}</div><div class="stat-label">Accuracy</div></div>
          <div class="stat"><div class="stat-value">{_fmt_pct(metrics["precision"])}</div><div class="stat-label">Precision</div></div>
          <div class="stat"><div class="stat-value">{_fmt_pct(metrics["recall"])}</div><div class="stat-label">Recall</div></div>
          <div class="stat"><div class="stat-value">{_fmt_pct(metrics["f1"])}</div><div class="stat-label">F1</div></div>
          <div class="stat"><div class="stat-value">{_fmt_num(metrics["roc_auc"])}</div><div class="stat-label">ROC AUC</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _results_step() -> None:
    df = st.session_state.df
    y_true = st.session_state.y_true
    y_prob = st.session_state.y_prob
    threshold = 0.5
    y_pred = np.asarray(y_prob) >= threshold
    pos = int(y_pred.sum())
    neg = int(len(y_pred) - pos)
    st.markdown(
        f'<div class="summary">n = {len(df)} passengers - {pos} predicted survived - '
        f"{neg} predicted died - fixed threshold {threshold:.2f}</div>",
        unsafe_allow_html=True,
    )

    if y_true is not None:
        metrics = compute_metrics(y_true, y_prob, threshold)
        _metrics_html(metrics)
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-label">Fig 1 - ROC curve</div>', unsafe_allow_html=True)
            try:
                st.pyplot(roc_figure(y_true, y_prob), clear_figure=True)
            except ValueError as exc:
                st.info(str(exc))
        with right:
            st.markdown('<div class="section-label">Fig 2 - Predicted probability</div>',
                        unsafe_allow_html=True)
            st.pyplot(probability_figure(y_prob, y_true, threshold), clear_figure=True)

        counts = confusion_counts(y_true, y_prob, threshold)
        st.markdown(f'<div class="section-label">Confusion matrix @ {threshold:.2f}</div>',
                    unsafe_allow_html=True)
        st.pyplot(confusion_figure(y_true, y_prob, threshold), clear_figure=True)
        st.markdown(
            f'<div class="note">TP {counts["tp"]} - FN {counts["fn"]} - '
            f'FP {counts["fp"]} - TN {counts["tn"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(f'No "{TARGET}" column found, so the app shows inference outputs without evaluation metrics.')
        st.markdown('<div class="section-label">Predicted probability</div>', unsafe_allow_html=True)
        st.pyplot(probability_figure(y_prob, threshold=threshold), clear_figure=True)

    table = prediction_table(df, y_prob, threshold)
    table["P(survived)"] = table["P(survived)"].map(lambda x: f"{x:.3f}")
    st.markdown('<div class="section-label">Passenger predictions</div>', unsafe_allow_html=True)
    st.dataframe(table, width="stretch", hide_index=True, height=300)
    st.download_button(
        "Download predictions CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="titanic_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Titanic Inference Report", layout="centered")
    _init_state()
    _style()
    _open_card()
    _stepper()
    _body_open()

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.step == 1:
        _dataset_step()
    elif st.session_state.step == 2:
        _model_step()
    elif st.session_state.step == 3:
        _inference_step()
    else:
        _results_step()

    _body_close()
    _close_card()


if __name__ == "__main__":
    main()
