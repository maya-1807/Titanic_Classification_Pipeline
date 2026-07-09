# Titanic Survival — Classification Pipeline

End-to-end PyTorch classification pipeline predicting Titanic passenger survival, built for the
ELTA "Software and AI Engineer" home assignment.

- **EDA** in a Jupyter notebook (`notebooks/eda.ipynb`)
- **Training** in a standalone script (`train.py`) that fetches data, preprocesses, trains a
  PyTorch model, and saves the weights + fitted preprocessor to disk
- **Inference UI** in Streamlit (`ds_app.py`) to load the model, run inference on a test CSV,
  and view evaluation metrics and plots

> **Note on the dataset:** Kaggle's Titanic competition ships a labeled `train.csv` (891 rows)
> and an *unlabeled* `test.csv` (the `Survived` column is hidden for leaderboard scoring). To
> produce a genuine held-out evaluation with metrics, we treat `train.csv` as the full labeled
> dataset and split it into train / validation / test ourselves (stratified on `Survived`).

## Project structure

```
.
├── train.py              # standalone training script
├── ds_app.py             # Streamlit inference + evaluation app
├── src/                  # shared package (data, preprocessing, model, evaluation)
├── notebooks/eda.ipynb   # exploratory data analysis
├── data/                 # datasets (a small sample CSV is committed)
├── models/               # saved weights + fitted preprocessor
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Kaggle credentials (for fetching the full dataset)

`train.py` fetches the dataset directly from Kaggle via `kagglehub`, which needs API
credentials:

1. Kaggle → Account → **Create New API Token** → downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (`chmod 600 ~/.kaggle/kaggle.json`).
3. Accept the competition rules once at https://www.kaggle.com/competitions/titanic/rules

Without credentials, you can still run the pipeline against the committed
`data/sample_test.csv` sample.

## Run

Train the model:

```bash
python train.py
```

Run the app:

```bash
streamlit run ds_app.py
```

## Architecture & design choices

_TODO: filled in as the implementation lands._
