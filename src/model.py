"""PyTorch MLP classifier for Titanic survival, with a scikit-learn-style wrapper.

`MLP` is the network; `TorchClassifier` wraps it with `.fit` / `.predict_proba` so
it drops into the experiment harness (and later `train.py`) interchangeably with
sklearn estimators. Reused by both `experiments/model_search.py` and `train.py`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Feed-forward net: [Linear -> ReLU -> Dropout] x len(hidden_dims) -> Linear(1)."""

    def __init__(self, in_dim: int, hidden_dims=(64, 32), dropout: float = 0.3,
                 batchnorm: bool = False):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchClassifier:
    """sklearn-style wrapper around `MLP` (fit / predict_proba / predict).

    Fixed-epoch training with Adam + BCEWithLogitsLoss; seeded for reproducibility.
    """

    def __init__(self, in_dim: int, hidden_dims=(64, 32), dropout: float = 0.3,
                 batchnorm: bool = False, lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 100, batch_size: int = 64, seed: int = 42,
                 device: str = "cpu"):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    def fit(self, X, y) -> "TorchClassifier":
        torch.manual_seed(self.seed)
        self.model_ = MLP(self.in_dim, self.hidden_dims, self.dropout,
                          self.batchnorm).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        """Return P(survived) for each row."""
        self.model_.eval()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return torch.sigmoid(self.model_(Xt)).cpu().numpy()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
