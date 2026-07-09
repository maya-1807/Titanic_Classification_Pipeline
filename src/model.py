"""PyTorch MLP classifier for Titanic survival, with a scikit-learn-style wrapper."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
OPTIMIZERS = {
    "adam":  lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
    "adamw": lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
    "sgd":   lambda p, lr, wd: torch.optim.SGD(p, lr=lr, weight_decay=wd, momentum=0.9),
}


class MLP(nn.Module):
    """Feed-forward net: [Linear -> (BN) -> act -> Dropout] x depth -> Linear(1)."""

    def __init__(self, in_dim: int, hidden_dims=(64, 32), dropout: float = 0.3,
                 batchnorm: bool = False, activation: str = "relu"):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers, d = [], in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers += [act(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchClassifier:
    """sklearn-style wrapper around `MLP` (fit / predict_proba / predict).

    Adam/AdamW/SGD + BCEWithLogitsLoss, seeded for reproducibility. Trains up to
    `max_epochs` but early-stops on an internal validation split (`val_fraction`)
    with `patience`, restoring the best weights — so run length is learned, not set.
    """

    def __init__(self, in_dim: int, hidden_dims=(64, 32), dropout: float = 0.3,
                 batchnorm: bool = False, activation: str = "relu", lr: float = 1e-3,
                 weight_decay: float = 1e-4, optimizer: str = "adam",
                 max_epochs: int = 300, patience: int = 20, batch_size: int = 64,
                 val_fraction: float = 0.15, seed: int = 42, device: str = "cpu"):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.seed = seed
        self.device = device

    def fit(self, X, y) -> "TorchClassifier":
        torch.manual_seed(self.seed)
        self.model_ = MLP(self.in_dim, self.hidden_dims, self.dropout,
                          self.batchnorm, self.activation).to(self.device)
        opt = OPTIMIZERS[self.optimizer](self.model_.parameters(), self.lr, self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        # Internal train / early-stopping split (keeps the caller's val set clean).
        X, y = np.asarray(X), np.asarray(y)
        tr, es = train_test_split(np.arange(len(X)), test_size=self.val_fraction,
                                  stratify=y, random_state=self.seed)
        to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        Xtr, ytr, Xes, yes = to_t(X[tr]), to_t(y[tr]), to_t(X[es]), to_t(y[es])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr, ytr),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        best_loss, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(Xes), yes).item()
            if val_loss < best_loss - 1e-4:
                best_loss, bad = val_loss, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            elif (bad := bad + 1) >= self.patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        """Return P(survived) for each row."""
        self.model_.eval()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return torch.sigmoid(self.model_(Xt)).cpu().numpy()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist weights plus the architecture needed to rebuild the net."""
        torch.save({
            "state_dict": self.model_.state_dict(),
            "arch": {"in_dim": self.in_dim, "hidden_dims": self.hidden_dims,
                     "dropout": self.dropout, "batchnorm": self.batchnorm,
                     "activation": self.activation},
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TorchClassifier":
        """Rebuild a fitted classifier from `save()`, ready for predict_proba/predict."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(**ckpt["arch"], device=device)
        obj.model_ = MLP(**ckpt["arch"]).to(device)
        obj.model_.load_state_dict(ckpt["state_dict"])
        obj.model_.eval()
        return obj
