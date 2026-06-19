"""Train and benchmark the GNNs against the classical models.

Running ``python -m src.graph_training`` trains GCN/GAT/MPNN on the Delaney
ESOL dataset using the *same* train/test split as the scikit-learn models,
evaluates every model on the held-out test set, runs k-fold cross-validation,
and writes:

* ``models/gnn_benchmark.json`` — unified metrics (R², RMSE, MAE, CV R²) for
  Ridge / Random Forest / Gradient Boosting / GCN / GAT / MPNN, plus held-out
  predictions for parity plots.
* ``models/<name>.pt`` — trained GNN weights.

The Streamlit app reads the JSON to render the benchmark table; it never trains
GNNs at request time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from torch_geometric.loader import DataLoader

from src.data_loader import TARGET_COL, load_dataset
from src.featurizer import FEATURE_COLUMNS, featurize_dataframe
from src.graph_data import graphs_from_dataframe
from src.graph_models import MODEL_REGISTRY, build_model
from src.models import MODEL_SPECS, RANDOM_STATE, TEST_SIZE, _build_pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BENCHMARK_PATH = MODELS_DIR / "gnn_benchmark.json"

EPOCHS_FINAL = 180
EPOCHS_CV = 90
CV_FOLDS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.005
DEVICE = torch.device("cpu")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _train_gnn(model, train_graphs, y_mean, y_std, epochs):
    """Train a GNN on standardized targets; returns the fitted model."""
    model.to(DEVICE).train()
    loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            target = (batch.y - y_mean) / y_std
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def _predict_gnn(model, graphs, y_mean, y_std) -> np.ndarray:
    model.eval()
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch).cpu().numpy() * y_std + y_mean)
    return np.concatenate(preds)


def _gnn_cv_r2(name, graphs, targets) -> tuple[float, float]:
    """k-fold cross-validated R² for a GNN architecture."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_i, val_i in kf.split(np.arange(len(graphs))):
        train_g = [graphs[i] for i in train_i]
        val_g = [graphs[i] for i in val_i]
        y_mean = float(targets[train_i].mean())
        y_std = float(targets[train_i].std()) or 1.0
        model = _train_gnn(build_model(name), train_g, y_mean, y_std, EPOCHS_CV)
        preds = _predict_gnn(model, val_g, y_mean, y_std)
        scores.append(r2_score(targets[val_i], preds))
    return float(np.mean(scores)), float(np.std(scores))


def run_benchmark() -> dict:
    print("Loading and featurizing dataset…")
    df = featurize_dataframe(load_dataset())
    targets = df[TARGET_COL].to_numpy()
    n = len(df)

    # Same split as src.models.train_models (identical n, seed, test_size).
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train, y_test = targets[train_idx], targets[test_idx]

    results: dict[str, dict] = {}

    # ── Classical models ────────────────────────────────────────────────────
    X = df[FEATURE_COLUMNS]
    for name, estimator in MODEL_SPECS.items():
        print(f"[sklearn] {name}…")
        pipe = _build_pipeline(estimator)
        pipe.fit(X.iloc[train_idx], y_train)
        y_pred = pipe.predict(X.iloc[test_idx])
        cv = cross_val_score(_build_pipeline(estimator), X, targets, cv=CV_FOLDS, scoring="r2")
        results[name] = {
            "family": "Classical",
            **_metrics(y_test, y_pred),
            "cv_r2_mean": float(cv.mean()),
            "cv_r2_std": float(cv.std()),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }

    # ── Graph neural networks ───────────────────────────────────────────────
    print("Building molecular graphs…")
    all_graphs = graphs_from_dataframe(df, target_col=TARGET_COL)
    train_graphs = [all_graphs[i] for i in train_idx]
    test_graphs = [all_graphs[i] for i in test_idx]
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0

    MODELS_DIR.mkdir(exist_ok=True)
    for name in MODEL_REGISTRY:
        t0 = time.time()
        print(f"[GNN] {name} — held-out training ({EPOCHS_FINAL} epochs)…")
        model = _train_gnn(build_model(name), train_graphs, y_mean, y_std, EPOCHS_FINAL)
        y_pred = _predict_gnn(model, test_graphs, y_mean, y_std)
        torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")

        print(f"[GNN] {name} — {CV_FOLDS}-fold cross-validation…")
        cv_mean, cv_std = _gnn_cv_r2(name, all_graphs, targets)
        results[name] = {
            "family": "Graph Neural Network",
            **_metrics(y_test, y_pred),
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }
        print(f"    {name}: R²={results[name]['r2']:.4f}  ({time.time()-t0:.0f}s)")

    payload = {
        "dataset": "Delaney ESOL",
        "n_compounds": n,
        "n_test": len(test_idx),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "results": results,
    }
    BENCHMARK_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {BENCHMARK_PATH}")
    return payload


if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    run_benchmark()
