"""Machine learning training and evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import TARGET_COL
from src.featurizer import FEATURE_COLUMNS

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_SPECS: dict[str, object] = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE,
    ),
    "Ridge Regression": Ridge(),
}


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    r2: float
    mse: float
    mae: float
    y_test: np.ndarray
    y_pred: np.ndarray


@dataclass
class TrainingBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    smiles_train: pd.Series
    smiles_test: pd.Series
    results: list[ModelResult]
    feature_columns: list[str]


def _build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def train_models(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_col: str = TARGET_COL,
    smiles_col: str = "smiles",
) -> TrainingBundle:
    """Train all regression models and return metrics plus fitted pipelines."""
    features = feature_columns or FEATURE_COLUMNS

    X = df[features]
    y = df[target_col]
    smiles = df[smiles_col]

    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X,
        y,
        smiles,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    results: list[ModelResult] = []
    for name, estimator in MODEL_SPECS.items():
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append(
            ModelResult(
                name=name,
                pipeline=pipeline,
                r2=r2_score(y_test, y_pred),
                mse=mean_squared_error(y_test, y_pred),
                mae=mean_absolute_error(y_test, y_pred),
                y_test=y_test.values,
                y_pred=y_pred,
            )
        )

    return TrainingBundle(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        smiles_train=smiles_train.reset_index(drop=True),
        smiles_test=smiles_test.reset_index(drop=True),
        results=results,
        feature_columns=features,
    )


def metrics_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    """Build a comparison table of model metrics."""
    return pd.DataFrame(
        [
            {
                "Model": r.name,
                "R²": round(r.r2, 4),
                "MSE": round(r.mse, 4),
                "MAE": round(r.mae, 4),
            }
            for r in results
        ]
    )


def print_metrics_table(results: list[ModelResult]) -> None:
    """Print a formatted model comparison table to stdout."""
    table = metrics_dataframe(results)
    print("\nModel Comparison (test set, 20% holdout)")
    print("=" * 55)
    print(table.to_string(index=False))
    print("=" * 55)


def get_best_model_name(results: list[ModelResult]) -> str:
    """Return the name of the model with the highest R² on the test set."""
    return max(results, key=lambda r: r.r2).name


def get_model_by_name(results: list[ModelResult], name: str) -> ModelResult:
    """Look up a trained model result by name."""
    for result in results:
        if result.name == name:
            return result
    raise KeyError(f"Model '{name}' not found.")


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.featurizer import featurize_dataframe

    raw = load_dataset()
    featurized = featurize_dataframe(raw)
    bundle = train_models(featurized)
    print_metrics_table(bundle.results)
