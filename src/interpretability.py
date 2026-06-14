"""Model interpretability tools: SHAP explanations, prediction intervals, similarity search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import shap
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

if TYPE_CHECKING:
    from src.models import ModelResult

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
CONFIDENCE_Z = 1.96  # ~95% critical value for a normal distribution

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS
)


def _morgan_fingerprint(smiles: str):
    """Return a Morgan (ECFP4) bit-vector fingerprint for a SMILES, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Compute Tanimoto similarity (0–1) between two SMILES strings."""
    fp1 = _morgan_fingerprint(smiles1)
    fp2 = _morgan_fingerprint(smiles2)
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def find_similar_compounds(
    query_smiles: str,
    training_smiles: pd.Series,
    training_logs: pd.Series,
    n_neighbors: int = 5,
    min_similarity: float = 0.3,
) -> pd.DataFrame:
    """
    Find the most similar training compounds to a query via Tanimoto similarity.

    Returns a DataFrame [SMILES, Similarity, Experimental LogS] sorted by descending
    similarity, keeping at most ``n_neighbors`` rows above ``min_similarity``. The
    query fingerprint is computed once and reused across the training set.
    """
    columns = ["SMILES", "Similarity", "Experimental LogS"]

    query_fp = _morgan_fingerprint(query_smiles)
    if query_fp is None:
        return pd.DataFrame(columns=columns)

    similarities = [
        DataStructs.TanimotoSimilarity(query_fp, fp) if fp is not None else 0.0
        for fp in (_morgan_fingerprint(s) for s in training_smiles)
    ]

    results_df = pd.DataFrame(
        {
            "SMILES": np.asarray(training_smiles),
            "Similarity": similarities,
            "Experimental LogS": np.asarray(training_logs),
        }
    )
    results_df = results_df[results_df["Similarity"] >= min_similarity]
    return (
        results_df.sort_values("Similarity", ascending=False)
        .head(n_neighbors)
        .reset_index(drop=True)
    )


def prediction_interval_halfwidth(
    model_result: ModelResult,
    confidence_z: float = CONFIDENCE_Z,
) -> float:
    """
    Half-width of a prediction interval, derived from the test-set residual spread.

    Assumes roughly homoscedastic, normally distributed residuals; the returned value
    is ``z * std(residuals)`` so a prediction can be reported as ``ŷ ± halfwidth``.
    """
    residuals = model_result.y_test - model_result.y_pred
    return float(confidence_z * np.std(residuals))


def explain_prediction(
    model_result: ModelResult,
    feature_row: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Compute per-feature SHAP contributions for a single prediction.

    The SHAP TreeExplainer operates on the underlying estimator, which expects the
    *scaled* feature space, so the row is pushed through the pipeline's scaler first.
    Returns a DataFrame [Feature, Contribution] sorted by absolute impact, or None for
    non-tree models that lack a TreeExplainer.
    """
    model = model_result.pipeline.named_steps["model"]
    if not isinstance(model, RandomForestRegressor):
        return None

    scaler = model_result.pipeline.named_steps["scaler"]
    scaled = scaler.transform(feature_row)

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(scaled))
    contributions = shap_values[0]

    result = pd.DataFrame(
        {
            "Feature": list(feature_row.columns),
            "Contribution": contributions,
        }
    )
    return (
        result.reindex(result["Contribution"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )
