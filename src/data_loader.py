"""Data ingestion for the Delaney ESOL solubility dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
)
SMILES_COL = "smiles"
TARGET_COL = "measured log solubility in mols per litre"
ESOL_PRED_COL = "ESOL predicted log solubility in mols per litre"

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "delaney-processed.csv"


def download_dataset(url: str = DATA_URL, cache_path: Path = DEFAULT_CACHE_PATH) -> Path:
    """Download the Delaney ESOL CSV if not already cached locally."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path

    df = pd.read_csv(url)
    df.to_csv(cache_path, index=False)
    return cache_path


def load_dataset(cache_path: Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """Load the Delaney dataset, downloading and caching on first access."""
    path = download_dataset(cache_path=cache_path)
    return pd.read_csv(path)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic statistics for the raw dataset."""
    target = df[TARGET_COL]
    return {
        "n_compounds": len(df),
        "columns": list(df.columns),
        "logS_mean": float(target.mean()),
        "logS_std": float(target.std()),
        "logS_min": float(target.min()),
        "logS_max": float(target.max()),
        "logS_median": float(target.median()),
    }


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of the dataset to stdout."""
    stats = dataset_summary(df)
    print("=" * 50)
    print("Delaney ESOL Dataset Summary")
    print("=" * 50)
    print(f"Compounds:        {stats['n_compounds']}")
    print(f"Columns:          {', '.join(stats['columns'])}")
    print(f"LogS mean:        {stats['logS_mean']:.3f}")
    print(f"LogS std:         {stats['logS_std']:.3f}")
    print(f"LogS median:      {stats['logS_median']:.3f}")
    print(f"LogS range:       [{stats['logS_min']:.3f}, {stats['logS_max']:.3f}]")
    print("=" * 50)


if __name__ == "__main__":
    dataset = load_dataset()
    print_dataset_summary(dataset)
    print("\nFirst 5 rows:")
    print(dataset.head())
