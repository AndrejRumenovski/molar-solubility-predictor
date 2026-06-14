"""RDKit-based molecular descriptor featurization."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

FEATURE_COLUMNS = [
    "MolLogP",
    "MolWt",
    "NumRotatableBonds",
    "AromaticProportion",
    "TPSA",
]


def smiles_to_descriptors(smiles: str) -> dict[str, float] | None:
    """
    Parse a SMILES string and return five solubility-relevant descriptors.

    Returns None if RDKit cannot parse the SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms == 0:
            return None

        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        aromatic_proportion = aromatic_atoms / heavy_atoms

        return {
            "MolLogP": Crippen.MolLogP(mol),
            "MolWt": Descriptors.MolWt(mol),
            "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
            "AromaticProportion": aromatic_proportion,
            "TPSA": Descriptors.TPSA(mol),
        }
    except Exception:
        return None


def featurize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """
    Append RDKit descriptor columns to a DataFrame and drop rows with invalid SMILES.
    """
    descriptors: list[dict[str, float] | None] = [
        smiles_to_descriptors(smiles) for smiles in df[smiles_col]
    ]

    valid_mask = [d is not None for d in descriptors]
    filtered = df.loc[valid_mask].copy().reset_index(drop=True)

    feature_rows: list[dict[str, Any]] = [d for d in descriptors if d is not None]
    feature_df = pd.DataFrame(feature_rows)

    return pd.concat([filtered, feature_df], axis=1)


def interpret_logs(logs: float) -> str:
    """Return a human-readable solubility interpretation for a LogS value."""
    if logs > 0:
        return "Highly soluble — readily dissolves in water."
    if logs > -4:
        return "Moderately soluble — partial dissolution expected."
    return "Poorly soluble — limited aqueous dissolution."
