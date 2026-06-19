"""Convert SMILES strings into PyTorch Geometric molecular graphs.

Each atom becomes a node with a fixed-length feature vector (element, degree,
formal charge, hybridization, attached hydrogens, aromaticity, ring membership);
each bond becomes a pair of directed edges carrying the bond order. This is the
representation consumed by the GCN / GAT / MPNN models in ``graph_models``.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Na", "K", "H"]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-1, 0, 1]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
NUM_HS = [0, 1, 2, 3, 4]


def _one_hot(value, choices: list) -> list[int]:
    """One-hot encode ``value`` over ``choices`` with a trailing 'unknown' slot."""
    vec = [0] * (len(choices) + 1)
    if value in choices:
        vec[choices.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom: Chem.Atom) -> list[int]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_TYPES)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
    )


# Dimensionality is implied by the encoders above; compute it once from methane.
NODE_FEATURE_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
EDGE_FEATURE_DIM = 1


def smiles_to_graph(smiles: str, y: float | None = None) -> Data | None:
    """Build a PyG ``Data`` graph from a SMILES string, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        edges += [[i, j], [j, i]]
        edge_attr += [[order], [order]]

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
    else:  # single-atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


def graphs_from_dataframe(df, smiles_col: str = "smiles", target_col: str | None = None) -> list[Data]:
    """Vectorize a dataframe of SMILES (and optional targets) into a list of graphs."""
    graphs: list[Data] = []
    for _, row in df.iterrows():
        y = float(row[target_col]) if target_col is not None else None
        g = smiles_to_graph(row[smiles_col], y=y)
        if g is not None:
            graphs.append(g)
    return graphs
