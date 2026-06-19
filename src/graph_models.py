"""Graph neural network architectures for molecular property regression.

Three message-passing variants, all reducing a molecular graph to a single
scalar (LogS) via global mean pooling + an MLP head:

* ``GCN``  — Graph Convolutional Network (Kipf & Welling, 2017).
* ``GAT``  — Graph Attention Network (Veličković et al., 2018).
* ``MPNN`` — Message Passing Neural Network with edge-conditioned convolutions
  and a GRU update (Gilmer et al., 2017), the only variant that uses bond features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import GRU, BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GCNConv, NNConv, global_mean_pool

from src.graph_data import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int = NODE_FEATURE_DIM, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        dim = in_dim
        for _ in range(layers):
            self.convs.append(GCNConv(dim, hidden))
            self.norms.append(BatchNorm1d(hidden))
            dim = hidden
        self.head = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
        return self.head(global_mean_pool(x, batch)).squeeze(-1)


class GAT(torch.nn.Module):
    def __init__(self, in_dim: int = NODE_FEATURE_DIM, hidden: int = 64, heads: int = 4, layers: int = 3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        dim = in_dim
        for i in range(layers):
            last = i == layers - 1
            n_heads = 1 if last else heads
            self.convs.append(GATConv(dim, hidden, heads=n_heads, concat=not last))
            self.norms.append(BatchNorm1d(hidden * (1 if last else heads)))
            dim = hidden * (1 if last else heads)
        self.head = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(norm(conv(x, edge_index)))
        return self.head(global_mean_pool(x, batch)).squeeze(-1)


class MPNN(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden: int = 64,
        steps: int = 3,
    ):
        super().__init__()
        self.steps = steps
        self.lin0 = Linear(in_dim, hidden)
        edge_nn = Sequential(Linear(edge_dim, 32), ReLU(), Linear(32, hidden * hidden))
        self.conv = NNConv(hidden, hidden, edge_nn, aggr="mean")
        self.gru = GRU(hidden, hidden)
        self.head = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)
        for _ in range(self.steps):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        return self.head(global_mean_pool(out, batch)).squeeze(-1)


MODEL_REGISTRY = {"GCN": GCN, "GAT": GAT, "MPNN": MPNN}


def build_model(name: str) -> torch.nn.Module:
    return MODEL_REGISTRY[name]()
