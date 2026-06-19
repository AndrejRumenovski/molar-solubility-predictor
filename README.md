# Molar Solubility Predictor

End-to-end **computational chemistry platform** that predicts aqueous solubility (LogS) and a panel of pharmaceutically relevant properties from SMILES strings, and benchmarks classical machine learning against graph neural networks on the [Delaney ESOL dataset](https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv).

## Features

- **Data & featurization** — auto-downloads and caches the Delaney ESOL dataset (1,128 compounds); RDKit descriptors (LogP, molecular weight, rotatable bonds, aromatic proportion, TPSA).
- **Classical models** — Random Forest, Gradient Boosting, and Ridge regression.
- **Graph neural networks** — GCN, GAT, and MPNN (PyTorch Geometric) trained on atom/bond molecular graphs, benchmarked against the classical models on an identical split with k-fold cross-validation.
- **Multi-property profiling** — solubility, lipophilicity, molecular weight, TPSA, drug-likeness (QED), Lipinski Rule of Five, structural-alert toxicity risk (Brenk/PAINS), and blood-brain-barrier penetration (Clark's logBB).
- **Explainable AI** — per-prediction SHAP feature contributions with a plain-language explanation, 95% prediction intervals, and nearest training compounds by Tanimoto similarity.
- **Research dashboard** — unified benchmark table (R², RMSE, MAE, CV R²), parity and residual plots for every model, feature importance, and headline impact metrics.
- **Auto research report** — one-click export of a formatted PDF paper (Abstract → Conclusion + References) with figures and statistics populated from the benchmark.

## Quick Start

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# (Optional) train the GNN benchmark — writes models/gnn_benchmark.json
python -m src.graph_training

# Run the Streamlit app
streamlit run app.py
```

The app runs without the GNN step (it falls back to the classical benchmark); running `src.graph_training` once unlocks the full GCN/GAT/MPNN comparison in the dashboard.

## Project Structure

```
molar-solubility-predictor/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/                   # Cached dataset (auto-downloaded)
├── models/                 # GNN benchmark JSON + trained weights
└── src/
    ├── data_loader.py      # Dataset download & caching
    ├── featurizer.py       # RDKit descriptor extraction
    ├── models.py           # Classical ML training & evaluation
    ├── interpretability.py # SHAP, prediction intervals, similarity search
    ├── properties.py       # Multi-property profiling (QED, tox, BBB, …)
    ├── graph_data.py       # SMILES → PyTorch Geometric graphs
    ├── graph_models.py     # GCN / GAT / MPNN architectures
    ├── graph_training.py   # GNN training, CV, and benchmark export
    └── report.py           # Auto-generated research-paper PDF
```

## Standalone Scripts

```bash
# Inspect dataset statistics
python -m src.data_loader

# Train classical models and print metrics table
python -m src.models

# Train + benchmark the graph neural networks
python -m src.graph_training
```

## Methodology Notes

- Toxicity risk uses medicinal-chemistry structural alerts (Brenk et al. 2008; Baell & Holloway 2010) via RDKit's `FilterCatalog`.
- Drug-likeness uses QED (Bickerton et al. 2012) and Lipinski's Rule of Five (Lipinski et al. 2001).
- BBB penetration probability is derived from Clark's (1999) logBB regression over TPSA and LogP, and is a heuristic estimate rather than a trained classifier.

## Reproducibility

All stochastic operations use `random_state=42`. GNNs and classical models share the same train/test split for a fair comparison.

## License

MIT
