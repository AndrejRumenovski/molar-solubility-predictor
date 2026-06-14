# Molar Solubility Predictor

End-to-end **QSPR (Quantitative Structure-Property Relationship)** pipeline that predicts aqueous solubility (LogS) from SMILES strings using the [Delaney ESOL dataset](https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv).

## Features

- Auto-downloads and caches the Delaney ESOL dataset (1,128 compounds)
- RDKit featurization: LogP, molecular weight, rotatable bonds, aromatic proportion, TPSA
- Trains Random Forest, Gradient Boosting, and Ridge regression models
- Interactive Streamlit dashboard with live SMILES inference and 2D structure rendering
- Plotly parity plots, feature importance charts, and model comparison metrics

## Quick Start

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Project Structure

```
molar-solubility-predictor/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/                   # Cached dataset (auto-downloaded)
└── src/
    ├── data_loader.py      # Dataset download & caching
    ├── featurizer.py       # RDKit descriptor extraction
    └── models.py           # ML training & evaluation
```

## Standalone Scripts

```bash
# Inspect dataset statistics
python -m src.data_loader

# Train models and print metrics table
python -m src.models
```

## Reproducibility

All stochastic operations use `random_state=42`.

## License

MIT
