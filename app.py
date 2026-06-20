"""Molar Solubility Predictor — computational chemistry research platform.

A multi-page Streamlit application (dashboard, live prediction, batch screening,
model benchmarks, molecular search, research reports) styled as a dense, dark
scientific terminal in the spirit of Bloomberg / Palantir / Schrödinger Maestro.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors

from src.data_loader import TARGET_COL, load_dataset
from src.featurizer import FEATURE_COLUMNS, featurize_dataframe, interpret_logs, smiles_to_descriptors
from src.interpretability import (
    explain_prediction,
    find_similar_compounds,
    natural_language_explanation,
    prediction_interval_halfwidth,
)
from src.models import get_model_by_name, metrics_dataframe, train_models
from src.properties import molecular_property_profile
from src.report import generate_report_pdf

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG = "#050816"
C_PANEL = "#0B1220"
C_PANEL2 = "#111827"
C_BORDER = "#1c2740"
C_ACCENT = "#00D4FF"
C_SUCCESS = "#00E676"
C_WARNING = "#FFB020"
C_DANGER = "#FF5C7A"
C_TEXT = "#E6EDF6"
C_MUTED = "#8A97AD"

SENTIMENT_COLOR = {"good": C_SUCCESS, "neutral": C_ACCENT, "warn": C_WARNING, "bad": C_DANGER}

NAV = [
    "Dashboard", "Live Prediction", "Batch Screening", "Model Benchmarks",
    "Molecular Search", "Research Reports", "Settings",
]
NAV_ICON = {
    "Dashboard": "▣", "Live Prediction": "⏵", "Batch Screening": "▦",
    "Model Benchmarks": "▲", "Molecular Search": "⊙", "Research Reports": "▤",
    "Settings": "⚙",
}

st.set_page_config(
    page_title="Molar Solubility Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "svg", "filename": "chart", "scale": 2},
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:#050816; --panel:#0B1220; --panel2:#111827; --panel3:#162033;
    --border:#1c2740; --border2:#26334d;
    --accent:#00D4FF; --accent-dim:rgba(0,212,255,0.10); --accent-bd:rgba(0,212,255,0.45);
    --success:#00E676; --warning:#FFB020; --danger:#FF5C7A;
    --text:#E6EDF6; --muted:#8A97AD; --faint:#5b6678;
}
@keyframes fadeUp { from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:none;} }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; } }

html, body, .stApp { background:var(--bg) !important; }
html, body, .stApp, p, label, div, span, li, td, th, input, button, select {
    font-family:'Inter', sans-serif !important;
}
/* Don't let the Inter override clobber Streamlit's Material icon font (or its
   ligature names render as literal text, e.g. "keyboard_double_arrow_left"). */
[data-testid="stIconMaterial"] {
    font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}
h1,h2,h3,h4 { font-family:'Inter',sans-serif !important; letter-spacing:-0.02em !important; color:var(--text) !important; }
.stApp { color:var(--text); }

/* Chrome */
/* Hide only the Deploy/menu/status — NOT the whole toolbar, which also holds
   the sidebar-expand control needed to bring the sidebar back when collapsed. */
#MainMenu, footer { visibility:hidden !important; }
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"] { display:none !important; }
[data-testid="stExpandSidebarButton"] button:hover { color:var(--accent) !important; }
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stDecoration"] { display:none !important; }
[data-testid="stMainBlockContainer"] { padding:1.1rem 1.8rem 3rem !important; max-width:100% !important; }
/* When the sidebar is collapsed, Streamlit floats the expand button top-left;
   indent the content so it doesn't overlap the header title. */
.stApp:has([data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stMainBlockContainer"] {
    padding-left:3.6rem !important;
}
[data-testid="stVerticalBlock"] { gap:0.7rem !important; }

/* Sidebar */
[data-testid="stSidebar"] { background:var(--panel) !important; border-right:1px solid var(--border) !important; width:248px !important; }
[data-testid="stSidebarContent"] { padding:1.1rem 0.9rem !important; }
[data-testid="stSidebar"] hr { border-color:var(--border) !important; margin:0.8rem 0 !important; }

/* Sidebar nav (radio) */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap:3px !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    display:flex; align-items:center; width:100%; padding:8px 11px; border-radius:7px;
    cursor:pointer; color:var(--muted); font-size:0.85rem !important; font-weight:500;
    border:1px solid transparent; transition:background .14s, color .14s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:var(--panel3); color:var(--text); }
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display:none !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background:var(--accent-dim); color:var(--accent); border:1px solid var(--accent-bd);
}

/* Glass panels (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:linear-gradient(180deg, rgba(20,28,44,0.65), rgba(11,18,32,0.65)) !important;
    border:1px solid var(--border) !important; border-radius:12px !important;
    padding:1rem 1.15rem !important; box-shadow:0 1px 2px rgba(0,0,0,0.35);
}

/* Inputs */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
    font-family:'JetBrains Mono',monospace !important; font-size:0.82rem !important;
    background:var(--panel2) !important; border:1px solid var(--border) !important; color:var(--text) !important;
}
[data-testid="stTextInput"] input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 2px var(--accent-dim) !important; }
[data-baseweb="select"] > div:first-child { background:var(--panel2) !important; border:1px solid var(--border) !important; font-size:0.82rem !important; }
[data-testid="stFileUploader"] { border:1px dashed var(--border) !important; border-radius:10px !important; background:var(--panel2) !important; }
hr { border-color:var(--border) !important; }

/* Buttons */
button[kind="primary"] {
    background:var(--accent) !important; color:#04121a !important; font-weight:600 !important;
    border:none !important; border-radius:8px !important; letter-spacing:0.01em !important;
}
button[kind="primary"]:hover { filter:brightness(1.08) !important; }
button[kind="secondary"] {
    background:var(--panel2) !important; color:var(--text) !important; border:1px solid var(--border) !important; border-radius:8px !important;
}
[data-testid="stDownloadButton"] button { background:var(--panel2) !important; color:var(--text) !important; border:1px solid var(--border2) !important; border-radius:8px !important; font-size:0.78rem !important; }
[data-testid="stDownloadButton"] button:hover { border-color:var(--accent) !important; color:var(--accent) !important; }

/* Tables */
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:10px !important; }

/* Tabs (used inside pages) */
button[data-baseweb="tab"] { color:var(--muted) !important; font-size:0.82rem !important; font-weight:500 !important; }
button[aria-selected="true"][data-baseweb="tab"] { color:var(--accent) !important; }
[data-baseweb="tab-highlight"] { background:var(--accent) !important; }

/* Typography helpers */
.eyebrow { font-size:0.6rem; font-weight:600; text-transform:uppercase; letter-spacing:0.13em; color:var(--muted); display:block; margin-bottom:0.5rem; }
.eyebrow.accent { color:var(--accent); }
.subcap { font-size:0.76rem; color:var(--muted); display:block; margin-bottom:0.5rem; line-height:1.45; }

/* Header bar */
.appbar { display:flex; align-items:center; gap:1.4rem; flex-wrap:wrap; }
.appbar .brand { font-size:1.05rem; font-weight:700; color:var(--text); letter-spacing:-0.02em; display:flex; align-items:center; gap:0.5rem; }
.appbar .stat { display:flex; flex-direction:column; line-height:1.15; }
.appbar .stat .k { font-size:0.55rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--faint); }
.appbar .stat .v { font-family:'JetBrains Mono',monospace !important; font-size:0.86rem; font-weight:500; color:var(--text); }
.appbar .stat .v.accent { color:var(--accent); }
.badge { font-size:0.62rem; font-weight:600; padding:0.18rem 0.55rem; border-radius:999px; border:1px solid var(--border2); color:var(--muted); white-space:nowrap; }
.badge.live { color:var(--success); border-color:rgba(0,230,118,0.4); }

/* KPI cards */
.grid { display:grid; gap:0.7rem; }
.kpi-grid { grid-template-columns:repeat(auto-fit, minmax(150px,1fr)); }
.kpi { background:rgba(17,24,39,0.5); border:1px solid var(--border); border-radius:11px; padding:0.85rem 1rem; }
.kpi .k { font-size:0.58rem; font-weight:600; text-transform:uppercase; letter-spacing:0.09em; color:var(--muted); display:block; margin-bottom:0.45rem; }
.kpi .v { font-family:'JetBrains Mono',monospace !important; font-size:1.7rem; font-weight:600; line-height:1; display:block; color:var(--text); }
.kpi .s { font-size:0.62rem; color:var(--faint); display:block; margin-top:0.35rem; }

/* Property / result cards */
.prop-grid { grid-template-columns:repeat(auto-fit, minmax(155px,1fr)); }
.prop { background:rgba(17,24,39,0.5); border:1px solid var(--border); border-left:3px solid var(--c,var(--accent));
        border-radius:10px; padding:0.8rem 0.9rem; transition:transform .14s, border-color .14s; }
.prop:hover { transform:translateY(-2px); }
.prop .k { font-size:0.56rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); display:block; margin-bottom:0.4rem; }
.prop .v { font-family:'JetBrains Mono',monospace !important; font-size:1.35rem; font-weight:600; line-height:1; color:var(--c,var(--text)); display:block; }
.prop .v small { font-size:0.62rem; color:var(--muted); }
.prop .c { font-size:0.68rem; font-weight:500; color:var(--c,var(--muted)); display:block; margin-top:0.35rem; }
.prop .t { font-size:0.58rem; color:var(--faint); display:block; margin-top:0.25rem; }

/* Model leaderboard */
.lead { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:0.55rem 0.7rem; margin-bottom:0.4rem; position:relative; }
.lead.best { border-color:var(--accent-bd); }
.lead .row1 { display:flex; align-items:center; justify-content:space-between; }
.lead .nm { font-size:0.76rem; font-weight:600; color:var(--text); display:flex; align-items:center; gap:0.4rem; }
.lead .dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.lead .bestbadge { font-size:0.5rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#04121a; background:var(--accent); padding:0.08rem 0.35rem; border-radius:4px; }
.lead .row2 { display:flex; gap:1rem; margin-top:0.35rem; }
.lead .met { display:flex; flex-direction:column; }
.lead .met .ml { font-size:0.5rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--faint); }
.lead .met .mv { font-family:'JetBrains Mono',monospace !important; font-size:0.8rem; font-weight:500; color:var(--accent); line-height:1.1; }

/* XAI note */
.xai { background:var(--accent-dim); border:1px solid var(--border); border-left:3px solid var(--accent);
       border-radius:9px; padding:0.75rem 1rem; font-size:0.82rem; line-height:1.5; color:var(--text); }
.xai b { color:var(--accent); }

/* Section title */
.sec { font-size:0.95rem; font-weight:600; color:var(--text); margin:0.2rem 0 0.1rem; letter-spacing:-0.01em; }

/* Scrollbar */
::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:5px; }
::-webkit-scrollbar-thumb:hover { background:var(--muted); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

_CHART = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(11,18,32,0.0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=C_MUTED, size=11),
    margin=dict(l=8, r=8, t=42, b=8),
    title_font=dict(size=13, color=C_TEXT),
)


# ── Cached data / artifacts ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Delaney ESOL dataset…")
def load_and_featurize() -> pd.DataFrame:
    return featurize_dataframe(load_dataset())


@st.cache_resource(show_spinner="Training models…")
def get_training_bundle():
    return train_models(load_and_featurize())


BENCHMARK_PATH = Path(__file__).resolve().parent / "models" / "gnn_benchmark.json"


@st.cache_data
def load_benchmark() -> dict | None:
    if BENCHMARK_PATH.exists():
        return json.loads(BENCHMARK_PATH.read_text())
    return None


@st.cache_data
def dataset_means() -> dict:
    df = load_and_featurize()
    return {
        "Aqueous Solubility (LogS)": float(df[TARGET_COL].mean()),
        "Lipophilicity (LogP)": float(df["MolLogP"].mean()),
        "Molecular Weight": float(df["MolWt"].mean()),
        "Polar Surface Area (TPSA)": float(df["TPSA"].mean()),
    }


@st.cache_data(show_spinner="Composing research report…")
def cached_report(_payload: dict, cache_key) -> bytes:
    return generate_report_pdf(_payload)


def benchmark_table(payload: dict) -> pd.DataFrame:
    rows = [
        {"Model": n, "Family": r["family"], "R²": round(r["r2"], 4),
         "RMSE": round(r["rmse"], 4), "MAE": round(r["mae"], 4),
         "CV R² (mean ± std)": f"{r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}"}
        for n, r in payload["results"].items()
    ]
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


def model_rows(bundle, benchmark) -> list[dict]:
    if benchmark:
        rows = [{"name": n, "r2": r["r2"], "mae": r["mae"], "family": r["family"]}
                for n, r in benchmark["results"].items()]
    else:
        rows = [{"name": m.name, "r2": m.r2, "mae": m.mae, "family": "Classical"}
                for m in bundle.results]
    return sorted(rows, key=lambda d: d["r2"], reverse=True)


def _status_color(r2: float) -> str:
    if r2 >= 0.87:
        return C_SUCCESS
    if r2 >= 0.83:
        return C_ACCENT
    if r2 >= 0.78:
        return C_WARNING
    return C_DANGER


# ── Chart builders ──────────────────────────────────────────────────────────────
def feature_importance_figure(model_result, feature_columns):
    model = model_result.pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = abs(model.coef_)
    else:
        return None
    df = pd.DataFrame({"Feature": feature_columns, "Importance": imp}).sort_values("Importance")
    fig = px.bar(df, x="Importance", y="Feature", orientation="h", title="Feature Importance",
                 color="Importance", color_continuous_scale=[[0, C_PANEL2], [1, C_ACCENT]])
    fig.update_layout(**_CHART, height=260, showlegend=False, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    return fig


def parity_plot(y_true, y_pred, smiles=None, title="Predicted vs. Experimental LogS"):
    data = {"Experimental LogS": list(y_true), "Predicted LogS": list(y_pred)}
    hover = {"Experimental LogS": ":.3f", "Predicted LogS": ":.3f"}
    if smiles is not None:
        data["SMILES"] = list(smiles)
        hover["SMILES"] = True
    df = pd.DataFrame(data)
    fig = px.scatter(df, x="Experimental LogS", y="Predicted LogS", hover_data=hover,
                     title=title, opacity=0.6, color_discrete_sequence=[C_ACCENT])
    lo = min(df["Experimental LogS"].min(), df["Predicted LogS"].min()) - 0.5
    hi = max(df["Experimental LogS"].max(), df["Predicted LogS"].max()) + 0.5
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                             line=dict(color=C_DANGER, dash="dash", width=1)))
    fig.update_layout(**_CHART, height=300, xaxis_title="Experimental LogS (log mol/L)",
                      yaxis_title="Predicted LogS (log mol/L)")
    return fig


def residuals_figure(y_true, y_pred, title="Residual Distribution"):
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig = px.histogram(pd.DataFrame({"Residual": res}), x="Residual", nbins=28, title=title,
                       color_discrete_sequence=[C_ACCENT])
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color=C_DANGER)
    fig.update_layout(**_CHART, height=300, xaxis_title="Residual (actual − predicted)",
                      yaxis_title="Count")
    fig.update_traces(marker_line_width=0, opacity=0.8)
    return fig


def shap_figure(contributions):
    df = contributions.sort_values("Contribution").copy()
    df["Direction"] = df["Contribution"].apply(lambda c: "Increases LogS" if c >= 0 else "Decreases LogS")
    fig = px.bar(df, x="Contribution", y="Feature", orientation="h", color="Direction",
                 color_discrete_map={"Increases LogS": C_ACCENT, "Decreases LogS": C_DANGER},
                 title="SHAP Feature Contributions")
    fig.add_vline(x=0, line_width=1, line_color=C_BORDER)
    fig.update_layout(**_CHART, height=260,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""))
    fig.update_traces(marker_line_width=0)
    return fig


def benchmark_bar(payload):
    rows = sorted(payload["results"].items(), key=lambda kv: kv[1]["r2"])
    names = [n for n, _ in rows]
    r2 = [r["r2"] for _, r in rows]
    cols = [C_ACCENT if r["family"].startswith("Graph") else C_MUTED for _, r in rows]
    fig = go.Figure(go.Bar(x=r2, y=names, orientation="h", marker_color=cols,
                           text=[f"{v:.3f}" for v in r2], textposition="outside",
                           textfont=dict(color=C_TEXT, size=10)))
    fig.update_layout(**_CHART, height=300, title="Test R² by Model", xaxis_title="Test R²",
                      xaxis_range=[0, 1.0])
    return fig


@st.cache_data
def correlation_figure():
    df = load_and_featurize()
    cols = FEATURE_COLUMNS + [TARGET_COL]
    corr = df[cols].corr().values
    labels = FEATURE_COLUMNS + ["LogS"]
    fig = px.imshow(corr, x=labels, y=labels, zmin=-1, zmax=1, text_auto=".2f",
                    color_continuous_scale=[[0, C_DANGER], [0.5, "#243049"], [1, C_ACCENT]],
                    title="Descriptor Correlation Matrix", aspect="auto")
    fig.update_layout(**_CHART, height=320, coloraxis_showscale=True)
    fig.update_traces(textfont_size=9)
    return fig


@st.cache_data(show_spinner="Computing learning curve…")
def learning_curve_figure():
    from sklearn.model_selection import learning_curve
    from src.models import MODEL_SPECS, _build_pipeline
    df = load_and_featurize()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COL]
    sizes, train_s, val_s = learning_curve(
        _build_pipeline(MODEL_SPECS["Random Forest"]), X, y, cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=train_s.mean(1), mode="lines+markers", name="Training R²",
                             line=dict(color=C_ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=sizes, y=val_s.mean(1), mode="lines+markers", name="CV R²",
                             line=dict(color=C_WARNING, width=2)))
    fig.update_layout(**_CHART, height=320, title="Learning Curve (Random Forest)",
                      xaxis_title="Training examples", yaxis_title="R²",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""))
    return fig


# ── HTML renderers ──────────────────────────────────────────────────────────────
def _pil_to_b64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_kpis(items: list[dict]) -> None:
    cards = "".join(
        f'<div class="kpi">'
        f'<span class="k">{it["k"]}</span>'
        f'<span class="v" style="color:{it.get("color", C_TEXT)}">{it["v"]}</span>'
        f'<span class="s">{it["s"]}</span></div>'
        for it in items
    )
    st.markdown(f'<div class="grid kpi-grid">{cards}</div>', unsafe_allow_html=True)


def render_leaderboard(rows: list[dict]) -> None:
    best = rows[0]["name"]
    html = []
    for r in rows:
        dot = _status_color(r["r2"])
        badge = '<span class="bestbadge">BEST</span>' if r["name"] == best else ""
        html.append(
            f'<div class="lead {"best" if r["name"]==best else ""}">'
            f'<div class="row1"><span class="nm"><span class="dot" style="background:{dot}"></span>{r["name"]}</span>{badge}</div>'
            f'<div class="row2">'
            f'<span class="met"><span class="ml">R²</span><span class="mv">{r["r2"]:.3f}</span></span>'
            f'<span class="met"><span class="ml">MAE</span><span class="mv">{r["mae"]:.3f}</span></span>'
            f'<span class="met"><span class="ml">Type</span><span class="mv" style="color:{C_MUTED};font-size:0.62rem;">'
            f'{"GNN" if r["family"].startswith("Graph") else "ML"}</span></span>'
            f'</div></div>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def render_property_cards(profile, halfwidth, means) -> None:
    cards = []
    for p in profile.properties:
        color = SENTIMENT_COLOR.get(p.sentiment, C_ACCENT)
        unit = f"<small> {p.unit}</small>" if p.unit else ""
        if p.name == "Aqueous Solubility (LogS)":
            trend = f'<span class="t">± {halfwidth:.2f} (95% CI)</span>'
        elif p.name in means:
            m = means[p.name]
            arrow = "▲ above" if p.value > m else "▼ below"
            trend = f'<span class="t">{arrow} set avg ({m:.1f})</span>'
        else:
            trend = f'<span class="t">{p.detail}</span>'
        cards.append(
            f'<div class="prop" style="--c:{color}">'
            f'<span class="k">{p.name}</span>'
            f'<span class="v">{p.display}{unit}</span>'
            f'<span class="c">{p.category}</span>{trend}</div>'
        )
    st.markdown(f'<div class="grid prop-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_mol_3d(smiles: str, height: int = 300) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
            return False
        AllChem.MMFFOptimizeMolecule(m)
        molblock = Chem.MolToMolBlock(m)
    except Exception:
        return False
    html = (
        '<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>'
        f'<div id="vw" style="height:{height}px;width:100%;position:relative;border-radius:10px;'
        'overflow:hidden;background:#0B1220;border:1px solid #1c2740;"></div>'
        '<script>'
        'function draw(){'
        'if(typeof $3Dmol==="undefined"){setTimeout(draw,120);return;}'
        'let el=document.getElementById("vw");'
        'let v=$3Dmol.createViewer(el,{backgroundColor:0x0B1220});'
        f'v.addModel(`{molblock}`,"mol");'
        'v.setStyle({},{stick:{radius:0.13,colorscheme:"cyanCarbon"},sphere:{scale:0.22}});'
        'v.zoomTo();v.render();}draw();'
        '</script>'
    )
    components.html(html, height=height + 4)
    return True


def structure_stats(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Bonds": mol.GetNumBonds(),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "Arom. rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }


# ── Pages ───────────────────────────────────────────────────────────────────────
def page_dashboard(bundle, benchmark, rows):
    best = rows[0]
    avg_mae = float(np.mean([r["mae"] for r in rows]))
    df = load_and_featurize()
    sample = molecular_property_profile("CC(=O)Oc1ccccc1C(=O)O", logs=0.0)
    n_props = len(sample.properties) if sample else 8

    render_kpis([
        {"k": "Best Model R²", "v": f"{best['r2']:.3f}", "s": best["name"], "color": C_SUCCESS},
        {"k": "Average MAE", "v": f"{avg_mae:.3f}", "s": "across all models"},
        {"k": "Dataset Size", "v": f"{len(df):,}", "s": "Delaney ESOL compounds"},
        {"k": "Predictions", "v": f"{st.session_state.pred_count}", "s": "this session", "color": C_ACCENT},
        {"k": "Descriptors", "v": f"{len(FEATURE_COLUMNS)}", "s": "RDKit features"},
        {"k": "Properties", "v": f"{n_props}", "s": "per molecule"},
    ])

    st.markdown('<div class="sec">Model Comparison</div>', unsafe_allow_html=True)
    with st.container(border=True):
        if benchmark:
            st.dataframe(benchmark_table(benchmark), hide_index=True, height=240)
        else:
            st.dataframe(metrics_dataframe(bundle.results), hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            if benchmark:
                st.plotly_chart(benchmark_bar(benchmark), config=PLOTLY_CONFIG)
            else:
                best_mr = max(bundle.results, key=lambda r: r.r2)
                st.plotly_chart(residuals_figure(best_mr.y_test, best_mr.y_pred), config=PLOTLY_CONFIG)
    with c2:
        with st.container(border=True):
            st.plotly_chart(correlation_figure(), config=PLOTLY_CONFIG)

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.plotly_chart(learning_curve_figure(), config=PLOTLY_CONFIG)
    with c4:
        with st.container(border=True):
            best_mr = max(bundle.results, key=lambda r: r.r2)
            st.plotly_chart(residuals_figure(best_mr.y_test, best_mr.y_pred,
                                             title=f"Residuals — {best_mr.name}"), config=PLOTLY_CONFIG)

    st.markdown('<div class="sec">Research Notes</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.session_state.notes = st.text_area(
            "notes", value=st.session_state.notes, label_visibility="collapsed",
            placeholder="Session research notes — observations, hypotheses, follow-ups…", height=90)


def page_live(bundle, rows, means):
    classical = [r["name"] for r in rows if not r["family"].startswith("Graph")]
    if not classical:
        classical = [m.name for m in bundle.results]

    left, center, right = st.columns([0.9, 1.15, 1.25], gap="small")

    with left:
        with st.container(border=True):
            st.markdown('<span class="eyebrow accent">Input</span>', unsafe_allow_html=True)
            smiles = st.text_input("SMILES", value="CC(=O)Oc1ccccc1C(=O)O",
                                   label_visibility="collapsed",
                                   placeholder="SMILES, e.g. aspirin").strip()
            model_name = st.selectbox("Model", classical, key="live_model_select")
            st.session_state.active_model = model_name
            run = st.button("▶  Run Prediction", type="primary", use_container_width=True)
            st.markdown('<span class="subcap" style="margin-top:0.6rem;">Live inference uses the '
                        "classical regressors; all six models are compared under Model Benchmarks."
                        "</span>", unsafe_allow_html=True)

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if smiles and mol is None:
        with center:
            st.error("Invalid SMILES — please check your input.")
        return
    if not smiles:
        return

    model = get_model_by_name(bundle.results, model_name)
    descriptors = smiles_to_descriptors(smiles)
    feature_row = pd.DataFrame([descriptors])
    predicted = float(model.pipeline.predict(feature_row)[0])
    halfwidth = prediction_interval_halfwidth(model)
    profile = molecular_property_profile(smiles, logs=predicted)

    if run:
        st.session_state.pred_count += 1
        st.session_state.history.insert(0, {
            "Time": dt.datetime.now().strftime("%H:%M:%S"), "SMILES": smiles,
            "Model": model_name, "LogS": round(predicted, 3),
        })
        st.session_state.history = st.session_state.history[:25]

    with center:
        with st.container(border=True):
            st.markdown('<span class="eyebrow accent">3D Structure · drag to rotate · scroll to zoom</span>',
                        unsafe_allow_html=True)
            if not render_mol_3d(smiles, height=270):
                img = Draw.MolToImage(mol, size=(360, 260))
                st.markdown(f'<img src="data:image/png;base64,{_pil_to_b64(img)}" '
                            f'style="width:100%;border-radius:10px;border:1px solid {C_BORDER};background:#f6f9ff;">',
                            unsafe_allow_html=True)
            stats = structure_stats(smiles)
            if stats:
                chips = "".join(
                    f'<span class="badge" style="margin-right:0.3rem;">{k}: '
                    f'<b style="color:{C_TEXT};">{v}</b></span>' for k, v in stats.items())
                st.markdown(f'<div style="margin-top:0.6rem;">{chips}</div>', unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown('<span class="eyebrow accent">Predicted Properties</span>', unsafe_allow_html=True)
            render_property_cards(profile, halfwidth, means)

    st.markdown('<div class="sec">Analytics</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        with st.container(border=True):
            st.plotly_chart(feature_importance_figure(model, bundle.feature_columns), config=PLOTLY_CONFIG)
    with a2:
        with st.container(border=True):
            contributions = explain_prediction(model, feature_row)
            if contributions is not None:
                st.plotly_chart(shap_figure(contributions), config=PLOTLY_CONFIG)
                expl = natural_language_explanation(contributions)
                if expl:
                    st.markdown(f'<div class="xai"><b>In plain terms:</b> {expl}</div>', unsafe_allow_html=True)
            else:
                st.info("SHAP explanations are available for tree-based models.")

    b1, b2 = st.columns(2)
    with b1:
        with st.container(border=True):
            st.markdown('<span class="eyebrow">Similar Training Compounds</span>', unsafe_allow_html=True)
            sim = find_similar_compounds(smiles, bundle.smiles_train, bundle.y_train, n_neighbors=6)
            if sim.empty:
                st.info("No close analogues in the training set — treat the prediction with caution.")
            else:
                sim = sim.copy()
                sim["Similarity"] = sim["Similarity"].round(3)
                sim["Experimental LogS"] = sim["Experimental LogS"].round(3)
                st.dataframe(sim, hide_index=True, height=240)
    with b2:
        with st.container(border=True):
            st.markdown('<span class="eyebrow">Descriptor Breakdown</span>', unsafe_allow_html=True)
            dd = pd.DataFrame([{"Descriptor": k, "Value": round(v, 4)} for k, v in descriptors.items()])
            st.dataframe(dd, hide_index=True, height=240)


def page_batch(bundle, rows):
    st.markdown('<div class="sec">Batch Screening</div>', unsafe_allow_html=True)
    st.markdown('<span class="subcap">Upload a CSV of SMILES to screen many compounds at once.</span>',
                unsafe_allow_html=True)
    classical = [r["name"] for r in rows if not r["family"].startswith("Graph")] or [m.name for m in bundle.results]
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
        with c2:
            model_name = st.selectbox("Model", classical, key="batch_model")
        if uploaded is not None:
            data = pd.read_csv(uploaded)
            st.caption(f"{len(data):,} rows · {len(data.columns)} columns")
            cand = [c for c in data.columns if "smiles" in c.lower()]
            idx = data.columns.tolist().index(cand[0]) if cand else 0
            col = st.selectbox("SMILES column", data.columns.tolist(), index=idx)
            if st.button("Run Screening", type="primary"):
                model = get_model_by_name(bundle.results, model_name)
                out = []
                with st.spinner(f"Screening {len(data):,} compounds…"):
                    for smi in data[col]:
                        smi = str(smi).strip()
                        desc = smiles_to_descriptors(smi)
                        if desc is None:
                            out.append({"SMILES": smi, "Valid": False, "Predicted LogS": None,
                                        "Interpretation": "Invalid SMILES"})
                        else:
                            p = float(model.pipeline.predict(pd.DataFrame([desc]))[0])
                            out.append({"SMILES": smi, "Valid": True, "Predicted LogS": round(p, 3),
                                        "Interpretation": interpret_logs(p)})
                res = pd.DataFrame(out)
                st.success(f"{int(res['Valid'].sum()):,} / {len(res):,} compounds screened.")
                st.dataframe(res, hide_index=True, height=320)
                st.download_button("⬇ Download results (CSV)", res.to_csv(index=False).encode(),
                                   "screening_results.csv", "text/csv")


def page_benchmarks(bundle, benchmark):
    st.markdown('<div class="sec">Model Benchmarks</div>', unsafe_allow_html=True)
    if not benchmark:
        st.info("Run `python -m src.graph_training` to generate the GCN/GAT/MPNN benchmark.")
        with st.container(border=True):
            st.dataframe(metrics_dataframe(bundle.results), hide_index=True)
        return
    st.markdown(
        f'<span class="subcap">Six models on an identical '
        f'{int((1-benchmark["test_size"])*100)}/{int(benchmark["test_size"]*100)} split with '
        f'{benchmark["cv_folds"]}-fold cross-validation.</span>', unsafe_allow_html=True)
    with st.container(border=True):
        st.dataframe(benchmark_table(benchmark), hide_index=True, height=250)

    with st.container(border=True):
        st.markdown('<span class="eyebrow accent">Diagnostics</span>', unsafe_allow_html=True)
        choice = st.selectbox("Model", list(benchmark["results"].keys()), key="bench_diag")
        try:
            mr = get_model_by_name(bundle.results, choice)
            yt, yp, sm = np.asarray(mr.y_test), np.asarray(mr.y_pred), bundle.smiles_test
        except KeyError:
            r = benchmark["results"][choice]
            yt, yp, sm = np.asarray(r["y_true"]), np.asarray(r["y_pred"]), None
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(parity_plot(yt, yp, smiles=sm, title=f"Parity — {choice}"), config=PLOTLY_CONFIG)
        with d2:
            st.plotly_chart(residuals_figure(yt, yp, title=f"Residuals — {choice}"), config=PLOTLY_CONFIG)


def page_search(bundle):
    st.markdown('<div class="sec">Molecular Search</div>', unsafe_allow_html=True)
    st.markdown('<span class="subcap">Find the nearest dataset analogues to a query molecule by '
                "Morgan-fingerprint Tanimoto similarity.</span>", unsafe_allow_html=True)
    df = load_and_featurize()
    with st.container(border=True):
        q = st.text_input("Query SMILES", value="CC(=O)Oc1ccccc1C(=O)O").strip()
        n = st.slider("Neighbours", 3, 20, 8)
    if not q:
        return
    if Chem.MolFromSmiles(q) is None:
        st.error("Invalid SMILES.")
        return
    sim = find_similar_compounds(q, df["smiles"], df[TARGET_COL], n_neighbors=n, min_similarity=0.1)
    cq, ct = st.columns([0.8, 1.6])
    with cq:
        with st.container(border=True):
            st.markdown('<span class="eyebrow accent">Query</span>', unsafe_allow_html=True)
            img = Draw.MolToImage(Chem.MolFromSmiles(q), size=(300, 220))
            st.markdown(f'<img src="data:image/png;base64,{_pil_to_b64(img)}" '
                        f'style="width:100%;border-radius:10px;border:1px solid {C_BORDER};background:#f6f9ff;">',
                        unsafe_allow_html=True)
    with ct:
        with st.container(border=True):
            st.markdown('<span class="eyebrow accent">Nearest Analogues</span>', unsafe_allow_html=True)
            if sim.empty:
                st.info("No analogues above the similarity threshold.")
            else:
                sim = sim.copy()
                sim["Similarity"] = sim["Similarity"].round(3)
                sim["Experimental LogS"] = sim["Experimental LogS"].round(3)
                st.dataframe(sim, hide_index=True, height=320)


def page_reports(benchmark):
    st.markdown('<div class="sec">Research Reports</div>', unsafe_allow_html=True)
    if benchmark is None:
        st.info("No benchmark found. Run `python -m src.graph_training` to enable report export.")
        return
    best_name = max(benchmark["results"], key=lambda k: benchmark["results"][k]["r2"])
    best = benchmark["results"][best_name]
    secs = ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion", "References"]
    chips = "".join(f'<span class="badge" style="margin:0.15rem 0.3rem 0.15rem 0;">{s}</span>' for s in secs)
    with st.container(border=True):
        st.markdown(
            f'<span class="eyebrow accent">Preview</span>'
            f'<div style="font-size:1.05rem;font-weight:600;color:{C_TEXT};margin-bottom:0.4rem;">'
            f'Benchmarking Classical ML against Graph Neural Networks for Aqueous Solubility Prediction</div>'
            f'<span class="subcap">A study of six models on {benchmark["n_compounds"]:,} Delaney ESOL '
            f'compounds. Best model <b style="color:{C_ACCENT}">{best_name}</b> reaches held-out R² '
            f'<b style="color:{C_ACCENT}">{best["r2"]:.3f}</b> (RMSE {best["rmse"]:.3f}, '
            f'MAE {best["mae"]:.3f}).</span>'
            f'<div style="margin-top:0.5rem;">{chips}</div>', unsafe_allow_html=True)
        pdf = cached_report(benchmark, BENCHMARK_PATH.stat().st_mtime)
        st.download_button("⬇ Download research report (PDF)", pdf,
                           "solubility_benchmark_report.pdf", "application/pdf", type="primary")
        st.caption(f"{len(pdf)/1024:.0f} KB · 2 figures · generated from the live benchmark")

    if st.session_state.history:
        st.markdown('<div class="sec">Experiment History</div>', unsafe_allow_html=True)
        with st.container(border=True):
            hist = pd.DataFrame(st.session_state.history)
            st.dataframe(hist, hide_index=True, height=240)
            st.download_button("⬇ Export history (CSV)", hist.to_csv(index=False).encode(),
                               "experiment_history.csv", "text/csv")


def page_settings(bundle, benchmark):
    st.markdown('<div class="sec">Settings & About</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f'<span class="eyebrow accent">Platform</span>'
            f'<span class="subcap">Computational chemistry platform for aqueous solubility (LogS) '
            f'prediction on the Delaney ESOL dataset. Classical ML (Ridge, Random Forest, Gradient '
            f'Boosting) is benchmarked against graph neural networks (GCN, GAT, MPNN), with '
            f'multi-property profiling, SHAP explainability, and automated PDF reporting.</span>',
            unsafe_allow_html=True)
        swatches = "".join(
            f'<span class="badge" style="border-color:{c};color:{c};margin-right:0.3rem;">{name}</span>'
            for name, c in [("Accent", C_ACCENT), ("Success", C_SUCCESS),
                            ("Warning", C_WARNING), ("Danger", C_DANGER)])
        st.markdown(f'<div style="margin-top:0.4rem;">{swatches}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="eyebrow">Reproducibility</span>', unsafe_allow_html=True)
        st.markdown('<span class="subcap">All stochastic operations use random_state=42. GNNs and '
                    "classical models share the same train/test split for a fair comparison.</span>",
                    unsafe_allow_html=True)
        if st.button("Clear session history & notes", type="secondary"):
            st.session_state.history = []
            st.session_state.notes = ""
            st.session_state.pred_count = 0
            st.rerun()


# ── App shell ───────────────────────────────────────────────────────────────────
def render_header(bundle, benchmark, rows, df_len):
    best = rows[0]["name"]
    active = st.session_state.get("active_model", best)
    with st.container(border=True):
        c_brand, c_export = st.columns([5, 1])
        with c_brand:
            st.markdown(
                '<div class="appbar">'
                '<span class="brand">🧪 Molar Solubility Predictor</span>'
                '<span class="badge live">● Delaney ESOL</span>'
                f'<span class="stat"><span class="k">Compounds</span><span class="v">{df_len:,}</span></span>'
                f'<span class="stat"><span class="k">Active model</span><span class="v accent">{active}</span></span>'
                f'<span class="stat"><span class="k">Best R²</span><span class="v">{rows[0]["r2"]:.3f}</span></span>'
                f'<span class="stat"><span class="k">Predictions</span><span class="v">{st.session_state.pred_count}</span></span>'
                '</div>', unsafe_allow_html=True)
        with c_export:
            if benchmark:
                csv = benchmark_table(benchmark).to_csv(index=False).encode()
            else:
                csv = metrics_dataframe(bundle.results).to_csv(index=False).encode()
            st.download_button("⬇ Export", csv, "model_benchmark.csv", "text/csv",
                               use_container_width=True)


def main():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("notes", "")
    st.session_state.setdefault("pred_count", 0)

    bundle = get_training_bundle()
    benchmark = load_benchmark()
    rows = model_rows(bundle, benchmark)
    means = dataset_means()
    df_len = len(load_and_featurize())
    st.session_state.setdefault("active_model", rows[0]["name"])

    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.5rem;padding:0 0.3rem 0.4rem;">'
            '<span style="font-size:1.15rem;">🧪</span>'
            '<span style="font-weight:700;font-size:0.95rem;color:#E6EDF6;letter-spacing:-0.02em;">SolubilityLab</span>'
            '</div>', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow" style="padding:0 0.3rem;">Navigation</span>', unsafe_allow_html=True)
        choice = st.radio("nav", NAV, label_visibility="collapsed",
                          format_func=lambda x: f"{NAV_ICON[x]}   {x}")
        st.divider()
        st.markdown('<span class="eyebrow accent" style="padding:0 0.3rem;">Model Performance</span>',
                    unsafe_allow_html=True)
        render_leaderboard(rows)

    render_header(bundle, benchmark, rows, df_len)

    if choice == "Dashboard":
        page_dashboard(bundle, benchmark, rows)
    elif choice == "Live Prediction":
        page_live(bundle, rows, means)
    elif choice == "Batch Screening":
        page_batch(bundle, rows)
    elif choice == "Model Benchmarks":
        page_benchmarks(bundle, benchmark)
    elif choice == "Molecular Search":
        page_search(bundle)
    elif choice == "Research Reports":
        page_reports(benchmark)
    elif choice == "Settings":
        page_settings(bundle, benchmark)


if __name__ == "__main__":
    main()
