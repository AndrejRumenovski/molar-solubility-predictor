"""Streamlit dashboard for aqueous solubility (LogS) prediction."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

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

RANDOM_FOREST = "Random Forest"

# ── Palette (kept in sync with .streamlit/config.toml) ───────────────────────
C_ACCENT   = "#00e5c0"
C_POSITIVE = "#3fb950"
C_WARNING  = "#d29922"
C_NEGATIVE = "#f85149"
C_BG       = "#07090f"
C_SURFACE  = "#111827"
C_SURFACE2 = "#1c2333"
C_BORDER   = "#1e293b"
C_TEXT     = "#c9d1d9"
C_MUTED    = "#6e7681"

st.set_page_config(
    page_title="Molar Solubility Predictor",
    page_icon="🧪",
    layout="wide",
)

# ── Hex-ring SVG texture (benzene motif) used in the prediction card ─────────
_HEX_SVG = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'"
    " width='56' height='100'%3E%3Cpath d='M28 66L0 50V16L28 0l28 16v34L28"
    " 66zm0-6l22-12.7V22.7L28 10 6 22.7v24.6L28 60z' fill='%23ffffff'"
    " fill-rule='evenodd'/%3E%3C/svg%3E\")"
)

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --accent:   {C_ACCENT};
    --dim:      rgba(0,229,192,0.10);
    --positive: {C_POSITIVE};
    --warning:  {C_WARNING};
    --negative: {C_NEGATIVE};
    --bg:       {C_BG};
    --surface:  {C_SURFACE};
    --surface2: {C_SURFACE2};
    --border:   {C_BORDER};
    --text:     {C_TEXT};
    --muted:    {C_MUTED};
}}

/* ── Global ────────────────────────────────────────────────────────────── */
html, body, .stApp {{
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg) !important;
}}
h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.025em !important;
}}
p, label, div, span, li, td, th {{
    font-family: 'Space Grotesk', sans-serif !important;
}}

/* ── Chrome ────────────────────────────────────────────────────────────── */
#MainMenu, footer {{ visibility: hidden !important; }}
[data-testid="stHeader"] {{
    background: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stMainBlockContainer"] {{ background-color: var(--bg) !important; }}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebarContent"] {{ padding-top: 1.5rem !important; }}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}}
button[data-baseweb="tab"] {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: var(--muted) !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 0 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.2s !important;
}}
button[aria-selected="true"][data-baseweb="tab"] {{
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}}
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"] {{ display: none !important; }}

/* ── Text input (SMILES string) ────────────────────────────────────────── */
[data-testid="stTextInput"] input {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.875rem !important;
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--dim) !important;
}}

/* ── Selectbox ─────────────────────────────────────────────────────────── */
[data-baseweb="select"] > div:first-child {{
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
}}

/* ── Primary button ────────────────────────────────────────────────────── */
button[kind="primary"] {{
    background-color: var(--accent) !important;
    color: {C_BG} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    transition: opacity 0.2s !important;
}}
button[kind="primary"]:hover {{ opacity: 0.85 !important; }}

/* ── File uploader ─────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface2) !important;
}}

/* ── Divider ───────────────────────────────────────────────────────────── */
hr {{ border-color: var(--border) !important; }}

/* ── Model card (sidebar) ──────────────────────────────────────────────── */
.model-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 6px;
    padding: 0.8rem 0.9rem;
    margin-bottom: 0.5rem;
}}
.model-card-name {{
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin: 0 0 0.55rem;
    display: block;
}}
.model-card-row {{ display: flex; gap: 1.5rem; }}
.model-metric {{ display: flex; flex-direction: column; }}
.model-metric-label {{
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.1rem;
}}
.model-metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--accent);
    line-height: 1;
}}

/* ── Section labels ────────────────────────────────────────────────────── */
.eyebrow {{
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    display: block;
    margin-bottom: 0.35rem;
}}
.sub-caption {{
    font-size: 0.78rem;
    color: var(--muted);
    display: block;
    margin-bottom: 0.6rem;
    line-height: 1.45;
}}

/* ── Prediction box  ── the signature element ──────────────────────────── */
@keyframes phosphor-pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0   var(--pred-glow), 0 0 16px var(--pred-glow); }}
    50%       {{ box-shadow: 0 0 0 3px var(--pred-glow), 0 0 32px var(--pred-glow); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .pred-box {{ animation: none !important; }}
}}

.pred-box {{
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin-top: 0.75rem;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--pred-border, var(--accent));
    background: var(--pred-bg, rgba(0,229,192,0.04));
    animation: phosphor-pulse 4s ease-in-out infinite;
}}
.pred-box::before {{
    content: '';
    position: absolute;
    inset: 0;
    background-image: {_HEX_SVG};
    background-size: 56px 100px;
    opacity: 0.035;
    pointer-events: none;
}}
.pred-eyebrow {{
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    display: block;
    margin-bottom: 0.3rem;
}}
.pred-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 500;
    line-height: 1;
    display: block;
    color: var(--pred-color, var(--accent));
}}
.pred-interval {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    display: block;
    margin-top: 0.2rem;
}}
.pred-hr {{
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 0.9rem 0;
}}
.pred-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--pred-color, var(--accent));
    background: var(--pred-badge-bg, rgba(0,229,192,0.14));
    border: 1px solid var(--pred-color, var(--accent));
    border-radius: 999px;
    padding: 0.5rem 1.1rem;
}}
.pred-badge-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--pred-color, var(--accent));
    box-shadow: 0 0 10px var(--pred-color, var(--accent));
    flex-shrink: 0;
}}
.pred-desc {{
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.55;
    margin-top: 0.7rem;
}}

/* ── Property profile cards (glassmorphism grid) ───────────────────────── */
.prop-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.7rem;
    margin-top: 0.4rem;
}}
.prop-card {{
    background: rgba(28,35,51,0.45);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-left: 3px solid var(--prop-color, var(--accent));
    border-radius: 8px;
    padding: 0.85rem 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}}
.prop-card:hover {{
    transform: translateY(-2px);
    border-color: var(--prop-color, var(--accent));
}}
.prop-name {{
    font-size: 0.58rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    display: block;
    margin-bottom: 0.45rem;
}}
.prop-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    line-height: 1;
    color: var(--prop-color, var(--text));
    display: block;
}}
.prop-cat {{
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--prop-color, var(--muted));
    display: block;
    margin-top: 0.4rem;
}}
.prop-detail {{
    font-size: 0.6rem;
    color: var(--muted);
    display: block;
    margin-top: 0.3rem;
    line-height: 1.4;
}}

/* ── XAI natural-language callout ───────────────────────────────────────── */
.xai-note {{
    background: var(--dim);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-top: 0.6rem;
    font-size: 0.85rem;
    line-height: 1.55;
    color: var(--text);
}}
.xai-note strong {{ color: var(--accent); font-weight: 600; }}

/* ── How It Works ──────────────────────────────────────────────────────── */
.hiw {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 2rem 2.5rem;
    margin-top: 2.5rem;
}}
.hiw-eyebrow {{
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent);
    display: block;
    margin-bottom: 0.5rem;
}}
.hiw h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: var(--text) !important;
    margin: 0 0 1.25rem !important;
}}
.hiw p {{
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--muted);
    margin-bottom: 0.75rem;
}}
.hiw strong {{ color: var(--text); font-weight: 500; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Plotly base layout ────────────────────────────────────────────────────────
_CHART = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(17,24,39,0.5)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color=C_MUTED, size=11),
    margin=dict(l=0, r=0, t=40, b=0),
)


# ── Cached data / training ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Delaney ESOL dataset…")
def load_and_featurize() -> pd.DataFrame:
    raw = load_dataset()
    return featurize_dataframe(raw)


@st.cache_resource(show_spinner="Training models…")
def get_training_bundle():
    df = load_and_featurize()
    return train_models(df)


BENCHMARK_PATH = Path(__file__).resolve().parent / "models" / "gnn_benchmark.json"


@st.cache_data
def load_benchmark() -> dict | None:
    """Load the GNN-vs-classical benchmark artifact, if it has been generated."""
    if BENCHMARK_PATH.exists():
        return json.loads(BENCHMARK_PATH.read_text())
    return None


def benchmark_table(payload: dict) -> pd.DataFrame:
    rows = [
        {
            "Model": name,
            "Family": r["family"],
            "R²": round(r["r2"], 4),
            "RMSE": round(r["rmse"], 4),
            "MAE": round(r["mae"], 4),
            "CV R² (mean ± std)": f"{r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}",
        }
        for name, r in payload["results"].items()
    ]
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _logs_style(logs: float) -> tuple[str, str, str, str, str]:
    """(border, background, text_color, glow_rgba, badge_bg) keyed to solubility class."""
    if logs > 0:
        return C_POSITIVE, "rgba(63,185,80,0.06)",  C_POSITIVE, "rgba(63,185,80,0.10)",  "rgba(63,185,80,0.16)"
    if logs > -4:
        return C_WARNING,  "rgba(210,153,34,0.06)", C_WARNING,  "rgba(210,153,34,0.10)", "rgba(210,153,34,0.16)"
    return     C_NEGATIVE, "rgba(248,81,73,0.06)",  C_NEGATIVE, "rgba(248,81,73,0.10)",  "rgba(248,81,73,0.16)"


def _pil_to_b64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_SENTIMENT_COLOR = {
    "good": C_POSITIVE,
    "neutral": C_ACCENT,
    "warn": C_WARNING,
    "bad": C_NEGATIVE,
}


def render_property_profile(profile) -> None:
    """Render the multi-property profile as a grid of glassmorphism cards."""
    cards = []
    for prop in profile.properties:
        color = _SENTIMENT_COLOR.get(prop.sentiment, C_ACCENT)
        unit = (
            f"<span style='font-size:0.7rem;color:{C_MUTED};'> {prop.unit}</span>"
            if prop.unit else ""
        )
        cards.append(
            f'<div class="prop-card" style="--prop-color:{color};">'
            f'<span class="prop-name">{prop.name}</span>'
            f'<span class="prop-value">{prop.display}{unit}</span>'
            f'<span class="prop-cat">{prop.category}</span>'
            f'<span class="prop-detail">{prop.detail}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="prop-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_impact_metrics(items: list[dict]) -> None:
    """Render a strip of headline metric cards (reuses the property-card grid)."""
    cards = []
    for it in items:
        color = it.get("color", C_ACCENT)
        cards.append(
            f'<div class="prop-card" style="--prop-color:{color};">'
            f'<span class="prop-name">{it["name"]}</span>'
            f'<span class="prop-value">{it["value"]}</span>'
            f'<span class="prop-detail">{it["detail"]}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="prop-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def diagnostics_arrays(name, bundle, payload):
    """Return (y_true, y_pred, smiles_or_None) for any model, classical or GNN."""
    try:
        mr = get_model_by_name(bundle.results, name)
        return np.asarray(mr.y_test), np.asarray(mr.y_pred), bundle.smiles_test
    except KeyError:
        r = payload["results"][name]
        return np.asarray(r["y_true"]), np.asarray(r["y_pred"]), None


def render_model_card(model_result) -> None:
    st.markdown(
        f"""
        <div class="model-card">
            <span class="model-card-name">{model_result.name}</span>
            <div class="model-card-row">
                <div class="model-metric">
                    <span class="model-metric-label">R²</span>
                    <span class="model-metric-value">{model_result.r2:.4f}</span>
                </div>
                <div class="model-metric">
                    <span class="model-metric-label">MAE</span>
                    <span class="model-metric-value">{model_result.mae:.4f}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Chart builders ────────────────────────────────────────────────────────────
def feature_importance_figure(model_result, feature_columns: list[str]):
    model = model_result.pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_)
    else:
        return None

    df = pd.DataFrame(
        {"Feature": feature_columns, "Importance": importances}
    ).sort_values("Importance", ascending=True)

    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        title="Feature Importance",
        color="Importance",
        color_continuous_scale=[[0, C_SURFACE2], [1, C_ACCENT]],
    )
    fig.update_layout(**_CHART, showlegend=False, height=320, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    return fig


def parity_plot(y_true, y_pred, smiles=None, title="Parity Plot — Predicted vs. Experimental LogS"):
    data = {"Experimental LogS": list(y_true), "Predicted LogS": list(y_pred)}
    hover = {"Experimental LogS": ":.3f", "Predicted LogS": ":.3f"}
    if smiles is not None:
        data["SMILES"] = list(smiles)
        hover["SMILES"] = True

    plot_df = pd.DataFrame(data)
    fig = px.scatter(
        plot_df,
        x="Experimental LogS", y="Predicted LogS",
        hover_data=hover, title=title,
        opacity=0.65, color_discrete_sequence=[C_ACCENT],
    )
    lo = min(plot_df["Experimental LogS"].min(), plot_df["Predicted LogS"].min()) - 0.5
    hi = max(plot_df["Experimental LogS"].max(), plot_df["Predicted LogS"].max()) + 0.5
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
        line=dict(color=C_NEGATIVE, dash="dash", width=1),
    ))
    fig.update_layout(**_CHART, height=420)
    return fig


def residuals_figure(y_true, y_pred, title="Residual Distribution"):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig = px.histogram(
        pd.DataFrame({"Residual (actual − predicted)": residuals}),
        x="Residual (actual − predicted)",
        nbins=30, title=title, color_discrete_sequence=[C_ACCENT],
    )
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color=C_NEGATIVE)
    fig.update_layout(**_CHART, height=420)
    fig.update_traces(marker_line_width=0, opacity=0.8)
    return fig


def shap_contribution_figure(contributions: pd.DataFrame):
    plot_df = contributions.sort_values("Contribution").copy()
    plot_df["Direction"] = plot_df["Contribution"].apply(
        lambda c: "Increases LogS" if c >= 0 else "Decreases LogS"
    )
    fig = px.bar(
        plot_df, x="Contribution", y="Feature", orientation="h",
        color="Direction",
        color_discrete_map={"Increases LogS": C_POSITIVE, "Decreases LogS": C_NEGATIVE},
        title="Feature Contributions (SHAP)",
    )
    fig.add_vline(x=0, line_width=1, line_color=C_BORDER)
    fig.update_layout(
        **_CHART, height=310,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def run_batch_predictions(smiles_series: pd.Series, model_result) -> pd.DataFrame:
    rows = []
    for smi in smiles_series:
        smi = str(smi).strip()
        desc = smiles_to_descriptors(smi)
        if desc is None:
            rows.append({
                "SMILES": smi, "Valid": False,
                "Predicted LogS": None,
                "Interpretation": "Invalid / unparseable SMILES",
                **{k: None for k in FEATURE_COLUMNS},
            })
        else:
            feat = pd.DataFrame([desc])
            pred = float(model_result.pipeline.predict(feat)[0])
            rows.append({
                "SMILES": smi, "Valid": True,
                "Predicted LogS": round(pred, 3),
                "Interpretation": interpret_logs(pred),
                **{k: round(v, 4) for k, v in desc.items()},
            })
    return pd.DataFrame(rows)


# ── App ───────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("🧪 Molar Solubility Predictor")
    st.caption(
        "QSPR + graph-neural-network platform · Delaney ESOL dataset · "
        "multi-property profiling · explainable AI"
    )

    bundle      = get_training_bundle()
    rf_result   = get_model_by_name(bundle.results, RANDOM_FOREST)
    metrics_df  = metrics_dataframe(bundle.results)
    model_names = [r.name for r in bundle.results]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<span class="eyebrow" style="color:var(--accent);">Model Performance</span>',
            unsafe_allow_html=True,
        )
        for mr in bundle.results:
            render_model_card(mr)

        st.divider()
        st.markdown('<span class="eyebrow">Feature Importance</span>', unsafe_allow_html=True)
        fi_fig = feature_importance_figure(rf_result, bundle.feature_columns)
        if fi_fig:
            st.plotly_chart(fi_fig)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_predict, tab_batch, tab_dashboard = st.tabs(
        ["Live Inference", "Batch Prediction", "Performance Dashboard"]
    )

    # ── Live Inference ────────────────────────────────────────────────────────
    with tab_predict:
        st.subheader("Live Inference Playground")
        st.markdown(
            "Enter a SMILES string to draw the molecule and predict its aqueous solubility."
        )

        col_input, col_model_sel = st.columns([3, 1])
        with col_input:
            user_smiles = st.text_input(
                "SMILES string",
                value="CC(=O)Oc1ccccc1C(=O)O",
                placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (Aspirin)",
            ).strip()
        with col_model_sel:
            active_model_name = st.selectbox("Model", model_names, key="live_model")

        active_model = get_model_by_name(bundle.results, active_model_name)

        if user_smiles:
            mol = Chem.MolFromSmiles(user_smiles)
            if mol is None:
                st.error("Invalid SMILES string — please check your input.")
            else:
                descriptors = smiles_to_descriptors(user_smiles)
                if descriptors is None:
                    st.error("Could not compute descriptors for this molecule.")
                else:
                    feature_row   = pd.DataFrame([descriptors])
                    predicted     = float(active_model.pipeline.predict(feature_row)[0])
                    interpretation = interpret_logs(predicted)
                    halfwidth      = prediction_interval_halfwidth(active_model)
                    border_c, bg_c, text_c, glow_c, badge_c = _logs_style(predicted)

                    # "Highly soluble — readily dissolves in water." → label + description
                    sol_class, _, sol_desc = interpretation.partition("—")
                    sol_class, sol_desc = sol_class.strip(), sol_desc.strip()

                    col_mol, col_pred = st.columns([1, 1])

                    with col_mol:
                        img    = Draw.MolToImage(mol, size=(400, 280))
                        b64    = _pil_to_b64(img)
                        label  = user_smiles[:52] + ("…" if len(user_smiles) > 52 else "")
                        st.markdown(
                            f"""
                            <div style="text-align:center;">
                                <img src="data:image/png;base64,{b64}"
                                     style="width:100%;border-radius:8px;
                                            border:1px solid {C_BORDER};
                                            background:#f8faff;"
                                     alt="2D structure">
                                <div style="font-family:'JetBrains Mono',monospace;
                                            font-size:0.7rem;color:{C_MUTED};
                                            margin-top:0.4rem;word-break:break-all;">
                                    {label}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col_pred:
                        st.markdown(
                            f"""
                            <div class="pred-box"
                                 style="--pred-border:{border_c};
                                        --pred-bg:{bg_c};
                                        --pred-color:{text_c};
                                        --pred-glow:{glow_c};
                                        --pred-badge-bg:{badge_c};">
                                <span class="pred-eyebrow">Predicted LogS</span>
                                <span class="pred-value">{predicted:.3f}</span>
                                <span class="pred-interval">&plusmn;&thinsp;{halfwidth:.3f}&nbsp;(95% CI)</span>
                                <hr class="pred-hr">
                                <div class="pred-badge">
                                    <span class="pred-badge-dot"></span>{sol_class}
                                </div>
                                <div class="pred-desc">{sol_desc}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            '<span class="eyebrow" style="margin-top:1.2rem;display:block;">'
                            "Computed Descriptors</span>",
                            unsafe_allow_html=True,
                        )
                        desc_df = pd.DataFrame(
                            [{"Descriptor": k, "Value": round(v, 4)}
                             for k, v in descriptors.items()]
                        )
                        st.dataframe(desc_df, hide_index=True)

                    # ── Multi-property profile ──────────────────────────────
                    profile = molecular_property_profile(user_smiles, logs=predicted)
                    if profile is not None:
                        st.markdown(
                            '<span class="eyebrow" style="margin-top:1.5rem;display:block;">'
                            "Molecular Property Profile</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<span class="sub-caption">Predicted and structure-derived properties '
                            "spanning solubility, drug-likeness, toxicity risk, and CNS penetration."
                            "</span>",
                            unsafe_allow_html=True,
                        )
                        render_property_profile(profile)

                    st.divider()
                    col_shap, col_similar = st.columns([1, 1])

                    with col_shap:
                        contributions = explain_prediction(active_model, feature_row)
                        if contributions is not None:
                            st.plotly_chart(shap_contribution_figure(contributions))
                            explanation = natural_language_explanation(contributions)
                            if explanation:
                                st.markdown(
                                    f'<div class="xai-note"><strong>In plain terms:</strong> '
                                    f"{explanation}</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info(
                                "SHAP explanations are available for Random Forest and "
                                "Gradient Boosting models only."
                            )

                    with col_similar:
                        st.markdown(
                            '<span class="eyebrow">Most Similar Training Compounds</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<span class="sub-caption">Nearest neighbours by Morgan-fingerprint'
                            " Tanimoto similarity.</span>",
                            unsafe_allow_html=True,
                        )
                        similar = find_similar_compounds(
                            user_smiles, bundle.smiles_train, bundle.y_train, n_neighbors=5,
                        )
                        if similar.empty:
                            st.info(
                                "No training compounds exceed the similarity threshold — "
                                "this molecule lies outside familiar chemical space. "
                                "Treat this prediction with extra caution."
                            )
                        else:
                            disp = similar.copy()
                            disp["Similarity"]        = disp["Similarity"].round(3)
                            disp["Experimental LogS"] = disp["Experimental LogS"].round(3)
                            st.dataframe(disp, hide_index=True)

    # ── Batch Prediction ──────────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Prediction")
        st.markdown(
            "Upload a CSV of SMILES strings, predict solubility for every compound, "
            "and download the results."
        )

        col_upload, col_opts = st.columns([2, 1])
        with col_upload:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
        with col_opts:
            batch_model_name = st.selectbox("Model", model_names, key="batch_model")

        if uploaded is not None:
            batch_input = pd.read_csv(uploaded)
            st.caption(f"{len(batch_input):,} rows · {len(batch_input.columns)} columns")

            candidates  = [c for c in batch_input.columns if "smiles" in c.lower()]
            default_idx = batch_input.columns.tolist().index(candidates[0]) if candidates else 0
            smiles_col  = st.selectbox(
                "SMILES column", batch_input.columns.tolist(), index=default_idx,
            )

            st.markdown('<span class="eyebrow">Preview</span>', unsafe_allow_html=True)
            st.dataframe(batch_input[[smiles_col]].head(5), hide_index=True)

            if st.button("Run Predictions", type="primary"):
                batch_model = get_model_by_name(bundle.results, batch_model_name)
                with st.spinner(f"Predicting {len(batch_input):,} compounds…"):
                    result_df = run_batch_predictions(batch_input[smiles_col], batch_model)

                valid_n = int(result_df["Valid"].sum())
                st.success(
                    f"{valid_n:,} / {len(result_df):,} compounds predicted successfully."
                )
                st.dataframe(result_df, hide_index=True)
                st.download_button(
                    "⬇ Download predictions as CSV",
                    result_df.to_csv(index=False).encode(),
                    file_name="solubility_predictions.csv",
                    mime="text/csv",
                )

    # ── Performance Dashboard ─────────────────────────────────────────────────
    with tab_dashboard:
        benchmark = load_benchmark()

        # ── Research impact metrics ────────────────────────────────────────────
        st.markdown(
            '<span class="eyebrow" style="color:var(--accent);">Research Impact</span>',
            unsafe_allow_html=True,
        )
        if benchmark:
            best_name = max(benchmark["results"], key=lambda k: benchmark["results"][k]["r2"])
            best = benchmark["results"][best_name]
            n_models = len(benchmark["results"])
            render_impact_metrics([
                {"name": "Compounds Analyzed", "value": f"{benchmark['n_compounds']:,}",
                 "detail": "Delaney ESOL benchmark dataset"},
                {"name": "Models Benchmarked", "value": str(n_models),
                 "detail": "Classical ML + graph neural networks"},
                {"name": "Best Model", "value": best_name,
                 "detail": f"{best['family']}", "color": C_POSITIVE},
                {"name": "Best Test R²", "value": f"{best['r2']:.3f}",
                 "detail": f"RMSE {best['rmse']:.3f} log units", "color": C_POSITIVE},
                {"name": "Prediction Error", "value": f"±{best['mae']:.3f}",
                 "detail": "Mean absolute error (best model)"},
            ])
        else:
            best_classical = max(bundle.results, key=lambda r: r.r2)
            render_impact_metrics([
                {"name": "Compounds Analyzed", "value": "1,128",
                 "detail": "Delaney ESOL benchmark dataset"},
                {"name": "Models Trained", "value": str(len(bundle.results)),
                 "detail": "Random Forest · Gradient Boosting · Ridge"},
                {"name": "Best Model", "value": best_classical.name,
                 "detail": "Highest held-out R²", "color": C_POSITIVE},
                {"name": "Best Test R²", "value": f"{best_classical.r2:.3f}",
                 "detail": f"MAE {best_classical.mae:.3f} log units", "color": C_POSITIVE},
            ])

        # ── Benchmark table ────────────────────────────────────────────────────
        st.markdown(
            '<span class="eyebrow" style="margin-top:1.75rem;display:block;">Model Benchmark</span>',
            unsafe_allow_html=True,
        )
        if benchmark:
            st.markdown(
                '<span class="sub-caption">Classical models and graph neural networks on an '
                f"identical {int((1-benchmark['test_size'])*100)}/{int(benchmark['test_size']*100)} "
                f"split, with {benchmark['cv_folds']}-fold cross-validation.</span>",
                unsafe_allow_html=True,
            )
            st.dataframe(benchmark_table(benchmark), hide_index=True)
            all_models = list(benchmark["results"].keys())
        else:
            st.markdown(
                '<span class="sub-caption">Showing classical models only. Run '
                "<code>python -m src.graph_training</code> to add the GCN / GAT / MPNN "
                "graph-neural-network benchmark.</span>",
                unsafe_allow_html=True,
            )
            st.dataframe(metrics_df, hide_index=True)
            all_models = model_names

        # ── Diagnostics ────────────────────────────────────────────────────────
        st.markdown(
            '<span class="eyebrow" style="margin-top:1.75rem;display:block;">Diagnostics</span>',
            unsafe_allow_html=True,
        )
        dash_model = st.selectbox("Select model", all_models, index=0, key="dashboard_model")
        y_true, y_pred, smiles = diagnostics_arrays(dash_model, bundle, benchmark)

        col_parity, col_resid = st.columns([1, 1])
        with col_parity:
            st.plotly_chart(parity_plot(y_true, y_pred, smiles=smiles,
                                        title=f"Parity Plot — {dash_model}"))
        with col_resid:
            st.plotly_chart(residuals_figure(y_true, y_pred,
                                             title=f"Residuals — {dash_model}"))

    # ── How It Works ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hiw">
            <span class="hiw-eyebrow">Background</span>
            <h3>How It Works</h3>
            <p>
                <strong>Quantitative Structure-Property Relationship (QSPR)</strong> models link
                molecular structure to physical properties without wet-lab experiments. This app
                uses the Delaney ESOL dataset — 1,128 compounds with experimentally measured
                aqueous solubility (LogS).
            </p>
            <p>
                <strong>Molecular descriptors</strong> are numerical features extracted from
                SMILES strings via <strong>RDKit</strong>: octanol-water partition coefficient
                (LogP), molecular weight, rotatable bonds, aromatic proportion, and topological
                polar surface area (TPSA). These capture hydrophobicity, size, flexibility,
                rigidity, and hydrogen-bonding capacity — the key drivers of aqueous solubility.
            </p>
            <p>
                Classical regression models (Random Forest, Gradient Boosting, Ridge) are
                benchmarked against <strong>graph neural networks</strong> (GCN, GAT, MPNN) that
                learn directly from the molecular graph — atoms as nodes, bonds as edges — on an
                identical train/test split with cross-validation.
            </p>
            <p>
                Every prediction is accompanied by a <strong>multi-property profile</strong>
                (drug-likeness, toxicity risk, blood-brain-barrier penetration) and
                <strong>explainable-AI</strong> attributions showing which descriptors drove the
                result. Enter any valid SMILES above to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
