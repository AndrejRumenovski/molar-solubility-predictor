"""Streamlit dashboard for aqueous solubility (LogS) prediction."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.data_loader import TARGET_COL, load_dataset
from src.featurizer import FEATURE_COLUMNS, featurize_dataframe, interpret_logs, smiles_to_descriptors
from src.models import get_model_by_name, metrics_dataframe, train_models

RANDOM_FOREST = "Random Forest"

st.set_page_config(
    page_title="Molar Solubility Predictor",
    page_icon="🧪",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.85;
        font-weight: 400;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .prediction-box {
        background: #f0f7ff;
        border-left: 4px solid #2d6a9f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .how-it-works {
        background: #fafafa;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading Delaney ESOL dataset…")
def load_and_featurize() -> pd.DataFrame:
    raw = load_dataset()
    return featurize_dataframe(raw)


@st.cache_resource(show_spinner="Training models…")
def get_training_bundle():
    df = load_and_featurize()
    return train_models(df)


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>',
        unsafe_allow_html=True,
    )


def feature_importance_figure(model_result, feature_columns: list[str]):
    pipeline = model_result.pipeline
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_)
    else:
        return None

    importance_df = pd.DataFrame(
        {"Feature": feature_columns, "Importance": importances}
    ).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, height=320, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def parity_plot(model_result, smiles_test: pd.Series):
    plot_df = pd.DataFrame(
        {
            "Experimental LogS": model_result.y_test,
            "Predicted LogS": model_result.y_pred,
            "SMILES": smiles_test.values,
        }
    )

    fig = px.scatter(
        plot_df,
        x="Experimental LogS",
        y="Predicted LogS",
        hover_data={"SMILES": True, "Experimental LogS": ":.3f", "Predicted LogS": ":.3f"},
        title="Parity Plot — Predicted vs. Experimental LogS",
        opacity=0.7,
    )

    lo = min(plot_df["Experimental LogS"].min(), plot_df["Predicted LogS"].min()) - 0.5
    hi = max(plot_df["Experimental LogS"].max(), plot_df["Predicted LogS"].max()) + 0.5
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y = x",
            line=dict(color="crimson", dash="dash"),
        )
    )
    fig.update_layout(height=450)
    return fig


def main() -> None:
    st.title("🧪 Molar Solubility Predictor")
    st.caption(
        "QSPR pipeline for aqueous solubility (LogS) prediction using the Delaney ESOL dataset"
    )

    bundle = get_training_bundle()
    rf_result = get_model_by_name(bundle.results, RANDOM_FOREST)
    metrics_df = metrics_dataframe(bundle.results)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Model Performance")
        for _, row in metrics_df.iterrows():
            render_metric_card(f"{row['Model']} — R²", f"{row['R²']:.4f}")
            render_metric_card(f"{row['Model']} — MAE", f"{row['MAE']:.4f} log units")

        st.divider()
        st.subheader("Feature Importance")
        fi_fig = feature_importance_figure(rf_result, bundle.feature_columns)
        if fi_fig:
            st.plotly_chart(fi_fig, use_container_width=True)

    # --- Main content ---
    tab_predict, tab_dashboard = st.tabs(["Live Inference", "Performance Dashboard"])

    with tab_predict:
        st.subheader("Live Inference Playground")
        st.markdown(
            "Enter a SMILES string to draw the molecule and predict its aqueous solubility (LogS)."
        )

        default_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        user_smiles = st.text_input(
            "SMILES string",
            value=default_smiles,
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (Aspirin)",
        ).strip()

        if user_smiles:
            mol = Chem.MolFromSmiles(user_smiles)
            if mol is None:
                st.error("Invalid SMILES string. Please check your input and try again.")
            else:
                col_mol, col_pred = st.columns([1, 1])

                with col_mol:
                    img = Draw.MolToImage(mol, size=(400, 300))
                    st.image(img, caption=f"2D structure: `{user_smiles}`")

                with col_pred:
                    descriptors = smiles_to_descriptors(user_smiles)
                    if descriptors is None:
                        st.error("Could not compute descriptors for this molecule.")
                    else:
                        feature_row = pd.DataFrame([descriptors])
                        predicted_logs = rf_result.pipeline.predict(feature_row)[0]
                        interpretation = interpret_logs(predicted_logs)

                        st.markdown(
                            f"""
                            <div class="prediction-box">
                                <strong>Predicted LogS:</strong> {predicted_logs:.3f}<br>
                                <strong>Interpretation:</strong> {interpretation}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**Computed Descriptors**")
                        desc_df = pd.DataFrame(
                            [{"Descriptor": k, "Value": round(v, 4)} for k, v in descriptors.items()]
                        )
                        st.dataframe(desc_df, hide_index=True, use_container_width=True)

    with tab_dashboard:
        st.subheader("Model Comparison")
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.subheader("Parity Plot")
        selected_model = st.selectbox(
            "Select model for parity plot",
            [r.name for r in bundle.results],
            index=0,
        )
        selected_result = get_model_by_name(bundle.results, selected_model)
        st.plotly_chart(
            parity_plot(selected_result, bundle.smiles_test),
            use_container_width=True,
        )

    # --- How It Works ---
    st.markdown(
        """
        <div class="how-it-works">
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
                Three regression models (Random Forest, Gradient Boosting, Ridge) are trained on
                80% of the data and evaluated on a held-out 20% test set. Enter any valid SMILES
                above to get an instant solubility prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
