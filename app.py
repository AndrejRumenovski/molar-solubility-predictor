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
from src.interpretability import (
    explain_prediction,
    find_similar_compounds,
    prediction_interval_halfwidth,
)
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
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
        font-weight: 400;
        color: #ffffff;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .prediction-box {
        background: #f0f7ff;
        border-left: 4px solid #2d6a9f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        color: #1a1a1a;
    }
    .how-it-works {
        background: #fafafa;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
        color: #333333;
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
    fig.update_layout(height=420)
    return fig


def residuals_figure(model_result):
    residuals = model_result.y_test - model_result.y_pred
    fig = px.histogram(
        pd.DataFrame({"Residual (actual − predicted)": residuals}),
        x="Residual (actual − predicted)",
        nbins=30,
        title="Residual Distribution",
        color_discrete_sequence=["#2d6a9f"],
    )
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="crimson")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def shap_contribution_figure(contributions: pd.DataFrame):
    plot_df = contributions.sort_values("Contribution")
    plot_df["Direction"] = plot_df["Contribution"].apply(
        lambda c: "Increases LogS" if c >= 0 else "Decreases LogS"
    )

    fig = px.bar(
        plot_df,
        x="Contribution",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "Increases LogS": "#2d6a9f",
            "Decreases LogS": "#c0504d",
        },
        title="Feature Contributions to This Prediction (SHAP)",
    )
    fig.add_vline(x=0, line_width=1, line_color="#888888")
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
    )
    return fig


def run_batch_predictions(smiles_series: pd.Series, model_result) -> pd.DataFrame:
    """Featurize and predict LogS for each SMILES in the series."""
    rows = []
    for smi in smiles_series:
        smi = str(smi).strip()
        desc = smiles_to_descriptors(smi)
        if desc is None:
            rows.append({
                "SMILES": smi,
                "Valid": False,
                "Predicted LogS": None,
                "Interpretation": "Invalid / unparseable SMILES",
                **{k: None for k in FEATURE_COLUMNS},
            })
        else:
            feat = pd.DataFrame([desc])
            pred = float(model_result.pipeline.predict(feat)[0])
            rows.append({
                "SMILES": smi,
                "Valid": True,
                "Predicted LogS": round(pred, 3),
                "Interpretation": interpret_logs(pred),
                **{k: round(v, 4) for k, v in desc.items()},
            })
    return pd.DataFrame(rows)


def main() -> None:
    st.title("🧪 Molar Solubility Predictor")
    st.caption(
        "QSPR pipeline for aqueous solubility (LogS) prediction using the Delaney ESOL dataset"
    )

    bundle = get_training_bundle()
    rf_result = get_model_by_name(bundle.results, RANDOM_FOREST)
    metrics_df = metrics_dataframe(bundle.results)
    model_names = [r.name for r in bundle.results]

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
            st.plotly_chart(fi_fig)

    # --- Tabs ---
    tab_predict, tab_batch, tab_dashboard = st.tabs(
        ["Live Inference", "Batch Prediction", "Performance Dashboard"]
    )

    # ── Live Inference ──────────────────────────────────────────────────────
    with tab_predict:
        st.subheader("Live Inference Playground")
        st.markdown(
            "Enter a SMILES string to draw the molecule and predict its aqueous solubility (LogS)."
        )

        col_input, col_model_sel = st.columns([3, 1])
        with col_input:
            default_smiles = "CC(=O)Oc1ccccc1C(=O)O"
            user_smiles = st.text_input(
                "SMILES string",
                value=default_smiles,
                placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (Aspirin)",
            ).strip()
        with col_model_sel:
            active_model_name = st.selectbox("Model", model_names, key="live_model")

        active_model = get_model_by_name(bundle.results, active_model_name)

        if user_smiles:
            mol = Chem.MolFromSmiles(user_smiles)
            if mol is None:
                st.error("Invalid SMILES string. Please check your input and try again.")
            else:
                descriptors = smiles_to_descriptors(user_smiles)
                if descriptors is None:
                    st.error("Could not compute descriptors for this molecule.")
                else:
                    feature_row = pd.DataFrame([descriptors])
                    predicted_logs = active_model.pipeline.predict(feature_row)[0]
                    interpretation = interpret_logs(predicted_logs)
                    halfwidth = prediction_interval_halfwidth(active_model)

                    col_mol, col_pred = st.columns([1, 1])

                    with col_mol:
                        img = Draw.MolToImage(mol, size=(400, 300))
                        st.image(img, caption=f"2D structure: `{user_smiles}`")

                    with col_pred:
                        st.markdown(
                            f"""
                            <div class="prediction-box">
                                <strong>Predicted LogS:</strong> {predicted_logs:.3f}
                                &plusmn; {halfwidth:.3f}
                                <span style="opacity:0.7;">(95% interval)</span><br>
                                <strong>Interpretation:</strong> {interpretation}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**Computed Descriptors**")
                        desc_df = pd.DataFrame(
                            [{"Descriptor": k, "Value": round(v, 4)} for k, v in descriptors.items()]
                        )
                        st.dataframe(desc_df, hide_index=True)

                    st.divider()
                    col_shap, col_similar = st.columns([1, 1])

                    with col_shap:
                        contributions = explain_prediction(active_model, feature_row)
                        if contributions is not None:
                            st.plotly_chart(shap_contribution_figure(contributions))
                            st.caption(
                                "Bars show how each descriptor pushed this prediction above "
                                "or below the dataset's average LogS."
                            )
                        else:
                            st.info(
                                "SHAP explanations are available for Random Forest and "
                                "Gradient Boosting models only."
                            )

                    with col_similar:
                        st.markdown("**Most Similar Training Compounds**")
                        st.caption("Nearest neighbours by Morgan-fingerprint Tanimoto similarity.")
                        similar = find_similar_compounds(
                            user_smiles,
                            bundle.smiles_train,
                            bundle.y_train,
                            n_neighbors=5,
                        )
                        if similar.empty:
                            st.info(
                                "No training compounds exceed the similarity threshold — "
                                "this molecule lies outside the model's familiar chemical space, "
                                "so treat the prediction with extra caution."
                            )
                        else:
                            display = similar.copy()
                            display["Similarity"] = display["Similarity"].round(3)
                            display["Experimental LogS"] = display["Experimental LogS"].round(3)
                            st.dataframe(display, hide_index=True)

    # ── Batch Prediction ────────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Prediction")
        st.markdown(
            "Upload a CSV containing SMILES strings to predict aqueous solubility for multiple "
            "compounds at once and download the results."
        )

        col_upload, col_batch_opts = st.columns([2, 1])
        with col_upload:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
        with col_batch_opts:
            batch_model_name = st.selectbox("Model", model_names, key="batch_model")

        if uploaded is not None:
            batch_input = pd.read_csv(uploaded)
            st.caption(f"Loaded {len(batch_input):,} rows · {len(batch_input.columns)} columns.")

            candidates = [c for c in batch_input.columns if "smiles" in c.lower()]
            default_idx = batch_input.columns.tolist().index(candidates[0]) if candidates else 0
            smiles_col = st.selectbox(
                "SMILES column",
                batch_input.columns.tolist(),
                index=default_idx,
            )

            st.markdown("**Preview (first 5 rows)**")
            st.dataframe(batch_input[[smiles_col]].head(5), hide_index=True)

            if st.button("Run Predictions", type="primary"):
                batch_model = get_model_by_name(bundle.results, batch_model_name)
                with st.spinner(f"Predicting {len(batch_input):,} compounds…"):
                    result_df = run_batch_predictions(batch_input[smiles_col], batch_model)

                valid_n = int(result_df["Valid"].sum())
                st.success(
                    f"Done — {valid_n:,} / {len(result_df):,} compounds predicted successfully."
                )
                st.dataframe(result_df, hide_index=True)

                st.download_button(
                    "⬇ Download predictions as CSV",
                    result_df.to_csv(index=False).encode(),
                    file_name="solubility_predictions.csv",
                    mime="text/csv",
                )

    # ── Performance Dashboard ───────────────────────────────────────────────
    with tab_dashboard:
        st.subheader("Model Comparison")
        st.dataframe(metrics_df, hide_index=True)

        st.subheader("Diagnostics")
        dash_model_name = st.selectbox(
            "Select model",
            model_names,
            index=0,
            key="dashboard_model",
        )
        selected_result = get_model_by_name(bundle.results, dash_model_name)

        col_parity, col_resid = st.columns([1, 1])
        with col_parity:
            st.plotly_chart(parity_plot(selected_result, bundle.smiles_test))
        with col_resid:
            st.plotly_chart(residuals_figure(selected_result))

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
