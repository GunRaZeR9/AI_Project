import streamlit as st

from data_processing import inject_missing_values, load_dsmts_data
from interface import (
    render_experiment_overview_tab,
    render_federated_study_tab,
    render_gru_forecast_tab,
    render_hyperparameter_analysis_tab,
    render_missing_value_preview_tab,
    render_normalization_study_tab,
    render_sidebar,
)

st.title("DSMTS Univariate Time Series Forecasting")
st.caption(
    "Dataset: Dynamical System Multivariate Time Series (DSMTS) — "
    "17-variable simulated complex dynamical system at 1 Hz."
)


@st.cache_data(show_spinner="Downloading DSMTS dataset…")
def get_data():
    return load_dsmts_data()


@st.cache_data(show_spinner=False)
def _apply_imputation(df, strategy: str, _fitted_imputer=None):
    from data_processing import process_for_viz

    _, df_out, _ = process_for_viz(df, strategy=strategy, fitted_imputer=_fitted_imputer)
    return df_out


full_df = get_data()
all_cols = full_df.columns.tolist()
total_rows = len(full_df)

cfg = render_sidebar(full_df=full_df, all_cols=all_cols, total_rows=total_rows)

df_clean = cfg["df_clean"]
missing_rates = cfg["missing_rates"]
impute_strategy = cfg["impute_strategy"]

df_with_missing = inject_missing_values(df_clean, missing_rates)
imputer_for_training = st.session_state.get("rf_imputer") if impute_strategy == "predictive_imputer" else None
df_for_training = _apply_imputation(
    df_with_missing,
    impute_strategy,
    _fitted_imputer=imputer_for_training,
)

st.caption(
    f"Using the latest **{len(df_clean):,}** rows  |  "
    f"Columns: {len(all_cols)}  |  "
    f"Target: **{cfg['target_col']}**"
)

(tab_forecast, tab_hp, tab_norm, tab_overview, tab_federated, tab_preview) = st.tabs(
    [
        "GRU Forecast",
        "Hyperparameter Analysis",
        "Normalization Study",
        "Experiment Overview",
        "Federated Learning",
        "Missing Value Preview",
    ]
)

with tab_forecast:
    render_gru_forecast_tab(df_for_training=df_for_training, cfg=cfg, impute_strategy=impute_strategy)

with tab_hp:
    render_hyperparameter_analysis_tab(df_for_training=df_for_training, cfg=cfg, impute_strategy=impute_strategy)

with tab_norm:
    render_normalization_study_tab(df_for_training=df_for_training, cfg=cfg, impute_strategy=impute_strategy)

with tab_overview:
    render_experiment_overview_tab(df_with_missing=df_with_missing, cfg=cfg)

with tab_federated:
    render_federated_study_tab(
        df_for_training=df_for_training,
        cfg=cfg,
        impute_strategy=impute_strategy,
        df_with_missing=df_with_missing,
        fitted_imputer=st.session_state.get("rf_imputer"),
    )

with tab_preview:
    render_missing_value_preview_tab(
        df_with_missing=df_with_missing,
        target_col=cfg["target_col"],
        fitted_imputer=st.session_state.get("rf_imputer"),
    )
