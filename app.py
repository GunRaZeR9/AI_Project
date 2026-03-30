import streamlit as st

from data_processing import RF_IMPUTER_CACHE, inject_missing_values, load_dsmts_data
from interface import (
    render_experiment_overview_tab,
    render_federated_study_tab,
    render_gru_forecast_tab,
    render_missing_value_preview_tab,
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


def _prep_cache_key(cfg, missing_rates, impute_strategy: str, imputer_for_training):
    if len(cfg["df_clean"]) > 0:
        idx_start = str(cfg["df_clean"].index[0])
        idx_end = str(cfg["df_clean"].index[-1])
    else:
        idx_start = ""
        idx_end = ""

    imputer_token = None
    if imputer_for_training is not None and RF_IMPUTER_CACHE.exists():
        imputer_token = RF_IMPUTER_CACHE.stat().st_mtime_ns

    rates_key = tuple(sorted((str(k), float(v)) for k, v in missing_rates.items()))
    return (
        int(cfg.get("max_rows", len(cfg["df_clean"]))),
        int(len(cfg["df_clean"])),
        idx_start,
        idx_end,
        rates_key,
        impute_strategy,
        cfg.get("target_col", ""),
        bool(imputer_for_training is not None),
        imputer_token,
    )


def _get_prepared_frames(cfg, missing_rates, impute_strategy, imputer_for_training):
    key = _prep_cache_key(cfg, missing_rates, impute_strategy, imputer_for_training)
    cached = st.session_state.get("_prepared_frames_cache")
    if cached and cached.get("key") == key:
        return cached["df_with_missing"], cached["df_for_training"]

    df_clean = cfg["df_clean"]
    df_with_missing = inject_missing_values(df_clean, missing_rates)
    df_for_training = _apply_imputation(
        df_with_missing,
        impute_strategy,
        _fitted_imputer=imputer_for_training,
    )

    st.session_state["_prepared_frames_cache"] = {
        "key": key,
        "df_with_missing": df_with_missing,
        "df_for_training": df_for_training,
    }
    return df_with_missing, df_for_training


full_df = get_data()
all_cols = full_df.columns.tolist()
total_rows = len(full_df)

cfg = render_sidebar(full_df=full_df, all_cols=all_cols, total_rows=total_rows)

df_clean = cfg["df_clean"]
missing_rates = cfg["missing_rates"]
impute_strategy = cfg["impute_strategy"]
rf_imputer = st.session_state.get("rf_imputer")
imputer_for_training = rf_imputer if impute_strategy == "predictive_imputer" else None

df_with_missing, df_for_training = _get_prepared_frames(
    cfg,
    missing_rates,
    impute_strategy,
    imputer_for_training,
)

st.caption(
    f"Using the latest **{len(df_clean):,}** rows  |  "
    f"Columns: {len(all_cols)}  |  "
    f"Target: **{cfg['target_col']}**"
)

(tab_forecast, tab_overview, tab_federated, tab_preview) = st.tabs(
    [
        "GRU Forecast",
        "Experiment Overview",
        "Federated Learning",
        "Missing Value Preview",
    ]
)

with tab_forecast:
    render_gru_forecast_tab(df_for_training=df_for_training, cfg=cfg, impute_strategy=impute_strategy)

with tab_overview:
    render_experiment_overview_tab(df_with_missing=df_with_missing, cfg=cfg)

with tab_federated:
    render_federated_study_tab(
        df_for_training=df_for_training,
        cfg=cfg,
        impute_strategy=impute_strategy,
        df_with_missing=df_with_missing,
        fitted_imputer=rf_imputer,
    )

with tab_preview:
    render_missing_value_preview_tab(
        df_with_missing=df_with_missing,
        target_col=cfg["target_col"],
        fitted_imputer=rf_imputer,
    )
