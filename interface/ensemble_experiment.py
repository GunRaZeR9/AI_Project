from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from data_processing import STRATEGIES, process_for_viz
from forecasting.studies.experiments import (
    parallel_independent_ensemble_study,
    sequential_residual_ensemble_study,
)
from forecasting.utils.common import train_test_split_series

from .common import available_metrics, format_metrics_table
from .constants import METRIC_COLUMNS, STRATEGY_LABELS


def _render_ensemble_results(results_df):
    if results_df is None or results_df.empty:
        st.info("No ensemble results yet. Run the experiment to see results.")
        return

    st.markdown("### Ensemble results")
    metric_cols = available_metrics(results_df, METRIC_COLUMNS)
    display_cols = [
        c
        for c in [
            "experiment_mode",
            "strategy",
            "scope",
            "role",
            "member_id",
            "model_type",
            "seed",
            *metric_cols,
            "train_loss_final",
            "val_loss_final",
            "runtime_sec",
            "status",
            "error",
        ]
        if c in results_df.columns
    ]

    styled = results_df[display_cols].copy().style
    styled = format_metrics_table(styled, include_runtime=True)
    st.dataframe(styled, width="stretch")

    ensemble_only = results_df[(results_df["role"] == "ensemble") & (results_df["status"] == "ok")].copy()
    if not ensemble_only.empty and "RMSE" in ensemble_only.columns:
        best_idx = int(np.nanargmin(ensemble_only["RMSE"].values))
        best_row = ensemble_only.iloc[best_idx]
        st.success(
            "Best ensemble: "
            f"mode={best_row.get('experiment_mode', '')}, "
            f"strategy={best_row.get('strategy', '')}, "
            f"scope={best_row.get('scope', '')}, "
            f"RMSE={best_row.get('RMSE', float('nan')):.6f}"
        )

    if not ensemble_only.empty and "RMSE" in ensemble_only.columns:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        labels = [
            f"{r.get('strategy', '')}|{r.get('experiment_mode', '')}|{r.get('scope', '')}"
            for _, r in ensemble_only.iterrows()
        ]
        vals = ensemble_only["RMSE"].astype(float).values
        x = np.arange(len(vals))
        ax.bar(x, vals, color="#2a6f97")
        ax.set_title("Ensemble RMSE comparison")
        ax.set_ylabel("RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    st.download_button(
        "Download ensemble results CSV",
        data=csv_bytes,
        file_name=f"{stamp}_ensemble_export.csv",
        mime="text/csv",
        key="btn_download_ensemble_csv",
    )


def render_ensemble_experiment_tab(df_with_missing, cfg):
    st.subheader("Ensemble Experiment")
    st.caption(
        "Run independent (parallel members) or sequential (residual-chain) ensembles "
        "while keeping dataset size and core hyperparameters from the sidebar."
    )

    ensemble_mode = st.radio(
        "Ensemble topology",
        options=["parallel_independent", "sequential_residual"],
        horizontal=True,
        format_func=lambda m: "Parallel independent" if m == "parallel_independent" else "Sequential residual chain",
        key="ensemble_topology",
    )

    selected_models = st.multiselect(
        "Models",
        options=["gru", "rnn", "lstm", "mlp", "tcn"],
        default=["gru", "rnn", "lstm", "mlp", "tcn"],
        format_func=lambda s: s.upper(),
        key="ensemble_models",
    )
    selected_strategies = st.multiselect(
        "Imputation strategies",
        options=STRATEGIES,
        default=STRATEGIES,
        format_func=lambda s: STRATEGY_LABELS.get(s, s),
        key="ensemble_strategies",
    )

    num_seeds = st.slider("Number of seeds", min_value=2, max_value=10, value=5, step=1)
    max_workers = st.slider(
        "Parallel workers (independent mode)",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Used only for independent mode. GPU runtime may auto-fallback to single worker.",
    )

    planned_members = len(selected_models) * num_seeds * 2 * max(len(selected_strategies), 1)
    st.caption(
        f"Estimated members: {planned_members} "
        f"({len(selected_models)} models x {num_seeds} seeds x 2 train halves x {len(selected_strategies)} strategies)"
    )

    run_ensemble = st.button("▶ Run Ensemble Experiment", type="primary", key="btn_ensemble")

    if run_ensemble:
        if not selected_models:
            st.warning("Select at least one model.")
            return
        if not selected_strategies:
            st.warning("Select at least one strategy.")
            return

        strategy_data = {}
        skipped = []
        for strat in selected_strategies:
            try:
                fitted_imp = st.session_state.get("rf_imputer") if strat == "predictive_imputer" else None
                _, df_imp, _ = process_for_viz(df_with_missing, strategy=strat, fitted_imputer=fitted_imp)
                if cfg["target_col"] not in df_imp.columns:
                    skipped.append(f"{strat} (target unavailable)")
                    continue

                ts_full = df_imp[cfg["target_col"]].astype(float).dropna()
                train_s, test_s = train_test_split_series(ts_full, test_size=cfg["test_size"])
                train_s = train_s.dropna()
                test_s = test_s.dropna()

                if train_s.empty or test_s.empty:
                    skipped.append(f"{strat} (insufficient data)")
                    continue

                strategy_data[strat] = (train_s.values, test_s.values)
            except Exception as exc:
                skipped.append(f"{strat} ({exc})")

        if not strategy_data:
            st.error("No valid strategy datasets were prepared.")
            return

        if skipped:
            st.warning("Skipped strategies: " + " | ".join(skipped))

        fixed_params = {
            "seq_len": cfg["seq_len"],
            "hidden": cfg["hidden"],
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "num_layers": cfg["num_layers"],
            "batch_size": 64,
            "max_train_points": 12_000,
            "activation": cfg["activation"],
            "dropout": cfg["dropout"],
            "optimizer_name": cfg["optimizer_name"],
            "loss_fn_name": cfg["loss_fn_name"],
            "weight_decay": cfg["weight_decay"],
            "l1_lambda": cfg["l1_lambda"],
            "val_fraction": cfg["val_fraction"],
        }

        progress = st.progress(0, text="Preparing ensemble runs…")

        def _cb(done, total, label):
            progress.progress(done / max(total, 1), text=f"{label}…")

        with st.spinner("Running ensemble experiment…"):
            if ensemble_mode == "parallel_independent":
                result_df = parallel_independent_ensemble_study(
                    strategy_data=strategy_data,
                    model_types=selected_models,
                    fixed_params=fixed_params,
                    normalization=cfg["normalization"],
                    num_seeds=num_seeds,
                    max_workers=max_workers,
                    progress_callback=_cb,
                )
            else:
                result_df = sequential_residual_ensemble_study(
                    strategy_data=strategy_data,
                    model_types=selected_models,
                    fixed_params=fixed_params,
                    normalization=cfg["normalization"],
                    num_seeds=num_seeds,
                    progress_callback=_cb,
                )

        progress.progress(1.0, text="Done.")
        st.session_state["ensemble_result"] = result_df

    if "ensemble_result" in st.session_state:
        _render_ensemble_results(st.session_state["ensemble_result"])
