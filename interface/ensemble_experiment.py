from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    model_perf = results_df[
        (results_df["status"] == "ok")
        & (results_df["role"].isin(["member", "stage"]))
        & results_df["RMSE"].notna()
    ].copy()
    if not model_perf.empty:
        order_cols = [c for c in ["strategy", "model_type", "seed", "member_id"] if c in model_perf.columns]
        model_perf = model_perf.sort_values(order_cols).copy()
        model_perf["run_idx"] = model_perf.groupby(["experiment_mode", "strategy"]).cumcount() + 1

        modes = list(model_perf["experiment_mode"].dropna().unique())
        fig, axes = plt.subplots(len(modes), 1, figsize=(13, max(4.2, 4.0 * len(modes))), squeeze=False)

        for idx, mode in enumerate(modes):
            ax = axes[idx, 0]
            sub = model_perf[model_perf["experiment_mode"] == mode].copy()

            strategies = sorted(sub["strategy"].dropna().unique())
            for strategy in strategies:
                strat_data = sub[sub["strategy"] == strategy].sort_values("run_idx")
                if strat_data.empty:
                    continue
                x = strat_data["run_idx"].values.astype(int)
                y = strat_data["RMSE"].values.astype(float)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.0,
                    markersize=5,
                    alpha=0.95,
                    label=STRATEGY_LABELS.get(str(strategy), str(strategy)),
                )

            if not sub.empty:
                y_min = float(sub["RMSE"].min())
                y_max = float(sub["RMSE"].max())
                pad = max(1e-6, (y_max - y_min) * 0.1)
                ax.set_ylim(y_min - pad, y_max + pad)

            ax.set_title(f"{mode} | Strategy RMSE trajectory")
            ax.set_xlabel("Run")
            ax.set_ylabel("RMSE")
            ax.set_xticks(sorted(sub["run_idx"].unique()))
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="upper left", ncols=2, fontsize=9, title="Strategy")

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
        "Run independent base models with a trained final fusion decision model, "
        "or run a strict sequential chain where each stage consumes original input plus previous output, "
        "while keeping dataset size and core hyperparameters from the sidebar."
    )

    ensemble_mode = st.radio(
        "Ensemble topology",
        options=["parallel_independent", "sequential_residual"],
        horizontal=True,
        format_func=lambda m: (
            "Independent models + final fusion"
            if m == "parallel_independent"
            else "Sequential chained pipeline (no fusion)"
        ),
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

    planned_members = len(selected_models) * num_seeds * max(len(selected_strategies), 1)
    st.caption(
        f"Estimated members: {planned_members} "
        f"({len(selected_models)} models x {num_seeds} seeds x {len(selected_strategies)} strategies)"
    )

    st.caption(
        "Mode details: Parallel mode trains members independently on the same data and applies fusion once. "
        "Sequential mode is strictly chained: stage input = original input + previous stage output, "
        "and final output is the terminal stage (no fusion step)."
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
        live_epoch_text = st.empty()
        live_table = st.empty()
        live_rows = []

        def _cb(done, total, label):
            progress.progress(done / max(total, 1), text=f"{label}…")

        def _epoch_cb(evt):
            live_epoch_text.caption(
                "Live epoch: "
                f"mode={evt.get('experiment_mode', '')} | "
                f"strategy={evt.get('strategy', '')} | "
                f"member={evt.get('member_id', '')} | "
                f"epoch={evt.get('epoch', '')} | "
                f"train={evt.get('train_loss', float('nan')):.6f} | "
                f"val={evt.get('val_loss', float('nan')):.6f}"
            )

        def _member_cb(row):
            live_rows.append(
                {
                    "experiment_mode": row.get("experiment_mode"),
                    "strategy": row.get("strategy"),
                    "role": row.get("role"),
                    "member_id": row.get("member_id"),
                    "model_type": row.get("model_type"),
                    "seed": row.get("seed"),
                    "RMSE": row.get("RMSE"),
                    "train_loss_final": row.get("train_loss_final"),
                    "val_loss_final": row.get("val_loss_final"),
                    "status": row.get("status"),
                }
            )
            live_df = pd.DataFrame(live_rows)
            if not live_df.empty:
                live_table.dataframe(live_df, width="stretch")

        with st.spinner("Running ensemble experiment…"):
            if ensemble_mode == "parallel_independent":
                result_df = parallel_independent_ensemble_study(
                    strategy_data=strategy_data,
                    model_types=selected_models,
                    fixed_params=fixed_params,
                    normalization=cfg["normalization"],
                    num_seeds=num_seeds,
                    progress_callback=_cb,
                    epoch_callback=_epoch_cb,
                    member_callback=_member_cb,
                )
            else:
                result_df = sequential_residual_ensemble_study(
                    strategy_data=strategy_data,
                    model_types=selected_models,
                    fixed_params=fixed_params,
                    normalization=cfg["normalization"],
                    num_seeds=num_seeds,
                    progress_callback=_cb,
                    epoch_callback=_epoch_cb,
                    member_callback=_member_cb,
                )

        progress.progress(1.0, text="Done.")
        st.session_state["ensemble_result"] = result_df

    if "ensemble_result" in st.session_state:
        _render_ensemble_results(st.session_state["ensemble_result"])
