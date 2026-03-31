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


STYLE_PALETTE = [
    "#0E7490",  # cyan deep
    "#EA580C",  # orange deep
    "#16A34A",  # green
    "#7C3AED",  # violet
    "#BE123C",  # rose
    "#1D4ED8",  # blue
    "#CA8A04",  # amber
    "#4B5563",  # slate
]


def _strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(str(strategy), str(strategy))


def _parse_stage_idx(member_id: str) -> int | None:
    text = str(member_id)
    if not text.startswith("stage_"):
        return None
    try:
        return int(text.split("_", maxsplit=1)[1])
    except Exception:
        return None


def _stylize_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.set_facecolor("#F8FAFC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def _build_strategy_colors(strategies: list[str]) -> dict[str, str]:
    if not strategies:
        return {}
    return {s: STYLE_PALETTE[i % len(STYLE_PALETTE)] for i, s in enumerate(strategies)}


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

    viz_metric_opts = [m for m in ["RMSE", "MAE", "MAPE (%)", "MSE"] if m in results_df.columns]
    if not viz_metric_opts:
        st.info("No numeric metric columns available for diagnostics plots.")
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
        st.download_button(
            "Download ensemble results CSV",
            data=csv_bytes,
            file_name=f"{stamp}_ensemble_export.csv",
            mime="text/csv",
            key="btn_download_ensemble_csv",
        )
        return

    default_metric_idx = viz_metric_opts.index("RMSE") if "RMSE" in viz_metric_opts else 0
    all_strategies = sorted(results_df["strategy"].dropna().unique().tolist())

    ctrl_a, ctrl_b, ctrl_c = st.columns([1.2, 1.8, 1.2])
    selected_metric = ctrl_a.selectbox(
        "Primary metric",
        options=viz_metric_opts,
        index=default_metric_idx,
        key="ensemble_plot_metric",
    )
    selected_strategies = ctrl_b.multiselect(
        "Strategies to display",
        options=all_strategies,
        default=all_strategies,
        format_func=_strategy_label,
        key="ensemble_plot_strategies",
    )
    share_y_limits = ctrl_c.checkbox(
        "Share y-limits", value=True, key="ensemble_plot_share_ylims"
    )

    ok_df = results_df[results_df["status"] == "ok"].copy()
    if selected_strategies:
        ok_df = ok_df[ok_df["strategy"].isin(selected_strategies)].copy()

    independent_df = ok_df[ok_df["experiment_mode"] == "parallel_independent"].copy()
    if not independent_df.empty:
        st.markdown("#### Independent diagnostics")
        st.caption(
            "Run trajectory, run-to-run improvement, loss diagnostics, and final efficiency frontier."
        )

        independent_members = independent_df[
            independent_df["role"].isin(["member"]) & independent_df[selected_metric].notna()
        ].copy()
        independent_ensemble = independent_df[
            independent_df["role"].isin(["ensemble"]) & independent_df[selected_metric].notna()
        ].copy()

        if not independent_members.empty:
            order_cols = [c for c in ["strategy", "model_type", "seed", "member_id"] if c in independent_members.columns]
            independent_members = independent_members.sort_values(order_cols).copy()
            independent_members["run_idx"] = independent_members.groupby(["strategy"]).cumcount() + 1

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.patch.set_facecolor("#F3F4F6")
            ax_traj, ax_delta = axes[0]
            ax_loss, ax_frontier = axes[1]

            strategies = sorted(independent_members["strategy"].dropna().unique().tolist())
            strat_color = _build_strategy_colors(strategies)

            for strategy in strategies:
                strat_data = independent_members[independent_members["strategy"] == strategy].sort_values("run_idx")
                if strat_data.empty:
                    continue
                x = strat_data["run_idx"].values.astype(int)
                y = strat_data[selected_metric].values.astype(float)
                ax_traj.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    alpha=0.95,
                    color=strat_color[strategy],
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    label=_strategy_label(strategy),
                )

                ens_strat = independent_ensemble[independent_ensemble["strategy"] == strategy]
                if not ens_strat.empty:
                    ens_val = float(ens_strat[selected_metric].mean())
                    ax_traj.axhline(
                        ens_val,
                        linestyle=":",
                        linewidth=1.4,
                        alpha=0.85,
                        color=strat_color[strategy],
                    )

            _stylize_ax(
                ax_traj,
                f"Independent | Strategy {selected_metric} trajectory",
                "Run",
                selected_metric,
            )
            if not independent_members.empty:
                ax_traj.set_xticks(sorted(independent_members["run_idx"].unique()))
            ax_traj.legend(loc="upper left", ncols=2, fontsize=8, title="Strategy", frameon=False)

            run_all = sorted(independent_members["run_idx"].dropna().astype(int).unique().tolist())
            if run_all:
                width = 0.8 / max(len(strategies), 1)
                for i, strategy in enumerate(strategies):
                    strat = independent_members[independent_members["strategy"] == strategy].copy()
                    grp = strat.groupby("run_idx", dropna=True)[selected_metric].mean().sort_index()
                    deltas = []
                    for run_idx in run_all:
                        prev_idx = run_idx - 1
                        if prev_idx not in grp.index or run_idx not in grp.index:
                            deltas.append(np.nan)
                        else:
                            deltas.append(float(grp.loc[prev_idx] - grp.loc[run_idx]))
                    x_positions = np.array(run_all, dtype=float) - 0.4 + width / 2 + i * width
                    ax_delta.bar(
                        x_positions,
                        np.array(deltas, dtype=float),
                        width=width,
                        color=strat_color[strategy],
                        alpha=0.9,
                        edgecolor="white",
                        linewidth=0.5,
                        label=_strategy_label(strategy),
                    )
                ax_delta.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
                ax_delta.set_xticks(run_all)
            _stylize_ax(
                ax_delta,
                f"Independent | Run-to-run Δ{selected_metric} (positive is better)",
                "Run",
                f"Δ{selected_metric}",
            )

            loss_agg = independent_members.groupby(["strategy", "run_idx"], dropna=True).agg(
                train_loss=("train_loss_final", "mean"),
                val_loss=("val_loss_final", "mean"),
            )
            if not loss_agg.empty:
                for strategy in strategies:
                    if strategy not in loss_agg.index.get_level_values(0):
                        continue
                    sub = loss_agg.loc[strategy].sort_index()
                    x = sub.index.values.astype(int)
                    ax_loss.plot(
                        x,
                        sub["train_loss"].values.astype(float),
                        linewidth=1.8,
                        marker="o",
                        markersize=4,
                        color=strat_color[strategy],
                        linestyle="-",
                        label=f"{_strategy_label(strategy)} train",
                    )
                    ax_loss.plot(
                        x,
                        sub["val_loss"].values.astype(float),
                        linewidth=1.2,
                        marker="x",
                        markersize=4,
                        color=strat_color[strategy],
                        linestyle="--",
                        label=f"{_strategy_label(strategy)} val",
                    )
            _stylize_ax(ax_loss, "Independent | Run loss diagnostics", "Run", "Loss")
            ax_loss.legend(loc="upper right", fontsize=7, ncols=2, frameon=False)

            runtime_sum = independent_members.groupby("strategy", dropna=True)["runtime_sec"].sum()
            final_metric = independent_ensemble.groupby("strategy", dropna=True)[selected_metric].mean()
            if final_metric.empty:
                final_metric = independent_members.groupby("strategy", dropna=True)[selected_metric].mean()
            frontier = pd.concat([runtime_sum, final_metric], axis=1, keys=["total_runtime", "final_metric"]).dropna()
            if not frontier.empty:
                ax_frontier.scatter(
                    frontier["total_runtime"].values.astype(float),
                    frontier["final_metric"].values.astype(float),
                    s=95,
                    alpha=0.85,
                    c=[strat_color.get(s, "tab:blue") for s in frontier.index],
                    edgecolors="white",
                    linewidths=0.7,
                )
                for strategy, row in frontier.iterrows():
                    ax_frontier.annotate(
                        _strategy_label(strategy),
                        (float(row["total_runtime"]), float(row["final_metric"])),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
            _stylize_ax(
                ax_frontier,
                f"Independent | Total runtime vs final {selected_metric}",
                "Total member runtime (s)",
                f"Final {selected_metric}",
            )

            if share_y_limits:
                y_vals = independent_members[selected_metric].dropna().astype(float)
                if len(y_vals) > 1:
                    y_min, y_max = float(y_vals.min()), float(y_vals.max())
                    pad = max(1e-6, 0.1 * (y_max - y_min))
                    ax_traj.set_ylim(y_min - pad, y_max + pad)
                    ax_delta.set_ylim(-(y_max - y_min + pad), (y_max - y_min + pad))

            fig.suptitle("Independent Ensemble Diagnostics", fontsize=13, fontweight="bold", x=0.02, ha="left")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        else:
            st.info("No successful independent-member rows to chart.")

    sequential_df = ok_df[ok_df["experiment_mode"] == "sequential_residual"].copy()
    if not sequential_df.empty:
        st.markdown("#### Sequential diagnostics")
        st.caption(
            "Stage trajectory, stage-to-stage improvement, loss diagnostics, and final efficiency frontier."
        )

        sequential_stage = sequential_df[
            sequential_df["role"].isin(["stage"]) & sequential_df[selected_metric].notna()
        ].copy()
        sequential_final = sequential_df[
            sequential_df["role"].isin(["final_output"]) & sequential_df[selected_metric].notna()
        ].copy()

        if not sequential_stage.empty:
            sequential_stage["stage_idx"] = sequential_stage["member_id"].map(_parse_stage_idx)
            sequential_stage = sequential_stage[sequential_stage["stage_idx"].notna()].copy()
            sequential_stage["stage_idx"] = sequential_stage["stage_idx"].astype(int)

            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
            fig2.patch.set_facecolor("#F3F4F6")
            ax_seq_traj, ax_delta = axes2[0]
            ax_loss, ax_seq_frontier = axes2[1]

            seq_strategies = sorted(sequential_stage["strategy"].dropna().unique().tolist())
            seq_color = _build_strategy_colors(seq_strategies)

            for strategy in seq_strategies:
                strat = sequential_stage[sequential_stage["strategy"] == strategy].copy()
                grp = strat.groupby("stage_idx", dropna=True)[selected_metric].mean().sort_index()
                if grp.empty:
                    continue
                ax_seq_traj.plot(
                    grp.index.values.astype(int),
                    grp.values.astype(float),
                    marker="o",
                    linewidth=2.0,
                    markersize=5,
                    color=seq_color[strategy],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=_strategy_label(strategy),
                )

            _stylize_ax(
                ax_seq_traj,
                f"Sequential | Stage mean {selected_metric} trajectory",
                "Stage",
                selected_metric,
            )
            ax_seq_traj.legend(loc="upper right", ncols=2, fontsize=8, title="Strategy", frameon=False)

            stage_all = sorted(sequential_stage["stage_idx"].dropna().astype(int).unique().tolist())
            if stage_all:
                width = 0.8 / max(len(seq_strategies), 1)
                for i, strategy in enumerate(seq_strategies):
                    strat = sequential_stage[sequential_stage["strategy"] == strategy].copy()
                    grp = strat.groupby("stage_idx", dropna=True)[selected_metric].mean().sort_index()
                    deltas = []
                    for stage_idx in stage_all:
                        prev_idx = stage_idx - 1
                        if prev_idx not in grp.index or stage_idx not in grp.index:
                            deltas.append(np.nan)
                        else:
                            deltas.append(float(grp.loc[prev_idx] - grp.loc[stage_idx]))
                    x_positions = np.array(stage_all, dtype=float) - 0.4 + width / 2 + i * width
                    ax_delta.bar(
                        x_positions,
                        np.array(deltas, dtype=float),
                        width=width,
                        color=seq_color[strategy],
                        alpha=0.9,
                        edgecolor="white",
                        linewidth=0.5,
                        label=_strategy_label(strategy),
                    )
                ax_delta.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
                ax_delta.set_xticks(stage_all)

            _stylize_ax(
                ax_delta,
                f"Sequential | Stage-to-stage Δ{selected_metric} (positive is better)",
                "Stage",
                f"Δ{selected_metric}",
            )

            loss_agg = sequential_stage.groupby(["strategy", "stage_idx"], dropna=True).agg(
                train_loss=("train_loss_final", "mean"),
                val_loss=("val_loss_final", "mean"),
            )
            if not loss_agg.empty:
                for strategy in seq_strategies:
                    if strategy not in loss_agg.index.get_level_values(0):
                        continue
                    sub = loss_agg.loc[strategy].sort_index()
                    x = sub.index.values.astype(int)
                    ax_loss.plot(
                        x,
                        sub["train_loss"].values.astype(float),
                        linewidth=1.8,
                        marker="o",
                        markersize=4,
                        color=seq_color[strategy],
                        linestyle="-",
                        label=f"{_strategy_label(strategy)} train",
                    )
                    ax_loss.plot(
                        x,
                        sub["val_loss"].values.astype(float),
                        linewidth=1.2,
                        marker="x",
                        markersize=4,
                        color=seq_color[strategy],
                        linestyle="--",
                        label=f"{_strategy_label(strategy)} val",
                    )
            _stylize_ax(ax_loss, "Sequential | Stage loss diagnostics", "Stage", "Loss")
            ax_loss.legend(loc="upper right", fontsize=7, ncols=2, frameon=False)

            runtime_sum = sequential_stage.groupby("strategy", dropna=True)["runtime_sec"].sum()
            final_metric = sequential_final.groupby("strategy", dropna=True)[selected_metric].mean()
            joined = pd.concat([runtime_sum, final_metric], axis=1, keys=["total_runtime", "final_metric"]).dropna()
            if not joined.empty:
                ax_seq_frontier.scatter(
                    joined["total_runtime"].values.astype(float),
                    joined["final_metric"].values.astype(float),
                    s=95,
                    alpha=0.9,
                    c=[seq_color.get(s, "tab:green") for s in joined.index],
                    edgecolors="white",
                    linewidths=0.7,
                )
                for strategy, row in joined.iterrows():
                    ax_seq_frontier.annotate(
                        _strategy_label(strategy),
                        (float(row["total_runtime"]), float(row["final_metric"])),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
            _stylize_ax(
                ax_seq_frontier,
                f"Sequential | Total runtime vs final {selected_metric}",
                "Total stage runtime (s)",
                f"Final {selected_metric}",
            )

            if share_y_limits:
                y_vals = sequential_stage[selected_metric].dropna().astype(float)
                if len(y_vals) > 1:
                    y_min, y_max = float(y_vals.min()), float(y_vals.max())
                    pad = max(1e-6, 0.1 * (y_max - y_min))
                    ax_seq_traj.set_ylim(y_min - pad, y_max + pad)

            fig2.suptitle("Sequential Ensemble Diagnostics", fontsize=13, fontweight="bold", x=0.02, ha="left")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig2, width="stretch")
            plt.close(fig2)
        else:
            st.info("No successful sequential-stage rows to chart.")

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
