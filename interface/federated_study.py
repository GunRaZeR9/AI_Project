import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from forecasting.studies.experiments import federated_training_study
from forecasting.inference.predictor import gru_one_step_predict
from forecasting.normalization.scalers import _apply_norm_params
from data_processing import STRATEGIES, partition_data_for_users, process_for_viz

from .constants import STRATEGY_LABELS


def _strategy_name(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


def _impute_for_strategy(df_with_missing, strategy: str, fitted_imputer=None):
    strategy_imputer = fitted_imputer if strategy == "predictive_imputer" else None
    _, df_out, _ = process_for_viz(
        df_with_missing,
        strategy=strategy,
        fitted_imputer=strategy_imputer,
    )
    return df_out


def _set_readable_round_ticks(ax, rounds, max_ticks: int = 12):
    if rounds is None:
        return
    clean_rounds = sorted({int(r) for r in rounds if pd.notna(r)})
    if not clean_rounds:
        return
    if len(clean_rounds) <= max_ticks:
        ax.set_xticks(clean_rounds)
        return

    idx = np.linspace(0, len(clean_rounds) - 1, max_ticks, dtype=int)
    ticks = [clean_rounds[i] for i in sorted(set(idx))]
    if ticks[0] != clean_rounds[0]:
        ticks = [clean_rounds[0], *ticks]
    if ticks[-1] != clean_rounds[-1]:
        ticks = [*ticks, clean_rounds[-1]]
    ax.set_xticks(sorted(set(ticks)))


def _sorted_model_ids(model_ids):
    return sorted(model_ids, key=lambda m: (m != "global_aggregated", m))


def _resolve_live_model_focus(epoch_df: pd.DataFrame, current_model: str, focus_mode: str, max_models: int):
    available = _sorted_model_ids(epoch_df["model_id"].dropna().unique().tolist()) if not epoch_df.empty else []
    if not available:
        return []

    if focus_mode == "All models":
        selected = available
    else:
        selected = [m for m in ["global_aggregated", current_model] if m in available]
        if not selected:
            selected = [available[0]]

    if max_models > 0 and len(selected) > max_models:
        selected = selected[:max_models]
    return selected


def render_live_federated_training_dashboard(
    epoch_df: pd.DataFrame,
    round_df: pd.DataFrame,
    strategy_label: str,
    live_plot_ph,
    local_epochs: int,
    current_round: int,
    current_model: str,
    focus_mode: str,
    max_models: int,
):
    if epoch_df is None or epoch_df.empty:
        return

    selected_models = _resolve_live_model_focus(epoch_df, current_model, focus_mode, max_models=max_models)
    if not selected_models:
        return

    nrows = len(selected_models) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12.5, max(4.5, 2.6 * nrows)))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    max_x = 1
    for i, model_id in enumerate(selected_models):
        ax = axes_arr[i]
        model_df = epoch_df[epoch_df["model_id"] == model_id].sort_values(["round", "epoch"])
        if model_df.empty:
            ax.set_title(f"{model_id} (waiting for epochs)")
            ax.grid(alpha=0.28, linestyle="--")
            continue

        x = (model_df["round"].astype(int) - 1) * max(1, int(local_epochs)) + model_df["epoch"].astype(int)
        max_x = max(max_x, int(np.nanmax(x.values)))
        train_loss = model_df["train_loss"].astype(float).values
        val_loss = model_df["val_loss"].astype(float).values

        ax.plot(x, train_loss, color="#1f77b4", linewidth=1.6, label="Train loss")
        if np.any(np.isfinite(val_loss)):
            ax.plot(x, val_loss, color="#ff7f0e", linewidth=1.6, label="Val loss")

        for boundary in range(1, int(current_round) + 1):
            xpos = boundary * max(1, int(local_epochs))
            ax.axvline(xpos, color="#b0b0b0", linewidth=0.8, linestyle=":", alpha=0.45)

        ax.set_title(f"{model_id} live local training")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.28, linestyle="--")

    round_ax = axes_arr[-1]
    if round_df is not None and not round_df.empty:
        plot_metric = "RMSE" if "RMSE" in round_df.columns else "MSE"
        for model_id in _sorted_model_ids(round_df["model_id"].dropna().unique().tolist()):
            m_df = round_df[round_df["model_id"] == model_id].sort_values("round")
            if m_df.empty or plot_metric not in m_df.columns:
                continue
            lw = 2.1 if model_id == "global_aggregated" else 1.5
            round_ax.plot(m_df["round"], m_df[plot_metric], marker="o", linewidth=lw, label=model_id)
        round_ax.set_title(f"Live round overview ({plot_metric})")
        round_ax.set_xlabel("Round")
        round_ax.set_ylabel(plot_metric)
        round_ax.grid(alpha=0.28, linestyle="--")
        _set_readable_round_ticks(round_ax, round_df["round"].dropna().tolist())
        handles, labels = round_ax.get_legend_handles_labels()
        if handles:
            round_ax.legend(loc="best", ncol=2, fontsize=8)
    else:
        round_ax.set_title("Live round overview (waiting for first completed user)")
        round_ax.set_xlabel("Round")
        round_ax.set_ylabel("RMSE")
        round_ax.grid(alpha=0.28, linestyle="--")

    for i in range(len(selected_models)):
        axes_arr[i].set_xlim(left=1, right=max_x + 1)
        handles, labels = axes_arr[i].get_legend_handles_labels()
        if handles:
            axes_arr[i].legend(loc="best", fontsize=8)
    fig.suptitle(f"Live federated training - {strategy_label}", y=1.01)
    plt.tight_layout()
    live_plot_ph.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_strategy_round_metric_overview(user_round_df: pd.DataFrame, strategy_label: str, key_prefix: str):
    if user_round_df is None or user_round_df.empty:
        st.info("No round metrics available for this strategy.")
        return

    st.markdown("### Round Overview - All Models")
    st.caption("MSE, RMSE, MAE and MAPE (%) across rounds for every user and global aggregated model.")

    model_ids = _sorted_model_ids(user_round_df["model_id"].dropna().unique().tolist())
    selected_models = st.multiselect(
        "Models included in round overview",
        options=model_ids,
        default=model_ids,
        key=f"{key_prefix}_round_models",
    )
    if not selected_models:
        st.info("Select at least one model to render the round overview.")
        return

    metric_specs = ["MSE", "RMSE", "MAE", "MAPE (%)"]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for idx, metric in enumerate(metric_specs):
        ax = axes_arr[idx]
        metric_max = float("-inf")
        for model_id in selected_models:
            m_df = user_round_df[user_round_df["model_id"] == model_id].sort_values("round")
            if m_df.empty or metric not in m_df.columns:
                continue
            y = m_df[metric].astype(float).values
            finite_y = y[np.isfinite(y)]
            if finite_y.size:
                metric_max = max(metric_max, float(np.max(finite_y)))
            lw = 2.2 if model_id == "global_aggregated" else 1.7
            ax.plot(m_df["round"].values, y, marker="o", linewidth=lw, label=model_id)
        ax.set_title(metric)
        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, linestyle="--")
        _set_readable_round_ticks(ax, user_round_df["round"].dropna().tolist())
        if np.isfinite(metric_max) and metric_max > 0:
            ax.set_ylim(0.0, metric_max * 1.05)

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=True)
    fig.suptitle(f"Round metrics by model - {strategy_label}", y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(
        user_round_df.sort_values(["model_id", "round"]).style.format(
            {
                "MSE": "{:,.6f}",
                "MAE": "{:,.6f}",
                "MAPE (%)": "{:,.4f}",
                "RMSE": "{:,.6f}",
                "runtime_sec": "{:,.2f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
    )


def _build_federated_prediction_curves(chosen_result: dict, user_data_dict: dict) -> dict:
    curves = {}
    if not chosen_result or not user_data_dict:
        return curves

    user_ids = chosen_result.get("user_ids", list(user_data_dict.keys()))
    local_models = chosen_result.get("local_models", [])
    global_norm_params = chosen_result.get("global_norm_params")

    if global_norm_params is None:
        return curves

    for idx, user_id in enumerate(user_ids):
        if idx >= len(local_models) or user_id not in user_data_dict:
            continue

        model_result = local_models[idx]
        raw_train_vals, raw_test_vals = user_data_dict[user_id]
        norm_train_vals = _apply_norm_params(raw_train_vals, global_norm_params)
        norm_test_vals = _apply_norm_params(raw_test_vals, global_norm_params)
        norm_full_vals = np.concatenate([norm_train_vals, norm_test_vals])
        preds = gru_one_step_predict(model_result, norm_full_vals, start_idx=len(norm_train_vals))

        n = min(len(norm_test_vals), len(preds))
        if n == 0:
            continue

        curves[user_id] = {
            "observed": np.asarray(norm_test_vals[:n], dtype=float),
            "predicted": np.asarray(preds[:n], dtype=float),
        }

    global_model_result = chosen_result.get("global_model_result")
    if global_model_result and user_ids:
        all_train_vals = np.concatenate([user_data_dict[uid][0] for uid in user_ids if uid in user_data_dict])
        all_test_vals = np.concatenate([user_data_dict[uid][1] for uid in user_ids if uid in user_data_dict])

        if len(all_train_vals) and len(all_test_vals):
            norm_all_train = _apply_norm_params(all_train_vals, global_norm_params)
            norm_all_test = _apply_norm_params(all_test_vals, global_norm_params)
            norm_combined = np.concatenate([norm_all_train, norm_all_test])
            global_preds = gru_one_step_predict(
                global_model_result,
                norm_combined,
                start_idx=len(norm_all_train),
            )

            n = min(len(norm_all_test), len(global_preds))
            if n > 0:
                curves["global_aggregated"] = {
                    "observed": np.asarray(norm_all_test[:n], dtype=float),
                    "predicted": np.asarray(global_preds[:n], dtype=float),
                }

    return curves


def render_observed_vs_predicted_federated(curves: dict, key_prefix: str = "fed"):
    if not curves:
        st.info("No observed-vs-predicted curves available for the selected strategy.")
        return

    st.markdown("### Observed vs Predicted (Selected Strategy)")
    st.caption(
        "One panel per model (users + aggregated). Values are on the normalized scale used in federated training."
    )

    model_ids = sorted(curves.keys(), key=lambda m: (m != "global_aggregated", m))
    selected_models = st.multiselect(
        "Models for observed vs predicted",
        options=model_ids,
        default=model_ids,
        key=f"{key_prefix}_obs_pred_models",
    )
    if not selected_models:
        st.info("Select at least one model to render observed vs predicted plots.")
        return

    max_points = st.slider(
        "Max points per model curve",
        min_value=100,
        max_value=3000,
        value=600,
        step=100,
        key=f"{key_prefix}_obs_pred_max_points",
    )
    use_shared_scale = st.checkbox(
        "Use same y-axis scale on all observed-vs-predicted panels",
        value=True,
        key=f"{key_prefix}_obs_pred_shared_ylim",
    )

    n_models = len(selected_models)
    ncols = 2 if n_models > 1 else 1
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, max(4.0, 3.9 * nrows)))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    shared_y_min = float("inf")
    shared_y_max = float("-inf")

    for i, model_id in enumerate(selected_models):
        ax = axes_arr[i]
        model_curves = curves[model_id]
        observed = np.asarray(model_curves.get("observed", []), dtype=float)
        predicted = np.asarray(model_curves.get("predicted", []), dtype=float)

        n = min(len(observed), len(predicted))
        if n == 0:
            ax.set_title(f"{model_id} (no data)")
            ax.axis("off")
            continue

        observed = observed[:n]
        predicted = predicted[:n]
        x = np.arange(n)

        if n > max_points:
            sample_idx = np.linspace(0, n - 1, max_points, dtype=int)
            x = x[sample_idx]
            observed = observed[sample_idx]
            predicted = predicted[sample_idx]

        combined = np.concatenate([observed, predicted])
        finite_combined = combined[np.isfinite(combined)]
        if finite_combined.size:
            shared_y_min = min(shared_y_min, float(np.min(finite_combined)))
            shared_y_max = max(shared_y_max, float(np.max(finite_combined)))

        rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
        mae = float(np.mean(np.abs(observed - predicted)))

        ax.plot(x, observed, color="black", linewidth=1.35, label="Observed")
        ax.plot(x, predicted, color="#d62728", linewidth=1.5, linestyle="--", label="Predicted")
        ax.set_title(f"{model_id}  |  RMSE={rmse:.4f}  MAE={mae:.4f}")
        ax.set_xlabel("Test step")
        ax.set_ylabel("Normalized target")
        ax.grid(alpha=0.28, linestyle="--")

    if use_shared_scale and np.isfinite(shared_y_min) and np.isfinite(shared_y_max):
        span = shared_y_max - shared_y_min
        pad = 0.03 * span if span > 0 else 0.05 * max(abs(shared_y_max), 1.0)
        y_low = shared_y_min - pad
        y_high = shared_y_max + pad
        for i in range(n_models):
            axes_arr[i].set_ylim(y_low, y_high)

    for k in range(n_models, len(axes_arr)):
        axes_arr[k].axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Observed vs Predicted Curves Per Federated Model", y=1.01)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_federated_study_tab(df_for_training, cfg, impute_strategy, df_with_missing=None, fitted_imputer=None):
    st.subheader("Federated Learning Study")
    st.caption(
        "Simulate federated learning: partition data across users, train local models, "
        "aggregate at server, and compare users vs global model."
    )

    st.info(
        "This run executes all imputation strategies one-by-one. "
        "Each strategy gets its own clean result section with full model coverage. "
        f"Current sidebar strategy is not used for filtering here: **{_strategy_name(impute_strategy)}**"
    )
    source_df = df_with_missing if df_with_missing is not None else df_for_training

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        num_users = st.selectbox(
            "Number of users",
            options=[3, 5, 7, 10],
            index=1,
            help="Data is split chronologically into N users.",
            key="fed_num_users",
        )
    with col2:
        aggregation_method = st.radio(
            "Aggregation method",
            options=["mean", "median"],
            index=0,
            help="How to combine local model weights.",
            key="fed_agg_method",
        )
    with col3:
        num_rounds = st.slider(
            "Federated rounds",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            key="fed_num_rounds",
        )
    with col4:
        st.caption("Local epochs/round")
        st.markdown(f"**{cfg['epochs']}** (from sidebar)")
    with col5:
        live_focus_mode = st.selectbox(
            "Live focus",
            options=["Current model + aggregated", "All models"],
            index=0,
            key="fed_live_focus_mode",
            help="Control how many model panels are redrawn during live training.",
        )

    live_max_models = st.slider(
        "Live max model panels",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        key="fed_live_max_models",
        help="Limit live redraw complexity for smoother UI updates.",
    )

    run_federated = st.button(
        "▶ Run Federated Study For All Strategies",
        type="primary",
        key="btn_federated",
    )

    if run_federated:
        min_rows_per_user = 100
        fixed_fed = {
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
            "model_type": cfg["model_type"],
        }

        strategy_results = {}
        strategy_errors = {}
        total_strategies = len(STRATEGIES)

        fed_progress = st.progress(0, text="Preparing multi-strategy federated training...")
        fed_status_ph = st.empty()
        live_plot_ph = st.empty()

        with st.spinner("Running federated training for all imputation strategies..."):
            for s_idx, strategy in enumerate(STRATEGIES):
                label = _strategy_name(strategy)
                fed_status_ph.caption(f"Running strategy **{label}** ({s_idx + 1}/{total_strategies})")
                live_epoch_rows = []
                live_round_rows = []
                try:
                    df_imputed = _impute_for_strategy(source_df, strategy, fitted_imputer=fitted_imputer)
                    if cfg["target_col"] not in df_imputed.columns:
                        raise ValueError(f"Target column '{cfg['target_col']}' not found after imputation")

                    ts_full = df_imputed[cfg["target_col"]].astype(float).dropna()
                    if len(ts_full) < num_users * min_rows_per_user:
                        raise ValueError(
                            f"Need at least {num_users * min_rows_per_user} rows, got {len(ts_full)}"
                        )

                    user_data_dict = partition_data_for_users(
                        ts_full,
                        num_users=num_users,
                        test_size=cfg["test_size"],
                    )

                    def _fed_cb(done, total, inner_label, _s_idx=s_idx, _total_s=total_strategies, _label=label):
                        base = _s_idx / _total_s
                        frac = base + (done / max(total, 1)) / _total_s
                        fed_progress.progress(min(1.0, frac), text=f"{_label}: {inner_label}...")

                    def _round_cb(round_idx, model_id, row):
                        live_round_rows.append(row)
                        epoch_df = pd.DataFrame(live_epoch_rows)
                        round_df = pd.DataFrame(live_round_rows)
                        render_live_federated_training_dashboard(
                            epoch_df=epoch_df,
                            round_df=round_df,
                            strategy_label=label,
                            live_plot_ph=live_plot_ph,
                            local_epochs=int(cfg["epochs"]),
                            current_round=int(round_idx),
                            current_model=str(model_id),
                            focus_mode=live_focus_mode,
                            max_models=int(live_max_models),
                        )

                    def _live_epoch_cb(round_idx, model_id, epoch_idx, train_hist, val_hist):
                        train_last = float(train_hist[-1]) if train_hist else float("nan")
                        val_last = float(val_hist[-1]) if val_hist else float("nan")
                        live_epoch_rows.append(
                            {
                                "round": int(round_idx),
                                "model_id": str(model_id),
                                "epoch": int(epoch_idx),
                                "train_loss": train_last,
                                "val_loss": val_last,
                                "status": "ok",
                            }
                        )
                        if len(live_epoch_rows) % 2 != 0:
                            return
                        epoch_df = pd.DataFrame(live_epoch_rows)
                        round_df = pd.DataFrame(live_round_rows)
                        render_live_federated_training_dashboard(
                            epoch_df=epoch_df,
                            round_df=round_df,
                            strategy_label=label,
                            live_plot_ph=live_plot_ph,
                            local_epochs=int(cfg["epochs"]),
                            current_round=int(round_idx),
                            current_model=str(model_id),
                            focus_mode=live_focus_mode,
                            max_models=int(live_max_models),
                        )

                    fed_result = federated_training_study(
                        user_data_dict=user_data_dict,
                        fixed_params=fixed_fed,
                        aggregation_method=aggregation_method,
                        normalization=cfg["normalization"],
                        num_rounds=num_rounds,
                        local_epochs_per_round=cfg["epochs"],
                        progress_callback=_fed_cb,
                        epoch_callback=_live_epoch_cb,
                        round_callback=_round_cb,
                    )

                    fed_result["user_data_dict"] = user_data_dict

                    render_live_federated_training_dashboard(
                        epoch_df=fed_result.get("epoch_metrics_df", pd.DataFrame()),
                        round_df=fed_result.get("user_round_metrics_df", pd.DataFrame()),
                        strategy_label=label,
                        live_plot_ph=live_plot_ph,
                        local_epochs=int(cfg["epochs"]),
                        current_round=int(num_rounds),
                        current_model="global_aggregated",
                        focus_mode=live_focus_mode,
                        max_models=int(live_max_models),
                    )

                    if fed_result.get("error"):
                        strategy_errors[strategy] = str(fed_result.get("error"))
                    strategy_results[strategy] = fed_result
                except Exception as exc:
                    strategy_errors[strategy] = str(exc)

                fed_progress.progress((s_idx + 1) / total_strategies, text=f"Completed {label}")

        fed_progress.progress(1.0, text="All strategies completed.")
        fed_status_ph.caption("Live training complete for all strategies.")
        st.session_state["federated_all_strategies_result"] = {
            "strategy_results": strategy_results,
            "strategy_errors": strategy_errors,
            "user_ids": next(iter(strategy_results.values())).get("user_ids", []) if strategy_results else [],
            "num_rounds": num_rounds,
            "num_users": num_users,
            "aggregation_method": aggregation_method,
        }

    if "federated_all_strategies_result" in st.session_state:
        bundle = st.session_state["federated_all_strategies_result"]
        strategy_results = bundle.get("strategy_results", {})
        strategy_errors = bundle.get("strategy_errors", {})

        shown_rounds = int(bundle.get("num_rounds", num_rounds))
        shown_users = int(bundle.get("num_users", num_users))
        if shown_rounds != int(num_rounds) or shown_users != int(num_users):
            st.warning(
                "Displaying previously computed results. "
                f"Shown run: rounds={shown_rounds}, users={shown_users}. "
                f"Current controls: rounds={int(num_rounds)}, users={int(num_users)}. "
                "Click 'Run Federated Study For All Strategies' to refresh charts."
            )
        else:
            st.caption(
                f"Displaying latest run: rounds={shown_rounds}, users={shown_users}, "
                f"aggregation={bundle.get('aggregation_method', aggregation_method)}."
            )

        if strategy_errors:
            st.warning("Some strategies failed. See details below.")
            err_rows = [{"strategy": _strategy_name(s), "error": e} for s, e in strategy_errors.items()]
            st.dataframe(pd.DataFrame(err_rows), use_container_width=True)

        if not strategy_results:
            st.error("No strategy produced a valid federated result.")
            return

        st.markdown("### Strategy-by-Strategy Results")
        for s_idx, strategy in enumerate(STRATEGIES):
            if strategy not in strategy_results:
                continue

            label = _strategy_name(strategy)
            st.markdown(f"## {label}")
            chosen = strategy_results[strategy]
            key_prefix = f"fed_{strategy}"

            strategy_results_df = chosen.get("results_df", pd.DataFrame())
            if strategy_results_df is not None and not strategy_results_df.empty:
                st.markdown("### Final Metrics Table")
                st.dataframe(
                    strategy_results_df.sort_values("model_id").style.format(
                        {
                            "MSE": "{:,.6f}",
                            "MAE": "{:,.6f}",
                            "MAPE (%)": "{:,.4f}",
                            "RMSE": "{:,.6f}",
                            "train_loss_final": "{:,.6f}",
                            "val_loss_final": "{:,.6f}",
                            "runtime_sec": "{:,.2f}",
                        },
                        na_rep="N/A",
                    ),
                    use_container_width=True,
                )

            strategy_user_round_df = chosen.get("user_round_metrics_df", pd.DataFrame())
            render_strategy_round_metric_overview(
                strategy_user_round_df,
                strategy_label=label,
                key_prefix=key_prefix,
            )

            try:
                selected_user_data = chosen.get("user_data_dict")
                if not selected_user_data:
                    df_imputed_selected = _impute_for_strategy(source_df, strategy, fitted_imputer=fitted_imputer)
                    if cfg["target_col"] in df_imputed_selected.columns:
                        ts_selected = df_imputed_selected[cfg["target_col"]].astype(float).dropna()
                        selected_user_data = partition_data_for_users(
                            ts_selected,
                            num_users=int(bundle.get("num_users", 5)),
                            test_size=cfg["test_size"],
                        )

                if selected_user_data:
                    prediction_curves = _build_federated_prediction_curves(chosen, selected_user_data)
                    render_observed_vs_predicted_federated(
                        prediction_curves,
                        key_prefix=f"{key_prefix}_obs",
                    )
                else:
                    st.warning(
                        f"Observed-vs-predicted unavailable for {label}: could not prepare user partitions."
                    )
            except Exception as exc:
                st.warning(f"Observed-vs-predicted unavailable for {label}: {exc}")

            if s_idx < len(STRATEGIES) - 1:
                st.divider()
