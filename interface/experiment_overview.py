import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from forecasting.studies.experiments import model_strategy_overview
from forecasting.utils.common import train_test_split_series
from data_processing import STRATEGIES, process_for_viz

from .common import available_metrics
from .constants import METRIC_COLUMNS, STRATEGY_LABELS


def render_model_strategy_overview(results_df, bar_metric="RMSE", heatmap_metrics=None):
    if results_df is None or results_df.empty:
        st.info("No overview results yet. Run the experiment to see comparisons.")
        return

    existing = available_metrics(results_df, METRIC_COLUMNS)
    if not existing:
        st.warning("Overview result has no metric columns to plot.")
        return

    hm_metrics = [m for m in (heatmap_metrics or existing) if m in existing]
    if not hm_metrics:
        hm_metrics = existing[:1]

    st.markdown("### Overview results")

    display_cols = ["strategy", "model_type", *existing]
    for extra in ["train_loss_final", "val_loss_final", "runtime_sec", "status"]:
        if extra in results_df.columns:
            display_cols.append(extra)

    table_df = results_df[display_cols].copy()
    st.dataframe(
        table_df.set_index(["strategy", "model_type"])
        .style.format(
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
        )
        .highlight_min(subset=existing, axis=0, color="#d4edda")
        .highlight_max(subset=existing, axis=0, color="#f8d7da"),
        width="stretch",
    )

    n = len(hm_metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for idx, metric in enumerate(hm_metrics):
        ax = axes_flat[idx]
        pivot = results_df.pivot_table(index="strategy", columns="model_type", values=metric, aggfunc="mean")
        if pivot.empty:
            ax.set_title(f"{metric} (no data)")
            ax.axis("off")
            continue

        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_title(f"{metric} heatmap")
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.iat[r, c]
                txt = "N/A" if np.isnan(v) else f"{v:.4f}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    chosen = bar_metric if bar_metric in existing else existing[0]
    pivot_bar = results_df.pivot_table(index="strategy", columns="model_type", values=chosen, aggfunc="mean")
    if pivot_bar.empty:
        st.info("No data available for grouped bar chart.")
        return

    x = np.arange(len(pivot_bar.index))
    models = pivot_bar.columns.tolist()
    width = 0.8 / max(len(models), 1)

    fig2, ax2 = plt.subplots(figsize=(10, 4.2))
    for i, model in enumerate(models):
        offsets = x + (i - len(models) / 2 + 0.5) * width
        ax2.bar(offsets, pivot_bar[model].values, width=width, label=model.upper())

    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_bar.index.tolist(), rotation=20, ha="right")
    ax2.set_ylabel(chosen)
    ax2.set_title(f"Model vs strategy comparison ({chosen})")
    ax2.legend(fontsize="small", ncols=min(len(models), 3))
    plt.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close(fig2)


def render_experiment_overview_tab(df_with_missing, cfg):
    st.subheader("Experiment Overview")
    st.caption(
        "Run selected model architectures across selected imputation strategies "
        "to compare performance in a single matrix view."
    )

    selected_models = st.multiselect(
        "Models",
        options=["gru", "lstm", "rnn"],
        default=["gru", "lstm", "rnn"],
        format_func=lambda s: s.upper(),
        key="overview_models",
    )
    selected_strategies = st.multiselect(
        "Imputation strategies",
        options=STRATEGIES,
        default=STRATEGIES,
        format_func=lambda s: STRATEGY_LABELS.get(s, s),
        key="overview_strategies",
    )

    metric_opts = METRIC_COLUMNS
    bar_metric = st.selectbox(
        "Metric for grouped bar chart",
        options=metric_opts,
        index=metric_opts.index("RMSE"),
        key="overview_bar_metric",
    )
    heatmap_metrics = st.multiselect(
        "Heatmaps to render",
        options=metric_opts,
        default=["MSE", "RMSE"],
        key="overview_heatmap_metrics",
    )

    run_count = len(selected_models) * len(selected_strategies)
    st.caption(f"Planned runs: **{run_count}** ({len(selected_models)} models x {len(selected_strategies)} strategies)")
    if run_count * cfg["epochs"] > 1200:
        st.warning(
            "Large run detected. Consider reducing rows, epochs, models, or strategies "
            "for faster iteration."
        )

    run_overview = st.button("▶ Run Overview Experiment", type="primary", key="btn_overview")

    if run_overview:
        if not selected_models:
            st.warning("Select at least one model.")
        elif not selected_strategies:
            st.warning("Select at least one imputation strategy.")
        else:
            strategy_data = {}
            skipped = []

            for strat in selected_strategies:
                try:
                    fitted_imp = st.session_state.get("rf_imputer") if strat == "predictive_imputer" else None
                    if strat == "predictive_imputer" and fitted_imp is None:
                        st.info(
                            "Predictive strategy selected without cached imputer. "
                            "This run will fit on the fly and may be slow."
                        )

                    _, df_imp, _ = process_for_viz(df_with_missing, strategy=strat, fitted_imputer=fitted_imp)
                    if cfg["target_col"] not in df_imp.columns:
                        skipped.append(f"{strat} (target column unavailable)")
                        continue

                    ts_full = df_imp[cfg["target_col"]].astype(float).dropna()
                    train_s, test_s = train_test_split_series(ts_full, test_size=cfg["test_size"])
                    train_s = train_s.dropna()

                    if train_s.empty or test_s.empty:
                        skipped.append(f"{strat} (insufficient train/test data)")
                        continue

                    strategy_data[strat] = (train_s.values, test_s.values)
                except Exception as exc:
                    skipped.append(f"{strat} ({exc})")

            if not strategy_data:
                st.error("No valid strategy datasets were prepared. Nothing to run.")
            else:
                if skipped:
                    st.warning("Skipped strategies: " + " | ".join(skipped))

                fixed_overview = {
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

                overview_progress = st.progress(0, text="Preparing runs…")
                st.markdown("**Live training loss (current run)**")
                overview_loss_ph = st.empty()
                overview_status_ph = st.empty()

                def _overview_cb(done, total, label):
                    overview_progress.progress(done / max(total, 1), text=f"Training {label}…")
                    overview_loss_ph.empty()

                def _overview_epoch_cb(epoch_idx, train_l, val_l, _epochs=cfg["epochs"]):
                    overview_status_ph.caption(
                        f"Epoch **{epoch_idx + 1} / {_epochs}** — "
                        f"train loss: `{train_l[-1]:.6f}`"
                        + (f"  |  val loss: `{val_l[-1]:.6f}`" if val_l else "")
                    )
                    import matplotlib.pyplot as _plt
                    import matplotlib.ticker as _mticker

                    fig, ax = _plt.subplots(figsize=(7, 2.2))
                    ax.plot(train_l, color="tab:blue", lw=1.5, label="Train")
                    if val_l:
                        ax.plot(val_l, color="tab:orange", lw=1.5, label="Val")
                    ax.set_title(f"Epoch {epoch_idx + 1}/{_epochs}", fontsize=9)
                    ax.set_xlabel("Epoch", fontsize=8)
                    ax.set_ylabel("MSE (log scale)", fontsize=8)
                    ax.set_yscale("log")
                    ax.yaxis.set_major_formatter(
                        _mticker.FuncFormatter(lambda x, _: f"{x:.6f}".rstrip("0").rstrip("."))
                    )
                    ax.tick_params(labelsize=7)
                    ax.legend(fontsize="x-small")
                    _plt.tight_layout()
                    overview_loss_ph.pyplot(fig, width="stretch")
                    _plt.close(fig)

                with st.spinner("Running model x strategy overview experiment…"):
                    overview_df = model_strategy_overview(
                        strategy_data=strategy_data,
                        model_types=selected_models,
                        fixed_params=fixed_overview,
                        normalization=cfg["normalization"],
                        progress_callback=_overview_cb,
                        epoch_callback=_overview_epoch_cb,
                    )

                overview_progress.progress(1.0, text="Done.")
                st.session_state["overview_result"] = overview_df

    if "overview_result" in st.session_state:
        render_model_strategy_overview(
            st.session_state["overview_result"],
            bar_metric=bar_metric,
            heatmap_metrics=heatmap_metrics,
        )
