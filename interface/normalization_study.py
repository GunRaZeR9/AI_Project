import matplotlib.pyplot as plt
import streamlit as st

from forecasting.studies.experiments import normalization_study
from forecasting.utils.common import train_test_split_series

from .common import available_metrics
from .constants import METRIC_COLUMNS


def render_normalization_study(norm_df):
    existing = available_metrics(norm_df, METRIC_COLUMNS)

    st.markdown("### Normalization study")
    st.dataframe(
        norm_df[existing]
        .style.format("{:,.6f}")
        .highlight_min(axis=0, color="#d4edda")
        .highlight_max(axis=0, color="#f8d7da"),
        use_container_width=True,
    )

    n_groups = len(norm_df)
    n_metrics = len(existing)
    x = range(n_groups)
    bar_width = 0.8 / n_metrics
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(9, 4))
    import numpy as np

    x_vals = np.arange(n_groups)
    for i, (lbl, color) in enumerate(zip(existing, colors)):
        offsets = x_vals + (i - n_metrics / 2 + 0.5) * bar_width
        ax.bar(offsets, norm_df[lbl].values, width=bar_width, label=lbl, color=color)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(norm_df.index.tolist())
    ax.set_title("GRU — impact of normalization on evaluation metrics")
    ax.set_ylabel("Metric value")
    ax.legend(fontsize="small")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    best_mode = norm_df["MSE"].idxmin()
    worst_mode = norm_df["MSE"].idxmax()
    st.markdown(f"**Best normalization** (min MSE): **{best_mode}**.  **Worst**: {worst_mode}.")


def render_normalization_study_tab(df_for_training, cfg, impute_strategy):
    st.subheader("Normalization Study")
    st.caption(
        "Trains the GRU three times — once with Min-Max scaling, once with "
        "Z-score standardisation, and once with no scaling — then compares "
        "all four evaluation metrics."
    )
    run_norm_study = st.button("▶ Run Normalization Study", type="primary", key="btn_norm")
    st.info(f"Imputation used for training: **{impute_strategy}**")

    if run_norm_study:
        ts_full = df_for_training[cfg["target_col"]].astype(float).dropna()
        train_s, test_s = train_test_split_series(ts_full, test_size=cfg["test_size"])
        train_s = train_s.dropna()

        fixed_norm = {
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

        norm_progress = st.progress(0, text="Training…")
        st.markdown("**Live training loss (current run)**")
        norm_loss_ph = st.empty()
        norm_status_ph = st.empty()

        def _norm_cb(done, total, label):
            norm_progress.progress(done / max(total, 1), text=f"Training with {label}…")
            norm_loss_ph.empty()

        def _norm_epoch_cb(epoch_idx, train_l, val_l, _epochs=cfg["epochs"]):
            norm_status_ph.caption(
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
            ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda x, _: f"{x:.6f}".rstrip("0").rstrip(".")))
            ax.tick_params(labelsize=7)
            ax.legend(fontsize="x-small")
            _plt.tight_layout()
            norm_loss_ph.pyplot(fig, use_container_width=True)
            _plt.close(fig)

        with st.spinner("Running normalization study…"):
            norm_df = normalization_study(
                train_s.values,
                test_s.values,
                fixed_params=fixed_norm,
                progress_callback=_norm_cb,
                epoch_callback=_norm_epoch_cb,
            )

        norm_progress.progress(1.0, text="Done.")
        st.session_state["norm_result"] = norm_df

    if "norm_result" in st.session_state:
        render_normalization_study(st.session_state["norm_result"])
