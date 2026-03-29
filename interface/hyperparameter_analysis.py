import matplotlib.pyplot as plt
import streamlit as st

from forecasting.studies.experiments import hyperparameter_sweep
from forecasting.utils.common import train_test_split_series

from .common import available_metrics, finite_best_index
from .constants import METRIC_COLUMNS, SWEEP_VALUES


def render_hyperparameter_analysis(sweep_df, param_name):
    existing = available_metrics(sweep_df, METRIC_COLUMNS)

    st.markdown(f"### Hyperparameter analysis — {param_name}")
    st.dataframe(
        sweep_df.set_index(param_name)[existing]
        .style.format("{:,.6f}")
        .highlight_min(axis=0, color="#d4edda")
        .highlight_max(axis=0, color="#f8d7da"),
        use_container_width=True,
    )

    fig, axes = plt.subplots(1, len(existing), figsize=(4 * len(existing), 4))
    if len(existing) == 1:
        axes = [axes]

    x_vals = sweep_df[param_name].values
    for ax, lbl in zip(axes, existing):
        y_vals = sweep_df[lbl].values
        ax.plot(x_vals, y_vals, marker="o", color="tab:blue")
        best_idx = finite_best_index(y_vals)
        if best_idx is not None:
            ax.scatter(
                [x_vals[best_idx]],
                [y_vals[best_idx]],
                color="green",
                zorder=5,
                label=f"Best: {x_vals[best_idx]}",
            )
        ax.set_title(lbl)
        ax.set_xlabel(param_name)
        ax.set_ylabel(lbl)
        ax.legend(fontsize="small")
        if param_name == "lr":
            ax.set_xscale("log")

    plt.suptitle(f"GRU — {param_name} sweep", fontsize=11, y=1.01)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    best_row = sweep_df.loc[sweep_df["MSE"].idxmin()]
    st.markdown(
        f"**Best {param_name}** (min MSE): `{best_row[param_name]}` → "
        f"MSE={best_row['MSE']:,.6f}, MAE={best_row['MAE']:,.6f}, "
        f"RMSE={best_row['RMSE']:,.6f}, MAPE={best_row['MAPE (%)']:,.4f}%"
    )


def render_hyperparameter_analysis_tab(df_for_training, cfg, impute_strategy):
    st.subheader("Hyperparameter Analysis")
    st.caption(
        "Sweep one hyperparameter across a predefined range while keeping the "
        "others fixed at the sidebar values. Each value trains a fresh GRU."
    )

    param_label_map = {
        "Learning Rate (lr)": "lr",
        "Hidden Units (hidden)": "hidden",
        "Sequence Length (seq_len)": "seq_len",
    }
    sweep_label = st.selectbox("Hyperparameter to sweep", list(param_label_map.keys()), key="sweep_select")
    sweep_param = param_label_map[sweep_label]
    sweep_values = SWEEP_VALUES[sweep_param]
    st.caption(f"Will train {len(sweep_values)} GRU models with {sweep_param} ∈ {sweep_values}")

    run_sweep = st.button("▶ Run Hyperparameter Sweep", type="primary", key="btn_sweep")
    st.info(f"Imputation used for training: **{impute_strategy}**")

    if run_sweep:
        ts_full = df_for_training[cfg["target_col"]].astype(float).dropna()
        train_s, test_s = train_test_split_series(ts_full, test_size=cfg["test_size"])
        train_s = train_s.dropna()

        fixed = {
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
        fixed.pop(sweep_param, None)

        progress_bar = st.progress(0, text="Running sweep…")
        st.markdown("**Live training loss (current run)**")
        sweep_loss_ph = st.empty()
        sweep_status_ph = st.empty()

        def _cb(done, total, label):
            progress_bar.progress(done / max(total, 1), text=f"Training {label}…")
            sweep_loss_ph.empty()

        def _sweep_epoch_cb(epoch_idx, train_l, val_l, _epochs=cfg["epochs"]):
            sweep_status_ph.caption(
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
            sweep_loss_ph.pyplot(fig, use_container_width=True)
            _plt.close(fig)

        with st.spinner("Running hyperparameter sweep…"):
            sweep_df = hyperparameter_sweep(
                train_s.values,
                test_s.values,
                param_name=sweep_param,
                param_values=sweep_values,
                fixed_params=fixed,
                normalization=cfg["normalization"],
                progress_callback=_cb,
                epoch_callback=_sweep_epoch_cb,
            )

        progress_bar.progress(1.0, text="Done.")
        st.session_state["sweep_result"] = {"df": sweep_df, "param": sweep_param}

    if "sweep_result" in st.session_state:
        sr = st.session_state["sweep_result"]
        render_hyperparameter_analysis(sr["df"], sr["param"])
