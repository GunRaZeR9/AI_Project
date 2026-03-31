import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from forecasting.inference.predictor import gru_forecast, gru_one_step_predict
from forecasting.metrics.evaluator import compute_metrics
from forecasting.training.trainer import train_gru
from forecasting.utils.common import train_test_split_series

from .common import DECIMAL_FMT

DEFAULT_GRU_PARAMS = {
    "batch_size": 64,
    "max_train_points": 12_000,
}


def _gru_train_and_plot(
    train_vals,
    seq_len,
    hidden,
    epochs,
    lr,
    normalization,
    loss_placeholder,
    status_placeholder,
    num_layers=2,
    activation="relu",
    dropout=0.0,
    optimizer_name="adam",
    loss_fn_name="mse",
    weight_decay=0.0,
    l1_lambda=0.0,
    val_fraction=0.1,
    model_type="gru",
    lr_scheduler_type="constant",
    lr_scheduler_kwargs=None,
):
    def _epoch_cb(epoch_idx, train_l, val_l):
        status_placeholder.caption(
            f"Epoch **{epoch_idx + 1} / {epochs}** — "
            f"train loss: `{train_l[-1]:.6f}`"
            + (f"  |  val loss: `{val_l[-1]:.6f}`" if val_l else "")
        )
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(train_l, color="tab:blue", lw=1.5, label="Train loss")
        if val_l:
            ax.plot(val_l, color="tab:orange", lw=1.5, label="Val loss")
        ax.set_title(f"Live training loss — epoch {epoch_idx + 1}/{epochs}", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("MSE (log scale)", fontsize=8)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(DECIMAL_FMT)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize="small")
        plt.tight_layout()
        loss_placeholder.pyplot(fig, width="stretch")
        plt.close(fig)

    return train_gru(
        train_vals,
        seq_len=seq_len,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        normalization=normalization,
        num_layers=num_layers,
        activation=activation,
        dropout=dropout,
        optimizer_name=optimizer_name,
        loss_fn_name=loss_fn_name,
        weight_decay=weight_decay,
        l1_lambda=l1_lambda,
        val_fraction=val_fraction,
        model_type=model_type,
        epoch_callback=_epoch_cb,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs or {},
        **DEFAULT_GRU_PARAMS,
    )


def render_forecasts(
    df,
    target_col,
    test_size=0.2,
    normalization="minmax",
    seq_len=60,
    hidden=64,
    epochs=40,
    lr=1e-3,
    num_layers=2,
    activation="relu",
    dropout=0.0,
    optimizer_name="adam",
    loss_fn_name="mse",
    weight_decay=0.0,
    l1_lambda=0.0,
    val_fraction=0.1,
    model_type="gru",
    future_steps=50,
    lr_scheduler_type="constant",
    lr_scheduler_kwargs=None,
):
    if target_col not in df.columns:
        st.error(f"Column '{target_col}' not found.")
        return

    ts = df[target_col].astype(float).dropna()
    if len(ts) < 100:
        st.warning("Not enough data to forecast (need ≥ 100 rows without NaN).")
        return

    train, test = train_test_split_series(ts, test_size=test_size)
    train = train.dropna()
    if train.empty:
        st.warning("Training set is empty after dropping NaNs.")
        return

    st.markdown("**Live training loss**")
    loss_ph = st.empty()
    status_ph = st.empty()
    with st.spinner("Training GRU model…"):
        result = _gru_train_and_plot(
            train.values,
            seq_len=seq_len,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            normalization=normalization,
            loss_placeholder=loss_ph,
            status_placeholder=status_ph,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            optimizer_name=optimizer_name,
            loss_fn_name=loss_fn_name,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            val_fraction=val_fraction,
            model_type=model_type,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
    status_ph.empty()

    runtime_device = str(result.get("device", "cpu")).lower()
    st.caption(f"Runtime device used: **{runtime_device.upper()}**")

    full_series = np.concatenate([train.values, test.values])
    test_preds = gru_one_step_predict(result, full_series, start_idx=len(train))
    future_preds = gru_forecast(result, full_series, steps=future_steps)

    metrics = compute_metrics(test.values, test_preds)
    st.markdown("**Evaluation metrics (test set)**")
    m_cols = st.columns(4)
    for col_ui, (lbl, val) in zip(m_cols, metrics.items()):
        col_ui.metric(lbl, f"{val:,.6f}" if not np.isnan(val) else "N/A")

    x_train = np.arange(len(train))
    x_test = np.arange(len(train), len(train) + len(test))
    x_future = np.arange(len(train) + len(test), len(train) + len(test) + future_steps)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_train[-500:], train.values[-500:], color="tab:blue", lw=1, label="Train (last 500)")
    ax.plot(x_test, test.values, color="black", lw=1, label="Test (actual)")
    ax.plot(x_test, test_preds, color="red", lw=1.5, linestyle="--", label="Test (predicted)")
    ax.plot(
        x_future,
        future_preds,
        color="darkorange",
        lw=1.5,
        linestyle=":",
        label=f"Future ({future_steps} steps)",
    )
    ax.axvline(len(train), color="gray", lw=0.8, linestyle="--")
    ax.axvline(len(train) + len(test), color="gray", lw=0.8, linestyle="--")
    ax.set_title(
        f"{model_type.upper()} Forecast — {target_col}  |  norm={normalization}\n"
        f"epochs={epochs}  lr={lr:g}  hidden={hidden}  seq_len={seq_len}  "
        f"activation={activation}  dropout={dropout}  optimiser={optimizer_name}  "
        f"loss={loss_fn_name}  wd={weight_decay}  l1={l1_lambda}",
        fontsize=9,
    )
    ax.set_ylabel(target_col)
    ax.set_xlabel("Time step")
    ax.legend(fontsize="small")
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    train_loss = result.get("train_loss", [])
    val_loss = result.get("val_loss", [])
    if train_loss:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(train_loss, label="Train loss", color="tab:blue", lw=1.5)
        if val_loss:
            ax2.plot(val_loss, label="Val loss", color="tab:orange", lw=1.5)
        ax2.set_title("GRU Training / Validation Loss — final")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE Loss (log scale)")
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(DECIMAL_FMT)
        ax2.legend(fontsize="small")
        plt.tight_layout()
        loss_ph.pyplot(fig2, width="stretch")
        plt.close(fig2)


def render_gru_forecast_tab(df_for_training, cfg, impute_strategy):
    st.subheader("GRU Forecast")
    st.markdown(
        r"""
        **Architecture:** one-step-ahead sliding-window GRU:

        $$h_t = \text{GRU}(x_t,\, h_{t-1}), \quad \hat{x}_{t+1} = W h_t + b$$

        Predictions on the test set are generated recursively.
        """
    )

    run_forecast = st.button("▶ Run Forecast", type="primary", key="btn_forecast")
    if run_forecast:
        st.session_state["forecast_cfg"] = {
            "target_col": cfg["target_col"],
            "test_size": cfg["test_size"],
            "normalization": cfg["normalization"],
            "seq_len": cfg["seq_len"],
            "hidden": cfg["hidden"],
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "num_layers": cfg["num_layers"],
            "activation": cfg["activation"],
            "dropout": cfg["dropout"],
            "optimizer_name": cfg["optimizer_name"],
            "loss_fn_name": cfg["loss_fn_name"],
            "weight_decay": cfg["weight_decay"],
            "l1_lambda": cfg["l1_lambda"],
            "val_fraction": cfg["val_fraction"],
            "model_type": cfg["model_type"],
            "future_steps": cfg["future_steps"],
            "lr_scheduler_type": cfg.get("lr_scheduler_type", "constant"),
            "lr_scheduler_kwargs": cfg.get("lr_scheduler_kwargs", {}),
        }

    if "forecast_cfg" in st.session_state:
        state_cfg = st.session_state["forecast_cfg"]
        st.caption(
            f"Results — {state_cfg['target_col']}  |  "
            f"norm={state_cfg['normalization']}  |  test {int(state_cfg['test_size'] * 100)} %"
        )
        st.info(f"Imputation used for training: **{impute_strategy}**")
        render_forecasts(
            df_for_training,
            target_col=state_cfg["target_col"],
            test_size=state_cfg["test_size"],
            normalization=state_cfg["normalization"],
            seq_len=state_cfg["seq_len"],
            hidden=state_cfg["hidden"],
            epochs=state_cfg["epochs"],
            lr=state_cfg["lr"],
            num_layers=state_cfg["num_layers"],
            activation=state_cfg["activation"],
            dropout=state_cfg["dropout"],
            optimizer_name=state_cfg["optimizer_name"],
            loss_fn_name=state_cfg["loss_fn_name"],
            weight_decay=state_cfg["weight_decay"],
            l1_lambda=state_cfg["l1_lambda"],
            val_fraction=state_cfg["val_fraction"],
            model_type=state_cfg["model_type"],
            future_steps=state_cfg["future_steps"],
            lr_scheduler_type=state_cfg.get("lr_scheduler_type", "constant"),
            lr_scheduler_kwargs=state_cfg.get("lr_scheduler_kwargs", {}),
        )
