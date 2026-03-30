import streamlit as st

from data_processing import (
    STRATEGIES,
    RF_IMPUTER_CACHE,
    load_imputer,
    train_and_save_imputer,
)

from .constants import STRATEGY_LABELS


def render_sidebar(full_df, all_cols, total_rows):
    with st.sidebar:
        st.header("Dataset")
        max_rows = st.number_input(
            "Rows to use (most recent)",
            min_value=1_000,
            max_value=total_rows,
            value=min(4000, total_rows),
            step=1_000,
            help=(
                "Selects the most recent N rows from the full 5 M-row dataset. "
                "Smaller values speed up GRU training and sweep runs."
            ),
        )
        df_clean = full_df.iloc[-max_rows:].copy()

        st.header("Forecast target")
        target_col = st.selectbox(
            "Select column to forecast",
            options=all_cols,
            index=all_cols.index("bed1") if "bed1" in all_cols else 0,
            help="Only the selected column is used as the univariate forecast series.",
        )

        st.header("Missing value injection (MCAR)")
        st.caption(
            "Randomly inject NaN values into the forecast target column at the chosen percentage. "
            "Seed is fixed so positions are reproducible for the same rate."
        )
        missing_rates = {}
        if target_col in df_clean.columns:
            pct = st.slider(f"{target_col} missing %", 0, 50, 50, step=1, key=f"miss_{target_col}")
            missing_rates[target_col] = pct / 100.0

        st.header("Predictive imputer cache")
        cache_exists = RF_IMPUTER_CACHE.exists()
        st.caption(
            f"Cache: `{RF_IMPUTER_CACHE.name}` — "
            + ("**ready ✓**" if cache_exists else "*not trained yet*")
        )
        if st.button(
            "Pre-train & save RF imputer",
            key="btn_pretrain_rf",
            help=(
                "Fits the RandomForest IterativeImputer on the current clean dataset "
                "and saves it to disk. Reuse happens automatically when the "
                "'Predictive (RandomForest)' strategy is selected."
            ),
        ):
            with st.spinner("Training RF imputer (this may take a minute)…"):
                df_fit = full_df.iloc[-max_rows:].copy().select_dtypes(include=["number"]).dropna()
                train_and_save_imputer(df_fit, cache_path=RF_IMPUTER_CACHE)
                st.session_state["rf_imputer"] = load_imputer(RF_IMPUTER_CACHE)
            st.success("RF imputer trained and saved.")
            st.rerun()

        if "rf_imputer" not in st.session_state and cache_exists:
            st.session_state["rf_imputer"] = load_imputer(RF_IMPUTER_CACHE)

        st.header("Imputation for training")
        impute_strategy = st.selectbox(
            "Strategy applied before GRU training",
            options=STRATEGIES,
            index=STRATEGIES.index("fill_mean"),
            format_func=lambda s: STRATEGY_LABELS.get(s, s),
            help=(
                "Missing values in the selected target column are filled with this "
                "strategy before the series is passed to the GRU trainer."
            ),
        )

        st.header("Train / test split")
        test_pct = st.slider("Test set size %", 10, 40, 20, step=5)
        test_size = test_pct / 100.0
        val_pct = st.slider("Validation set size %", 5, 30, 10, step=5)
        val_fraction_val = val_pct / 100.0
        future_steps_val = st.slider(
            "Future forecast steps",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help=(
                "Number of recursive steps predicted beyond the test set. "
                "Fewer steps = less error compounding and more realistic forecast."
            ),
        )

        st.header("Model")
        model_type_val = st.selectbox(
            "Architecture",
            options=["gru", "lstm", "rnn"],
            index=0,
            format_func=lambda s: s.upper(),
            help="Recurrent architecture used for training and forecasting.",
        )

        st.header("Hyperparameters")
        norm_mode = st.selectbox(
            "Normalization",
            ["minmax", "zscore", "none"],
            index=0,
            help="Scaling applied to the training series before GRU training.",
        )
        seq_len_val = st.select_slider(
            "Sequence length (seq_len)",
            options=[4, 8, 16, 32, 60, 128, 256],
            value=4,
        )
        hidden_val = st.select_slider(
            "Hidden units",
            options=[4, 16, 32, 64, 128, 256],
            value=16,
        )
        epochs_val = st.slider("Epochs", 10, 200, 100, step=10)
        lr_val = st.select_slider(
            "Learning rate",
            options=[0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.01,
            format_func=lambda x: f"{x:g}",
        )

        st.header("Network architecture")
        num_layers_val = st.slider(
            "Layers",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
            help="Number of stacked recurrent layers.",
        )
        activation_val = st.selectbox(
            "Activation function",
            options=["relu", "sigmoid", "tanh", "leaky_relu"],
            index=0,
            help="Applied to the GRU output before the linear head.",
        )
        dropout_val = st.number_input(
            "Dropout rate",
            min_value=0.0,
            max_value=0.9,
            value=0.0,
            step=0.05,
            format="%.2f",
            help="Dropout applied after the activation, before the linear head.",
        )

        st.header("Optimizer & loss")
        optimizer_val = st.selectbox(
            "Optimizer",
            options=["adam", "sgd", "rmsprop"],
            index=0,
        )
        loss_fn_val = st.selectbox(
            "Loss function",
            options=["mse", "rmse", "mae", "huber"],
            index=0,
            help="'rmse' tracks sqrt(MSE) loss and 'mae' uses L1Loss.",
        )
        weight_decay_val = st.number_input(
            "L2 weight decay",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=1e-4,
            format="%.4f",
            help="L2 regularization passed to the optimizer as weight_decay.",
        )
        l1_lambda_val = st.number_input(
            "L1 lasso (λ)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=1e-4,
            format="%.4f",
            help="L1 regularization added to the training loss each step.",
        )

        st.header("Learning rate schedule")
        lr_scheduler_val = st.selectbox(
            "Learning rate scheduler",
            options=["constant", "step", "exponential", "cosine", "warmup_decay"],
            index=3,  # Default to cosine annealing (often works well)
            help="Strategy for adjusting learning rate during training. "
            "'constant' = fixed LR, 'step' = discrete drops, 'exponential' = smooth decay, "
            "'cosine' = cosine annealing, 'warmup_decay' = linear warmup then decay.",
        )

        # Conditional parameters based on scheduler type
        lr_scheduler_kwargs = {}
        if lr_scheduler_val == "step":
            step_size = st.slider(
                "Step size (epochs)",
                min_value=1,
                max_value=50,
                value=max(1, epochs_val // 4),
                step=1,
                help="Number of epochs between each learning rate reduction.",
            )
            gamma = st.slider(
                "Multiplicative factor (gamma)",
                min_value=0.01,
                max_value=0.99,
                value=0.1,
                step=0.01,
                help="Factor to multiply LR by at each step.",
            )
            lr_scheduler_kwargs = {"step_size": step_size, "gamma": gamma}
        elif lr_scheduler_val == "exponential":
            decay_rate = st.slider(
                "Decay rate",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.001,
                help="Rate of exponential decay per epoch.",
            )
            lr_scheduler_kwargs = {"decay_rate": decay_rate}
        elif lr_scheduler_val == "cosine":
            min_lr = st.number_input(
                "Minimum learning rate",
                min_value=1e-8,
                max_value=1e-3,
                value=1e-6,
                step=1e-7,
                format="%.2e",
                help="Minimum LR reached at the end of cosine annealing.",
            )
            lr_scheduler_kwargs = {"min_lr": min_lr}
        elif lr_scheduler_val == "warmup_decay":
            warmup_epochs = st.slider(
                "Warmup epochs",
                min_value=1,
                max_value=max(2, epochs_val // 5),
                value=max(1, epochs_val // 10),
                step=1,
                help="Number of epochs to linearly increase LR.",
            )
            warmup_decay_type = st.selectbox(
                "Decay type (after warmup)",
                options=["cosine", "step", "exponential"],
                index=0,
                help="Type of decay schedule to apply after warmup phase.",
            )
            lr_scheduler_kwargs = {
                "warmup_epochs": warmup_epochs,
                "decay_type": warmup_decay_type,
            }

    return {
        "max_rows": max_rows,
        "df_clean": df_clean,
        "target_col": target_col,
        "missing_rates": missing_rates,
        "impute_strategy": impute_strategy,
        "test_size": test_size,
        "val_fraction": val_fraction_val,
        "future_steps": future_steps_val,
        "model_type": model_type_val,
        "normalization": norm_mode,
        "seq_len": seq_len_val,
        "hidden": hidden_val,
        "epochs": epochs_val,
        "lr": lr_val,
        "num_layers": num_layers_val,
        "activation": activation_val,
        "dropout": dropout_val,
        "optimizer_name": optimizer_val,
        "loss_fn_name": loss_fn_val,
        "weight_decay": weight_decay_val,
        "l1_lambda": l1_lambda_val,
        "lr_scheduler_type": lr_scheduler_val,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
    }
