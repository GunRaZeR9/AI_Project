from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from forecasting.inference.predictor import gru_forecast, gru_one_step_predict
from forecasting.metrics.evaluator import compute_metrics
from forecasting.training.trainer import train_gru
from forecasting.utils.common import _get_device


def hyperparameter_sweep(
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    param_name: str,
    param_values: List[Any],
    fixed_params: Dict[str, Any],
    normalization: str = "minmax",
    progress_callback=None,
    epoch_callback=None,
) -> pd.DataFrame:
    """Train separate models sweeping *param_name* over *param_values*."""
    rows = []
    total = len(param_values)

    for i, val in enumerate(param_values):
        if progress_callback:
            progress_callback(i, total, f"{param_name}={val}")

        kwargs = {k: v for k, v in fixed_params.items()}
        kwargs[param_name] = val
        kwargs["normalization"] = normalization
        kwargs["epoch_callback"] = epoch_callback

        try:
            result = train_gru(train_vals, **kwargs)
            test_preds = gru_forecast(result, train_vals, steps=len(test_vals))
            metrics = compute_metrics(test_vals, test_preds)
        except Exception:
            metrics = {
                "MSE": float("nan"),
                "MAE": float("nan"),
                "MAPE (%)": float("nan"),
                "RMSE": float("nan"),
            }

        rows.append({param_name: val, **metrics})

    if progress_callback:
        progress_callback(total, total, "Done")

    return pd.DataFrame(rows)


def normalization_study(
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    fixed_params: Dict[str, Any],
    progress_callback=None,
    epoch_callback=None,
) -> pd.DataFrame:
    """Train the model with Min-Max, Z-score, and no normalization; compare metrics."""
    configs = [("minmax", "Min-Max"), ("zscore", "Z-score"), ("none", "None")]
    rows = []

    for i, (mode, label) in enumerate(configs):
        if progress_callback:
            progress_callback(i, len(configs), label)

        kwargs = {k: v for k, v in fixed_params.items()}
        kwargs["normalization"] = mode
        kwargs["epoch_callback"] = epoch_callback

        try:
            result = train_gru(train_vals, **kwargs)
            test_preds = gru_forecast(result, train_vals, steps=len(test_vals))
            metrics = compute_metrics(test_vals, test_preds)
        except Exception:
            metrics = {
                "MSE": float("nan"),
                "MAE": float("nan"),
                "MAPE (%)": float("nan"),
                "RMSE": float("nan"),
            }

        rows.append({"Normalization": label, **metrics})

    if progress_callback:
        progress_callback(len(configs), len(configs), "Done")

    return pd.DataFrame(rows).set_index("Normalization")


def model_strategy_overview(
    strategy_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_types: List[str],
    fixed_params: Dict[str, Any],
    normalization: str = "minmax",
    progress_callback=None,
    epoch_callback=None,
) -> pd.DataFrame:
    """Run a full-factorial experiment over imputation strategy x model type."""
    rows: List[Dict[str, Any]] = []
    combos = [(s, m) for s in strategy_data.keys() for m in model_types]
    total = len(combos)

    for done, (strategy, model_type) in enumerate(combos):
        if progress_callback:
            progress_callback(done, total, f"{strategy} | {model_type}")

        train_vals, test_vals = strategy_data[strategy]
        started = perf_counter()

        kwargs = {k: v for k, v in fixed_params.items()}
        kwargs["normalization"] = normalization
        kwargs["model_type"] = model_type
        kwargs["epoch_callback"] = epoch_callback

        try:
            result = train_gru(train_vals, **kwargs)
            full_series = np.concatenate([train_vals, test_vals])
            test_preds = gru_one_step_predict(result, full_series, start_idx=len(train_vals))
            metrics = compute_metrics(test_vals, test_preds)
            rows.append(
                {
                    "strategy": strategy,
                    "model_type": model_type,
                    "normalization": normalization,
                    **metrics,
                    "train_loss_final": (
                        float(result["train_loss"][-1]) if result.get("train_loss") else float("nan")
                    ),
                    "val_loss_final": (
                        float(result["val_loss"][-1]) if result.get("val_loss") else float("nan")
                    ),
                    "runtime_sec": float(perf_counter() - started),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "strategy": strategy,
                    "model_type": model_type,
                    "normalization": normalization,
                    "MSE": float("nan"),
                    "MAE": float("nan"),
                    "MAPE (%)": float("nan"),
                    "RMSE": float("nan"),
                    "train_loss_final": float("nan"),
                    "val_loss_final": float("nan"),
                    "runtime_sec": float(perf_counter() - started),
                    "status": "error",
                    "error": str(exc),
                }
            )

    if progress_callback:
        progress_callback(total, total, "Done")

    return pd.DataFrame(rows)


def federated_training_study(
    user_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fixed_params: Dict[str, Any],
    aggregation_method: str = "mean",
    normalization: str = "minmax",
    num_rounds: int = 5,
    local_epochs_per_round: int | None = None,
    progress_callback=None,
    epoch_callback=None,
    round_callback=None,
) -> Dict[str, Any]:
    """Run multi-round federated training with shared global initialization."""
    from forecasting.models.architectures import _GRUNet, _MODEL_CLASSES, average_model_weights
    from forecasting.normalization.scalers import _apply_norm_params, _normalize

    user_ids = list(user_data_dict.keys())
    num_users = len(user_ids)
    rounds = max(1, int(num_rounds))
    local_epochs = int(local_epochs_per_round) if local_epochs_per_round is not None else int(
        fixed_params.get("epochs", 1)
    )
    local_epochs = max(1, local_epochs)
    total_steps = rounds * (num_users + 1)

    all_train_data = np.concatenate([user_data_dict[uid][0] for uid in user_ids])
    _, global_norm_params = _normalize(all_train_data, normalization)

    model_type = fixed_params.get("model_type", "gru").lower()
    _NetClass = _MODEL_CLASSES.get(model_type, _GRUNet)
    init_device = _get_device()
    init_model = _NetClass(
        hidden=fixed_params.get("hidden", 64),
        num_layers=fixed_params.get("num_layers", 2),
        activation=fixed_params.get("activation", "relu"),
        dropout=fixed_params.get("dropout", 0.0),
        device=init_device,
    )
    shared_initial_state = {
        k: v.detach().clone() for k, v in init_model.state_dict().items()
    }
    global_state = {k: v.detach().clone() for k, v in shared_initial_state.items()}

    local_models_last_round: List[Dict[str, Any]] = []
    final_user_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    user_round_rows: List[Dict[str, Any]] = []  # Track all users across all rounds
    epoch_metrics_rows: List[Dict[str, Any]] = []  # Track per-epoch local training losses
    aggregated_state = None
    global_model_result = None

    all_train_vals = np.concatenate([user_data_dict[uid][0] for uid in user_ids])
    all_test_vals = np.concatenate([user_data_dict[uid][1] for uid in user_ids])
    norm_all_train = _apply_norm_params(all_train_vals, global_norm_params)
    norm_all_test = _apply_norm_params(all_test_vals, global_norm_params)
    norm_combined = np.concatenate([norm_all_train, norm_all_test])

    step = 0
    for round_idx in range(rounds):
        local_states: List[Dict[str, Any]] = []
        local_models_this_round: List[Dict[str, Any]] = []
        user_rows_this_round: List[Dict[str, Any]] = []

        for user_id in user_ids:
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, f"Round {round_idx + 1}: training {user_id}")

            raw_train_vals, raw_test_vals = user_data_dict[user_id]
            train_vals = _apply_norm_params(raw_train_vals, global_norm_params)
            test_vals = _apply_norm_params(raw_test_vals, global_norm_params)
            started = perf_counter()

            kwargs = {k: v for k, v in fixed_params.items()}
            kwargs["epochs"] = local_epochs
            kwargs["normalization"] = "none"
            kwargs["initial_state"] = {k: v.detach().clone() for k, v in global_state.items()}

            def _epoch_cb_local(epoch_idx, train_l, val_l, _round=round_idx + 1, _user=user_id):
                train_last = float(train_l[-1]) if train_l else float("nan")
                val_last = float(val_l[-1]) if val_l else float("nan")
                epoch_metrics_rows.append(
                    {
                        "round": _round,
                        "model_id": _user,
                        "epoch": int(epoch_idx) + 1,
                        "train_loss": train_last,
                        "val_loss": val_last,
                        "status": "ok",
                    }
                )
                if epoch_callback is not None:
                    try:
                        epoch_callback(_round, _user, int(epoch_idx) + 1, train_l, val_l)
                    except TypeError:
                        epoch_callback(epoch_idx, train_l, val_l)

            kwargs["epoch_callback"] = _epoch_cb_local

            try:
                result = train_gru(train_vals, **kwargs)
                local_models_this_round.append(result)
                local_states.append(result["model"].state_dict())

                full_series = np.concatenate([train_vals, test_vals])
                test_preds = gru_one_step_predict(result, full_series, start_idx=len(train_vals))
                metrics = compute_metrics(test_vals, test_preds)

                user_rows_this_round.append(
                    {
                        "model_id": user_id,
                        "model_type": result.get("model_type", model_type),
                        "MSE": metrics["MSE"],
                        "MAE": metrics["MAE"],
                        "MAPE (%)": metrics["MAPE (%)"],
                        "RMSE": metrics["RMSE"],
                        "train_loss_final": float(result["train_loss"][-1]) if result.get("train_loss") else float("nan"),
                        "val_loss_final": float(result["val_loss"][-1]) if result.get("val_loss") else float("nan"),
                        "runtime_sec": float(perf_counter() - started),
                        "status": "ok",
                        "error": "",
                    }
                )
                # Also track per-round metrics for this user
                user_round_rows.append(
                    {
                        "round": round_idx + 1,
                        "model_id": user_id,
                        "MSE": metrics["MSE"],
                        "MAE": metrics["MAE"],
                        "MAPE (%)": metrics["MAPE (%)"],
                        "RMSE": metrics["RMSE"],
                        "runtime_sec": float(perf_counter() - started),
                        "status": "ok",
                    }
                )
                if round_callback is not None:
                    round_callback(round_idx + 1, user_id, user_round_rows[-1])
            except Exception as exc:
                user_rows_this_round.append(
                    {
                        "model_id": user_id,
                        "model_type": model_type,
                        "MSE": float("nan"),
                        "MAE": float("nan"),
                        "MAPE (%)": float("nan"),
                        "RMSE": float("nan"),
                        "train_loss_final": float("nan"),
                        "val_loss_final": float("nan"),
                        "runtime_sec": float(perf_counter() - started),
                        "status": "error",
                        "error": str(exc),
                    }
                )
                # Also track per-round error for this user
                user_round_rows.append(
                    {
                        "round": round_idx + 1,
                        "model_id": user_id,
                        "MSE": float("nan"),
                        "MAE": float("nan"),
                        "MAPE (%)": float("nan"),
                        "RMSE": float("nan"),
                        "runtime_sec": float(perf_counter() - started),
                        "status": "error",
                    }
                )
                if round_callback is not None:
                    round_callback(round_idx + 1, user_id, user_round_rows[-1])

        step += 1
        if progress_callback:
            progress_callback(step, total_steps, f"Round {round_idx + 1}: aggregating and evaluating")

        try:
            aggregated_state = average_model_weights(local_states, aggregation_method=aggregation_method)
            global_state = {k: v.detach().clone() for k, v in aggregated_state.items()}
        except Exception as exc:
            return {
                "results_df": pd.DataFrame(user_rows_this_round),
                "round_metrics_df": pd.DataFrame(round_rows),
                "user_round_metrics_df": pd.DataFrame(user_round_rows),
                "epoch_metrics_df": pd.DataFrame(epoch_metrics_rows),
                "local_models": local_models_this_round,
                "aggregated_model": None,
                "global_model_result": None,
                "user_ids": user_ids,
                "global_norm_params": global_norm_params,
                "error": f"Round {round_idx + 1} aggregation failed: {exc}",
            }

        started_global = perf_counter()
        try:
            global_device = local_models_this_round[0].get("device", "cpu") if local_models_this_round else "cpu"
            global_model = _NetClass(
                hidden=fixed_params.get("hidden", 64),
                num_layers=fixed_params.get("num_layers", 2),
                activation=fixed_params.get("activation", "relu"),
                dropout=fixed_params.get("dropout", 0.0),
                device=global_device,
            )
            global_model.load_state_dict(global_state)
            global_model.eval_mode()

            global_model_result = {
                "model": global_model,
                "norm_params": {"mode": "none"},
                "seq_len": fixed_params.get("seq_len", 60),
                "device": global_device,
                "model_type": model_type,
                "train_loss": [],
                "val_loss": [],
            }

            global_test_preds = gru_one_step_predict(
                global_model_result,
                norm_combined,
                start_idx=len(norm_all_train),
            )
            global_metrics = compute_metrics(norm_all_test, global_test_preds)

            round_rows.append(
                {
                    "round": round_idx + 1,
                    "MSE": global_metrics["MSE"],
                    "MAE": global_metrics["MAE"],
                    "MAPE (%)": global_metrics["MAPE (%)"],
                    "RMSE": global_metrics["RMSE"],
                    "runtime_sec": float(perf_counter() - started_global),
                    "status": "ok",
                }
            )
            # Also track aggregated model per-round metrics
            user_round_rows.append(
                {
                    "round": round_idx + 1,
                    "model_id": "global_aggregated",
                    "MSE": global_metrics["MSE"],
                    "MAE": global_metrics["MAE"],
                    "MAPE (%)": global_metrics["MAPE (%)"],
                    "RMSE": global_metrics["RMSE"],
                    "runtime_sec": float(perf_counter() - started_global),
                    "status": "ok",
                }
            )
            if round_callback is not None:
                round_callback(round_idx + 1, "global_aggregated", user_round_rows[-1])
            epoch_metrics_rows.append(
                {
                    "round": round_idx + 1,
                    "model_id": "global_aggregated",
                    "epoch": local_epochs,
                    "train_loss": float("nan"),
                    "val_loss": float("nan"),
                    "status": "ok",
                }
            )
        except Exception as exc:
            round_rows.append(
                {
                    "round": round_idx + 1,
                    "MSE": float("nan"),
                    "MAE": float("nan"),
                    "MAPE (%)": float("nan"),
                    "RMSE": float("nan"),
                    "runtime_sec": float(perf_counter() - started_global),
                    "status": f"error: {exc}",
                }
            )
            # Also track aggregated model error per-round
            user_round_rows.append(
                {
                    "round": round_idx + 1,
                    "model_id": "global_aggregated",
                    "MSE": float("nan"),
                    "MAE": float("nan"),
                    "MAPE (%)": float("nan"),
                    "RMSE": float("nan"),
                    "runtime_sec": float(perf_counter() - started_global),
                    "status": f"error: {exc}",
                }
            )
            if round_callback is not None:
                round_callback(round_idx + 1, "global_aggregated", user_round_rows[-1])
            epoch_metrics_rows.append(
                {
                    "round": round_idx + 1,
                    "model_id": "global_aggregated",
                    "epoch": local_epochs,
                    "train_loss": float("nan"),
                    "val_loss": float("nan"),
                    "status": f"error: {exc}",
                }
            )

        local_models_last_round = local_models_this_round
        final_user_rows = user_rows_this_round

    final_rows = list(final_user_rows)
    if round_rows:
        last_round = round_rows[-1]
        final_rows.append(
            {
                "model_id": "global_aggregated",
                "model_type": model_type,
                "MSE": last_round["MSE"],
                "MAE": last_round["MAE"],
                "MAPE (%)": last_round["MAPE (%)"],
                "RMSE": last_round["RMSE"],
                "train_loss_final": float("nan"),
                "val_loss_final": float("nan"),
                "runtime_sec": last_round.get("runtime_sec", float("nan")),
                "status": "ok" if str(last_round.get("status", "")).startswith("ok") else "error",
                "error": "" if str(last_round.get("status", "")).startswith("ok") else str(last_round.get("status", "")),
            }
        )

    if progress_callback:
        progress_callback(total_steps, total_steps, "Done")

    return {
        "results_df": pd.DataFrame(final_rows),
        "round_metrics_df": pd.DataFrame(round_rows),
        "user_round_metrics_df": pd.DataFrame(user_round_rows),  # All users + aggregated by round
        "epoch_metrics_df": pd.DataFrame(epoch_metrics_rows),
        "local_models": local_models_last_round,
        "aggregated_model": aggregated_state,
        "global_model_result": global_model_result,
        "user_ids": user_ids,
        "global_norm_params": global_norm_params,
    }
