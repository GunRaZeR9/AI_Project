from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from forecasting.inference.predictor import gru_one_step_predict
from forecasting.metrics.evaluator import compute_metrics
from forecasting.training.trainer import train_gru
from forecasting.utils.common import _get_device, split_train_temporal_halves


def _has_finite_core_metrics(metrics: Dict[str, Any]) -> bool:
    """Core metrics must be finite for a valid training/eval step."""
    core = [metrics.get("MSE"), metrics.get("MAE"), metrics.get("RMSE")]
    try:
        vals = np.asarray(core, dtype=float)
    except Exception:
        return False
    return bool(np.all(np.isfinite(vals)))


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
            test_vals = raw_test_vals  # Do NOT pre-normalize; gru_one_step_predict handles normalization
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

                # Use raw data for prediction; gru_one_step_predict normalizes internally
                full_series = np.concatenate([raw_train_vals, raw_test_vals])
                test_preds = gru_one_step_predict(result, full_series, start_idx=len(raw_train_vals))
                if not np.all(np.isfinite(np.asarray(test_preds, dtype=float))):
                    raise ValueError("Non-finite local predictions produced.")
                metrics = compute_metrics(test_vals, test_preds)
                if not _has_finite_core_metrics(metrics):
                    raise ValueError(f"Non-finite local metrics: {metrics}")

                local_models_this_round.append(result)
                local_states.append(result["model"].state_dict())

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

        if not local_states:
            failed_rows = [r for r in user_rows_this_round if str(r.get("status", "")) == "error"]
            failed_models = [str(r.get("model_id", "unknown")) for r in failed_rows]
            first_error = next((str(r.get("error", "")) for r in failed_rows if r.get("error")), "")
            fail_msg = (
                f"Round {round_idx + 1} produced no local model states. "
                f"Failed users: {', '.join(failed_models) if failed_models else 'unknown'}."
            )
            if first_error:
                fail_msg = f"{fail_msg} First error: {first_error}"
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
                "error": fail_msg,
            }

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
            if not np.all(np.isfinite(np.asarray(global_test_preds, dtype=float))):
                raise ValueError("Non-finite aggregated predictions produced.")
            global_metrics = compute_metrics(norm_all_test, global_test_preds)
            if not _has_finite_core_metrics(global_metrics):
                raise ValueError(f"Non-finite aggregated metrics: {global_metrics}")

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


def _normalize_seed_list(seeds: List[int] | None, num_seeds: int) -> List[int]:
    if seeds:
        out = [int(s) for s in seeds]
    else:
        out = list(range(max(1, int(num_seeds))))
    return sorted(set(out))


def _member_train_eval(
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    model_type: str,
    seed: int,
    normalization: str,
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    started = perf_counter()
    kwargs = {k: v for k, v in fixed_params.items()}
    kwargs["normalization"] = normalization
    kwargs["model_type"] = model_type
    kwargs["random_seed"] = int(seed)
    result = train_gru(train_vals, **kwargs)
    full_series = np.concatenate([train_vals, test_vals])
    preds = gru_one_step_predict(result, full_series, start_idx=len(train_vals))
    metrics = compute_metrics(test_vals, preds)
    return {
        "result": result,
        "preds": preds,
        "metrics": metrics,
        "runtime_sec": float(perf_counter() - started),
    }


def _fit_linear_fusion(features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Invalid feature/target shapes for linear fusion.")
    x_aug = np.column_stack([x, np.ones(len(x), dtype=float)])
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    return coef[:-1], float(coef[-1])


def _predict_linear_fusion(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    return x @ weights + float(bias)


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape or yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def parallel_independent_ensemble_study(
    strategy_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_types: List[str],
    fixed_params: Dict[str, Any],
    normalization: str = "minmax",
    seeds: List[int] | None = None,
    num_seeds: int = 5,
    progress_callback=None,
    epoch_callback=None,
    member_callback=None,
) -> pd.DataFrame:
    """Independent member models + trained final fusion decision model."""
    rows: List[Dict[str, Any]] = []
    seed_list = _normalize_seed_list(seeds, num_seeds)
    if not model_types:
        return pd.DataFrame(rows)

    total_members = len(strategy_data) * len(model_types) * len(seed_list)
    done_members = 0

    for strategy, data_pair in strategy_data.items():
        train_vals, test_vals = data_pair
        half_1, half_2 = split_train_temporal_halves(train_vals)
        member_outputs: List[Dict[str, Any]] = []

        for model_type in model_types:
            for seed in seed_list:
                started = perf_counter()
                kwargs = {k: v for k, v in fixed_params.items()}
                kwargs["normalization"] = normalization
                kwargs["model_type"] = model_type
                kwargs["random_seed"] = int(seed)

                def _epoch_cb_local(epoch_idx, train_l, val_l, _strategy=strategy, _model=model_type, _seed=seed):
                    if epoch_callback is not None:
                        epoch_callback(
                            {
                                "experiment_mode": "parallel_independent",
                                "strategy": _strategy,
                                "role": "member",
                                "member_id": f"{_model}_seed{_seed}",
                                "model_type": _model,
                                "seed": int(_seed),
                                "epoch": int(epoch_idx) + 1,
                                "train_loss": float(train_l[-1]) if train_l else float("nan"),
                                "val_loss": float(val_l[-1]) if val_l else float("nan"),
                            }
                        )

                kwargs["epoch_callback"] = _epoch_cb_local

                try:
                    result = train_gru(half_1, **kwargs)

                    val_series = np.concatenate([half_1, half_2])
                    val_preds = gru_one_step_predict(result, val_series, start_idx=len(half_1))

                    test_series = np.concatenate([train_vals, test_vals])
                    test_preds = gru_one_step_predict(result, test_series, start_idx=len(train_vals))

                    metrics = compute_metrics(test_vals, test_preds)
                    member_outputs.append(
                        {
                            "result": result,
                            "model_type": model_type,
                            "seed": int(seed),
                            "val_preds": val_preds,
                            "test_preds": test_preds,
                            "runtime_sec": float(perf_counter() - started),
                        }
                    )
                    rows.append(
                        {
                            "experiment_mode": "parallel_independent",
                            "strategy": strategy,
                            "scope": "base_members_half1",
                            "role": "member",
                            "member_id": f"{model_type}_seed{seed}",
                            "model_type": model_type,
                            "seed": seed,
                            "MSE": metrics["MSE"],
                            "MAE": metrics["MAE"],
                            "MAPE (%)": metrics["MAPE (%)"],
                            "RMSE": metrics["RMSE"],
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
                    if member_callback is not None:
                        member_callback(rows[-1])
                except Exception as exc:
                    rows.append(
                        {
                            "experiment_mode": "parallel_independent",
                            "strategy": strategy,
                            "scope": "base_members_half1",
                            "role": "member",
                            "member_id": f"{model_type}_seed{seed}",
                            "model_type": model_type,
                            "seed": seed,
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
                    if member_callback is not None:
                        member_callback(rows[-1])

                done_members += 1
                if progress_callback:
                    progress_callback(done_members, total_members, f"{strategy} members")

        if not member_outputs:
            continue

        try:
            meta_x = np.column_stack([m["val_preds"] for m in member_outputs])
            meta_y = half_2.astype(float)
            test_x = np.column_stack([m["test_preds"] for m in member_outputs])
            weights, bias = _fit_linear_fusion(meta_x, meta_y)
            fused_test = _predict_linear_fusion(test_x, weights, bias)
            fused_metrics = compute_metrics(test_vals, fused_test)
            rows.append(
                {
                    "experiment_mode": "parallel_independent",
                    "strategy": strategy,
                    "scope": "fusion_on_half2",
                    "role": "ensemble",
                    "member_id": "linear_fusion",
                    "model_type": "fusion_linear",
                    "seed": float("nan"),
                    "MSE": fused_metrics["MSE"],
                    "MAE": fused_metrics["MAE"],
                    "MAPE (%)": fused_metrics["MAPE (%)"],
                    "RMSE": fused_metrics["RMSE"],
                    "train_loss_final": float("nan"),
                    "val_loss_final": float("nan"),
                    "runtime_sec": float("nan"),
                    "status": "ok",
                    "error": "",
                }
            )
            if member_callback is not None:
                member_callback(rows[-1])
        except Exception as exc:
            rows.append(
                {
                    "experiment_mode": "parallel_independent",
                    "strategy": strategy,
                    "scope": "fusion_on_half2",
                    "role": "ensemble",
                    "member_id": "linear_fusion",
                    "model_type": "fusion_linear",
                    "seed": float("nan"),
                    "MSE": float("nan"),
                    "MAE": float("nan"),
                    "MAPE (%)": float("nan"),
                    "RMSE": float("nan"),
                    "train_loss_final": float("nan"),
                    "val_loss_final": float("nan"),
                    "runtime_sec": float("nan"),
                    "status": "error",
                    "error": str(exc),
                }
            )
            if member_callback is not None:
                member_callback(rows[-1])

    if progress_callback:
        progress_callback(total_members, total_members, "Done")
    return pd.DataFrame(rows)


def sequential_residual_ensemble_study(
    strategy_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_types: List[str],
    fixed_params: Dict[str, Any],
    normalization: str = "minmax",
    seeds: List[int] | None = None,
    num_seeds: int = 5,
    progress_callback=None,
    epoch_callback=None,
    member_callback=None,
) -> pd.DataFrame:
    """Strict chained sequential pipeline: stage input = original input + previous stage output."""
    rows: List[Dict[str, Any]] = []
    if not model_types:
        return pd.DataFrame(rows)

    seed_list = _normalize_seed_list(seeds, num_seeds)
    total_steps = len(strategy_data) * len(model_types)
    done_steps = 0

    for strategy, data_pair in strategy_data.items():
        train_vals, test_vals = data_pair
        original_train = train_vals.astype(float).copy()
        original_test = test_vals.astype(float).copy()
        prev_train_pred = np.zeros_like(original_train, dtype=float)
        prev_test_pred = np.zeros_like(original_test, dtype=float)
        final_stage_pred = np.zeros_like(original_test, dtype=float)

        for stage_idx, model_type in enumerate(model_types, start=1):
            seed = int(seed_list[(stage_idx - 1) % len(seed_list)])
            started = perf_counter()
            kwargs = {k: v for k, v in fixed_params.items()}
            kwargs["normalization"] = normalization
            kwargs["model_type"] = model_type
            kwargs["random_seed"] = seed

            train_input = original_train + prev_train_pred
            test_input = original_test + prev_test_pred

            def _epoch_cb_local(epoch_idx, train_l, val_l, _strategy=strategy, _model=model_type, _seed=seed, _stage=stage_idx):
                if epoch_callback is not None:
                    epoch_callback(
                        {
                            "experiment_mode": "sequential_residual",
                            "strategy": _strategy,
                            "role": "stage",
                            "member_id": f"stage_{_stage}",
                            "model_type": _model,
                            "seed": int(_seed),
                            "epoch": int(epoch_idx) + 1,
                            "train_loss": float(train_l[-1]) if train_l else float("nan"),
                            "val_loss": float(val_l[-1]) if val_l else float("nan"),
                        }
                    )

            kwargs["epoch_callback"] = _epoch_cb_local

            try:
                result = train_gru(train_input, **kwargs)

                full_chain = np.concatenate([train_input, test_input])
                stage_chain_test_pred = gru_one_step_predict(result, full_chain, start_idx=len(train_input))
                stage_chain_train_pred = gru_one_step_predict(result, train_input, start_idx=0)

                if stage_chain_test_pred.shape != prev_test_pred.shape:
                    raise ValueError(
                        "Sequential chain output shape mismatch on test split: "
                        f"got {stage_chain_test_pred.shape}, expected {prev_test_pred.shape}"
                    )
                if stage_chain_train_pred.shape != prev_train_pred.shape:
                    raise ValueError(
                        "Sequential chain output shape mismatch on train split: "
                        f"got {stage_chain_train_pred.shape}, expected {prev_train_pred.shape}"
                    )

                # Convert chain predictions back to target space by removing previous-stage contribution.
                stage_test_pred = stage_chain_test_pred - prev_test_pred
                stage_train_pred = stage_chain_train_pred - prev_train_pred

                final_stage_pred = stage_test_pred
                prev_train_pred = stage_train_pred
                prev_test_pred = stage_test_pred

                stage_metrics = compute_metrics(original_test, stage_test_pred)
                rows.append(
                    {
                        "experiment_mode": "sequential_residual",
                        "strategy": strategy,
                        "scope": "strict_chain_full_train",
                        "role": "stage",
                        "member_id": f"stage_{stage_idx}",
                        "model_type": model_type,
                        "seed": seed,
                        "MSE": stage_metrics["MSE"],
                        "MAE": stage_metrics["MAE"],
                        "MAPE (%)": stage_metrics["MAPE (%)"],
                        "RMSE": stage_metrics["RMSE"],
                        "dependency": "original_plus_previous_output",
                        "prev_output_rmse": _safe_rmse(original_test, prev_test_pred),
                        "prev_output_preview": np.array2string(prev_test_pred[:3], precision=4, separator=", "),
                        "stage_output_preview": np.array2string(stage_test_pred[:3], precision=4, separator=", "),
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
                if member_callback is not None:
                    member_callback(rows[-1])
            except Exception as exc:
                rows.append(
                    {
                        "experiment_mode": "sequential_residual",
                        "strategy": strategy,
                        "scope": "strict_chain_full_train",
                        "role": "stage",
                        "member_id": f"stage_{stage_idx}",
                        "model_type": model_type,
                        "seed": seed,
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
                if member_callback is not None:
                    member_callback(rows[-1])

            done_steps += 1
            if progress_callback:
                progress_callback(done_steps, total_steps, f"{strategy} stage {stage_idx}")

        final_metrics = compute_metrics(original_test, final_stage_pred)
        rows.append(
            {
                "experiment_mode": "sequential_residual",
                "strategy": strategy,
                "scope": "strict_chain_full_train",
                "role": "final_output",
                "member_id": "chain_terminal",
                "model_type": model_types[-1],
                "seed": float("nan"),
                "MSE": final_metrics["MSE"],
                "MAE": final_metrics["MAE"],
                "MAPE (%)": final_metrics["MAPE (%)"],
                "RMSE": final_metrics["RMSE"],
                "dependency": "strict_chain_no_fusion",
                "train_loss_final": float("nan"),
                "val_loss_final": float("nan"),
                "runtime_sec": float("nan"),
                "status": "ok",
                "error": "",
            }
        )
        if member_callback is not None:
            member_callback(rows[-1])

    if progress_callback:
        progress_callback(total_steps, total_steps, "Done")
    return pd.DataFrame(rows)
