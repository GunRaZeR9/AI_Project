METRIC_COLUMNS = ["MSE", "MAE", "MAPE (%)", "RMSE"]

SWEEP_VALUES = {
    "lr": [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "hidden": [4, 16, 32, 64, 128, 256],
    "seq_len": [4, 16, 32, 64, 128, 256],
}

STRATEGY_LABELS = {
    "ffill": "Forward fill",
    "fill_zero": "Fill with zero",
    "fill_mean": "Fill with column mean",
    "window_mean": "Rolling window mean",
    "predictive_imputer": "Predictive (RandomForest)",
}
