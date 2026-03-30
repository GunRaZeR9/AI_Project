METRIC_COLUMNS = ["MSE", "MAE", "MAPE (%)", "RMSE"]

STRATEGY_LABELS = {
    "ffill": "Forward fill",
    "fill_zero": "Fill with zero",
    "fill_mean": "Fill with column mean",
    "window_mean": "Rolling window mean",
    "predictive_imputer": "Predictive (RandomForest)",
}
