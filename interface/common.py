import numpy as np
from matplotlib.ticker import FuncFormatter


def available_metrics(df, candidates=None):
    cols = candidates or ["MSE", "MAE", "MAPE (%)", "RMSE"]
    return [c for c in cols if c in df.columns]


DECIMAL_FMT = FuncFormatter(lambda x, _: f"{x:.6f}".rstrip("0").rstrip("."))


def format_metrics_table(styler, include_runtime=False):
    fmt = {
        "MSE": "{:,.6f}",
        "MAE": "{:,.6f}",
        "MAPE (%)": "{:,.4f}",
        "RMSE": "{:,.6f}",
        "train_loss_final": "{:,.6f}",
        "val_loss_final": "{:,.6f}",
    }
    if include_runtime:
        fmt["runtime_sec"] = "{:,.2f}"
    return styler.format(fmt, na_rep="N/A")


def finite_best_index(values):
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return int(np.nanargmin(arr))
