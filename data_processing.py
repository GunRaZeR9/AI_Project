import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Default path for the cached RF imputer
RF_IMPUTER_CACHE = Path(__file__).parent / "rf_imputer.joblib"


def get_rf_imputer_cache(dataset_key: str) -> Path:
    """Return a dataset-specific cache path for the RF imputer."""
    if dataset_key == "dsmts":
        return RF_IMPUTER_CACHE
    return Path(__file__).parent / f"rf_imputer_{dataset_key}.joblib"

# Available missing-value strategies used by the app UI
STRATEGIES = [
    'ffill',
    'fill_mean',
    'window_mean',
    'predictive_imputer',
]

def _drop_high_missing_columns(df, threshold=0.65):
    frac_missing = df.isna().mean()
    cols_to_drop = frac_missing[frac_missing > threshold].index.tolist()
    return df.drop(columns=cols_to_drop), cols_to_drop


def _rolling_window_fill(df, window=3):
    # Vectorized fill for numeric columns: faster than per-column Python loops.
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_filled.fillna(0)

    num_df = df_filled[numeric_cols]
    rolled = num_df.ffill().rolling(window=window, min_periods=1).mean()
    df_filled.loc[:, numeric_cols] = num_df.fillna(rolled)
    return df_filled


def train_and_save_imputer(df_clean, cache_path=None, random_state=42):
    """Fit an IterativeImputer on *df_clean*, persist it with joblib, and return it.

    Parameters
    ----------
    df_clean   : NaN-free DataFrame used to fit the imputer.
    cache_path : Path to save the fitted imputer. Defaults to RF_IMPUTER_CACHE.
    random_state : RNG seed for reproducibility.

    Returns
    -------
    Fitted IterativeImputer instance.
    """
    import joblib
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor

    if cache_path is None:
        cache_path = RF_IMPUTER_CACHE
    cache_path = Path(cache_path)

    imp = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, random_state=random_state),
        max_iter=10,
        random_state=random_state,
    )
    # Fit on complete rows only so the imputer learns from clean signal
    clean_vals = df_clean.select_dtypes(include=[np.number]).dropna().values
    imp.fit(clean_vals)
    joblib.dump(imp, cache_path)
    return imp


def load_imputer(cache_path=None):
    """Load a previously saved IterativeImputer from *cache_path*.

    Returns None if the file does not exist.
    """
    import joblib

    if cache_path is None:
        cache_path = RF_IMPUTER_CACHE
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    return joblib.load(cache_path)


def predictive_imputer(df, fitted_imputer=None, random_state=42):
    """Impute missing values using an IterativeImputer + RandomForest.

    If *fitted_imputer* is provided the imputer is used in transform-only mode
    (no retraining). Otherwise a fresh imputer is fit_transformed on *df*.
    Falls back to column-mean fill if sklearn is unavailable.
    """
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor

        if fitted_imputer is not None:
            arr = fitted_imputer.transform(df.values)
        else:
            imp = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=50, random_state=random_state),
                max_iter=10,
                random_state=random_state,
            )
            arr = imp.fit_transform(df.values)
        return pd.DataFrame(arr, columns=df.columns, index=df.index)
    except Exception:
        # fallback: simple mean fill for any remaining NA
        return df.fillna(df.mean())


def process_for_viz(df, strategy='fill_mean', window=3, drop_col_threshold=0.65,
                    fitted_imputer=None):
    """Return (X_before, X_after, dropped_cols).

    X_before: numeric columns of df before applying the selected strategy.
    X_after:  same columns after applying the strategy (unscaled).
    dropped_cols: columns removed when using drop_columns_threshold.

    Parameters
    ----------
    fitted_imputer : Optional pre-trained IterativeImputer. When provided and
                     strategy == 'predictive_imputer', the imputer is used in
                     transform-only mode (no retraining).
    """
    # Keep numeric columns only (Bitcoin OHLCV are already numeric)
    X = df.select_dtypes(include=[np.number]).copy()
    X_before = X.copy()

    dropped_cols = []
    if strategy == 'ffill':
        X_after = X_before.ffill()
    elif strategy == 'fill_mean':
        X_after = X_before.fillna(X_before.mean())
    elif strategy == 'window_mean':
        X_after = _rolling_window_fill(X_before, window=window)
    elif strategy == 'predictive_imputer':
        X_after = predictive_imputer(X_before, fitted_imputer=fitted_imputer)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_after = X_after.fillna(0).astype(float)
    return X_before, X_after, dropped_cols


# ------------------ DSMTS dataset helpers ------------------

def load_dsmts_data(local_csv: str | None = None) -> pd.DataFrame:
    """Load the Dynamical System Multivariate Time Series (DSMTS) dataset.

    Downloads via ``kagglehub`` on first call; subsequent calls are cached by
    Streamlit's ``@st.cache_data`` wrapper in ``app.py``.  If ``local_csv`` is
    provided the download is skipped entirely.

    The function:

    * Finds the CSV file in the downloaded dataset directory.
    * Parses the ``timestamp`` column into a ``DatetimeIndex``.
    * Retains all numeric signal columns (drops the timestamp after indexing).
    * Sorts the index in ascending chronological order.
    * Returns the **full** cleaned DataFrame; row-count slicing happens in
      ``app.py`` so the download is never re-triggered by the UI slider.

    Parameters
    ----------
    local_csv : str or None
        Path to a pre-downloaded CSV file.  When supplied the Kaggle download
        is skipped and the file is read directly.

    Returns
    -------
    pandas.DataFrame
        Cleaned signal DataFrame with a ``DatetimeIndex``.
    """
    if local_csv is not None:
        df = pd.read_csv(local_csv)
    else:
        try:
            import kagglehub
        except ImportError:
            raise ImportError(
                "kagglehub is required to download the DSMTS dataset.\n"
                'Install it with:  pip install "kagglehub[pandas-datasets]"'
            )
        dataset_path = kagglehub.dataset_download(
            "patrickfleith/dynamical-system-multivariate-time-series-forecast"
        )
        csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in downloaded dataset at: {dataset_path}")
        csv_file = sorted(csv_files)[0]
        df = pd.read_csv(csv_file)

    # Parse timestamp column → DatetimeIndex
    ts_col = next((c for c in df.columns if c.lower() == 'timestamp'), None)
    if ts_col is not None:
        df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True, errors='coerce')
        df = df.set_index(ts_col)
    df.index.name = 'timestamp'

    # Keep only numeric signal columns
    df = df.select_dtypes(include=[np.number]).copy()

    # Sort chronologically (ascending)
    df = df.sort_index()
    return df


def load_bitcoin_historical_data(local_csv: str | None = None) -> pd.DataFrame:
    """Load the Bitcoin historical OHLCV dataset from KaggleHub.

    Dataset: ``mczielinski/bitcoin-historical-data``
    """
    if local_csv is not None:
        df = pd.read_csv(local_csv)
    else:
        try:
            import kagglehub
        except ImportError:
            raise ImportError(
                "kagglehub is required to download the Bitcoin dataset.\n"
                'Install it with:  pip install "kagglehub[pandas-datasets]"'
            )

        dataset_path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
        csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in downloaded dataset at: {dataset_path}")

        preferred = [p for p in csv_files if os.path.basename(p).lower() == "btcusd_1-min_data.csv"]
        csv_file = preferred[0] if preferred else sorted(csv_files)[0]
        df = pd.read_csv(csv_file)

    ts_col = next((c for c in df.columns if c.lower() == "timestamp"), None)
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], unit="s", utc=True, errors="coerce")
        df[ts_col] = ts.dt.tz_localize(None)
        df = df.set_index(ts_col)
    df.index.name = "timestamp"

    # Keep numeric columns only to match the rest of the app's assumptions.
    df = df.select_dtypes(include=[np.number]).copy()
    df = df.dropna(how="all")
    df = df.sort_index()
    return df


DATASETS = {
    "dsmts": {
        "label": "DSMTS (patrickfleith)",
        "loader": load_dsmts_data,
        "default_target": "bed1",
        "title": "DSMTS Univariate Time Series Forecasting",
        "caption": (
            "Dataset: Dynamical System Multivariate Time Series (DSMTS) - "
            "17-variable simulated complex dynamical system at 1 Hz."
        ),
    },
    "bitcoin": {
        "label": "Bitcoin Historical Data (mczielinski)",
        "loader": load_bitcoin_historical_data,
        "default_target": "Close",
        "title": "Bitcoin Univariate Time Series Forecasting",
        "caption": (
            "Dataset: Bitcoin historical OHLCV at 1-minute resolution from KaggleHub."
        ),
    },
}


def inject_missing_values(
    df: pd.DataFrame,
    missing_rates: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inject MCAR (Missing Completely At Random) missing values into a DataFrame.

    For each column listed in ``missing_rates`` the function randomly selects
    ``round(rate * len(df))`` row positions and replaces them with ``NaN``.
    A fixed ``random_state`` ensures reproducible NaN positions for the same
    (rate, row-count) combination — useful for consistent UI previews.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean input data (no existing NaNs expected, but harmless if present).
    missing_rates : dict, optional
        Mapping of ``column_name → fraction`` (0.0 – 1.0).  Columns not
        present in ``df`` are silently skipped.  When ``None`` no values are
        injected and a copy of ``df`` is returned unchanged.
    random_state : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with NaN values injected at the requested rates.
    """
    if not missing_rates:
        return df.copy()

    rng = np.random.default_rng(random_state)
    df_out = df.copy()
    n = len(df_out)

    for col, rate in missing_rates.items():
        if col not in df_out.columns or rate <= 0:
            continue
        n_missing = max(1, int(round(rate * n)))
        positions = rng.choice(n, size=n_missing, replace=False)
        df_out.iloc[positions, df_out.columns.get_loc(col)] = np.nan

    return df_out


def partition_data_for_users(
    series: pd.Series,
    num_users: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Partition a time series into num_users chronological blocks and split each into train/test.

    Parameters
    ----------
    series : pandas.Series
        The univariate time series to partition (assumed already sorted chronologically).
    num_users : int
        Number of user blocks to create. Defaults to 5.
    test_size : float
        Fraction for test set within each user block.
    seed : int
        Random seed (for reproducibility, though this function is deterministic).

    Returns
    -------
    dict
        Mapping ``{user_id: (train_vals, test_vals)}`` where user_id is "user_0", "user_1", etc.
        Each (train_vals, test_vals) pair contains numpy arrays.
    """
    from forecasting.utils.common import train_test_split_series

    vals = series.astype(float).dropna().values
    total_len = len(vals)
    chunk_size = total_len // num_users

    user_data = {}
    for u in range(num_users):
        start_idx = u * chunk_size
        # Last user gets any remaining rows
        end_idx = start_idx + chunk_size if u < num_users - 1 else total_len

        user_chunk = vals[start_idx:end_idx]

        # Chronological train/test split per user
        user_series = pd.Series(user_chunk)
        train_s, test_s = train_test_split_series(user_series, test_size=test_size)
        train_vals = train_s.dropna().values
        test_vals = test_s.dropna().values

        user_data[f"user_{u}"] = (train_vals, test_vals)

    return user_data
