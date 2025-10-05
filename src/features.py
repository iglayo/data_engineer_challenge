from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def resample_to_hourly(df: pd.DataFrame, datetime_col: str = "datetime", value_col: str = "target",
                       how: str = "mean") -> pd.DataFrame:
    """
    Aggregate high-frequency data to hourly resolution.
    how: aggregation function to apply ('mean','median','sum','max','min').
    Returns DataFrame with one row per hour (tz-aware UTC).
    """
    df = df.copy()
    # Ensure datetime and UTC tz
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df = df.set_index(datetime_col)

    # Choose aggregation
    if how not in {"mean", "median", "sum", "max", "min"}:
        raise ValueError("how must be one of mean/median/sum/max/min")

    agg = getattr(df[value_col].resample("h"), how)()

    # After resampling, agg is a Series indexed by hourly timestamps (tz-aware UTC)
    agg = agg.reset_index().rename(columns={datetime_col: "datetime", value_col: value_col})
    return agg

def partial_hour_features(raw_df: pd.DataFrame, datetime_col: str = "datetime", value_col: str = "target") -> Dict[str, float]:
    """
    Compute features from the partial hour between the last complete hour and last observed timestamp.
    Returns dictionary e.g. {'partial_count': n, 'partial_mean': x, 'partial_std': y, 'last_obs': z}
    If no partial data (last sample at exact hour), returns zeros/NaN accordingly.
    """
    df = raw_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    last_ts = df[datetime_col].max()
    last_complete = last_ts.floor("H")
    partial = df[(df[datetime_col] > last_complete) & (df[datetime_col] <= last_ts)]
    if partial.empty:
        return {"partial_count": 0, "partial_mean": np.nan, "partial_std": np.nan, "last_obs": np.nan}
    return {
        "partial_count": int(len(partial)),
        "partial_mean": float(partial[value_col].mean()),
        "partial_std": float(partial[value_col].std(ddof=0)) if len(partial)>1 else 0.0,
        "last_obs": float(partial[value_col].iloc[-1])
    }

def ensure_hourly_index(df: pd.DataFrame, datetime_col: str = "datetime", value_col: str = "target",
                        method_fill: Optional[str] = "ffill") -> pd.DataFrame:
    """
    Ensure hourly index in UTC and reindex to continuous hourly periods.
    We need this because ESIOS API may return missing hours, and we have more than one sample per hour.
    method_fill: 'ffill', 'bfill', None or 'interpolate'
    """
    if datetime_col not in df.columns:
        raise ValueError(f"{datetime_col} not in dataframe")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df = df.sort_values(datetime_col).set_index(datetime_col)

    # Create full hourly index between min and max
    full_idx = pd.date_range(start=df.index.min().floor("h"), end=df.index.max().ceil("h"), freq="h", tz="UTC")
    df = df.reindex(full_idx)

    # Preserve the value column name
    if value_col in df.columns:
        # Fill strategy
        if method_fill == "ffill":
            df[value_col] = df[value_col].ffill()
        elif method_fill == "bfill":
            df[value_col] = df[value_col].bfill()
        elif method_fill == "interpolate":
            df[value_col] = df[value_col].interpolate(limit_direction="both")
        elif method_fill is None:
            pass
        else:
            raise ValueError("method_fill must be one of {'ffill','bfill','interpolate',None}")

    df = df.rename_axis("datetime").reset_index()
    return df


def create_lag_features(df: pd.DataFrame, value_col: str = "target", lags: List[int] = [1,24,168]) -> pd.DataFrame:
    """
    Add lag features in hours. Example lags: [1,24,168] -> t-1, t-24, t-168 (week)
    """
    df = df.copy()
    df = df.set_index("datetime")
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df[value_col].shift(lag)
    df = df.reset_index()
    return df

def create_rolling_features(df: pd.DataFrame, value_col: str = "target", windows: List[int] = [3,24,168]) -> pd.DataFrame:
    """
    Add rolling mean, std, median for given window sizes (in hours).
    """
    df = df.copy().set_index("datetime")
    for w in windows:
        df[f"{value_col}_roll_mean_{w}"] = df[value_col].rolling(window=w, min_periods=1).mean()
        df[f"{value_col}_roll_std_{w}"] = df[value_col].rolling(window=w, min_periods=1).std().fillna(0)
        df[f"{value_col}_roll_med_{w}"] = df[value_col].rolling(window=w, min_periods=1).median()
    df = df.reset_index()
    return df

def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Add cyclic and categorical time features:
    - hour, dayofweek, month, dayofyear, is_weekend
    - cyclical encodings for hour (sin/cos)
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df["hour"] = df[datetime_col].dt.hour
    df["dayofweek"] = df[datetime_col].dt.dayofweek
    df["month"] = df[datetime_col].dt.month
    df["dayofyear"] = df[datetime_col].dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

    # cyclical encoding for hour (24h)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def train_val_split_time(df: pd.DataFrame, datetime_col: str = "datetime", val_hours: int = 24*7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: last `val_hours` rows are validation.
    Returns (train_df, val_df)
    """
    df = df.copy().sort_values(datetime_col)
    # Assume regular hourly index; if not, split by timestamp
    cutoff = df[datetime_col].max() - pd.Timedelta(hours=val_hours)
    train = df[df[datetime_col] <= cutoff].reset_index(drop=True)
    val = df[df[datetime_col] > cutoff].reset_index(drop=True)
    return train, val

def build_features_pipeline(raw_df: pd.DataFrame,
                            value_col: str = "target",
                            lags: Optional[List[int]] = None,
                            rolling_windows: Optional[List[int]] = None,
                            fill_method: str = "ffill",
                            min_train_rows: int = 48) -> pd.DataFrame:
    """
    Build features robustly:
    - optionally keep only exact-hour rows
    - ensure hourly index
    - dynamically choose valid lags so that after dropping first max_lag rows
      we still keep at least `min_train_rows` rows for training
    """
    if lags is None:
        lags = [1, 24, 168]
    if rolling_windows is None:
        rolling_windows = [3, 24, 168]

    df_in = raw_df.copy()

    df = ensure_hourly_index(df_in, method_fill=fill_method)
    n_rows = len(df)
    if n_rows <= 0:
        raise ValueError("No rows after ensuring hourly index")

    # Choose valid lags but ensure at least `min_train_rows` remain after dropping max_lag
    requested_lags = sorted(set(lags))
    valid_lags = [lag for lag in requested_lags if lag < n_rows]  # initial filter

    # Trim largest lags until (n_rows - max_lag) >= min_train_rows or no more lags
    while valid_lags:
        max_lag = max(valid_lags)
        if (n_rows - max_lag) >= min_train_rows:
            break
        # remove the largest lag and retry
        removed = valid_lags.pop(-1)
        logger.warning("Trimmed lag %d because not enough rows (%d) to keep min_train_rows=%d", removed, n_rows, min_train_rows)

    # If no valid lags left, pick a safe default
    if not valid_lags:
        fallback = [1, 24]
        valid_lags = [lag for lag in fallback if lag < n_rows]
        logger.warning("No valid requested lags left; falling back to %s (valid: %s)", fallback, valid_lags)

    logger.info("Using valid_lags=%s with n_rows=%d", valid_lags, n_rows)

    df = create_lag_features(df, value_col=value_col, lags=valid_lags if valid_lags else [])
    df = create_rolling_features(df, value_col=value_col, windows=rolling_windows)
    df = add_time_features(df)

    # Deterministic trimming: drop first max_lag rows (if any)
    if valid_lags:
        max_lag = max(valid_lags)
        if n_rows > max_lag:
            df = df.iloc[max_lag:].reset_index(drop=True)
            logger.info("Dropped first %d rows (max_lag) keeping %d rows for training", max_lag, len(df))
    else:
        df = df.reset_index(drop=True)
    return df