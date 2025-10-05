from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
                            fill_method: str = "ffill") -> pd.DataFrame:
    """
    Orchestrates the feature creation:
    - enforce hourly index & fill
    - create lags (only those that make sense)
    - create rolling features
    - add time features
    - drop initial rows deterministically based on the max valid lag (safer than dropna)
    """
    if lags is None:
        lags = [1, 24, 168]
    if rolling_windows is None:
        rolling_windows = [3, 24, 168]

    df = ensure_hourly_index(raw_df, method_fill=fill_method)
    n_rows = len(df)

    # Keep only lags that are strictly smaller than available rows
    valid_lags = [lag for lag in lags if lag < n_rows]
    if len(valid_lags) < len(lags):
        logger.warning("Requested lags %s trimmed to valid lags %s due to limited series length (%d rows)",
                       lags, valid_lags, n_rows)

    # Create lag features (empty list -> no lag columns created)
    df = create_lag_features(df, value_col=value_col, lags=valid_lags if valid_lags else [])
    df = create_rolling_features(df, value_col=value_col, windows=rolling_windows)
    df = add_time_features(df)

    # Deterministic trimming: drop the first max_lag rows (they cannot have full lag history)
    if valid_lags:
        max_lag = max(valid_lags)
        if n_rows > max_lag:
            logger.info("Dropping first %d rows to remove incomplete lag features", max_lag)
            df = df.iloc[max_lag:].reset_index(drop=True)
        else:
            # shouldn't happen because valid_lags filtered, but keep safe fallback
            logger.info("n_rows <= max_lag, keeping all rows (no drop applied)")
            df = df.reset_index(drop=True)
    else:
        logger.info("No valid lag columns available (series too short). Keeping all rows after feature creation.")
        df = df.reset_index(drop=True)

    return df