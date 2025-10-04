from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

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
    full_idx = pd.date_range(start=df.index.min().floor("H"), end=df.index.max().ceil("H"), freq="H", tz="UTC")
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