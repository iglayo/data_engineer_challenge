import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Optional
from .config import BASE_URL, ESIOS_API_KEY, RAW_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_esios(indicator_id: int, start: datetime, end: datetime, token: Optional[str] = None) -> pd.DataFrame:
    token = token or ESIOS_API_KEY
    if not token:
        raise RuntimeError("No ESIOS token set. Export ESIOS_API_KEY env var before running.")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": token
    }
    params = {"start_date": start.isoformat(), "end_date": end.isoformat()}
    url = f"{BASE_URL}{indicator_id}"
    logger.info("Fetching indicator %s from %s to %s", indicator_id, params["start_date"], params["end_date"])
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    values = payload.get("indicator", {}).get("values", [])
    if not values:
        logger.warning("Empty payload from ESIOS for indicator %s", indicator_id)
        return pd.DataFrame(columns=["datetime", "target"])
    df = pd.DataFrame(values)
    # Estructura esperada: 'datetime' y 'value'
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    if "value" in df.columns:
        df.rename(columns={"value": "target"}, inplace=True)
    # Keep only relevant columns
    keep = [c for c in ("datetime", "target") if c in df.columns]
    return df[keep]

def save_raw_parquet(df: pd.DataFrame, indicator_id: int, start: datetime, end: datetime):
    fname = RAW_DIR / f"{indicator_id}_{start.date().isoformat()}_{end.date().isoformat()}.parquet"
    df.to_parquet(fname, index=False)
    logger.info("Saved raw parquet to %s", fname)
    return fname

# Fallback: cargar csv local si falla la API
def load_local_csv(path):
    logger.info("Loading local CSV fallback: %s", path)
    return pd.read_csv(path, parse_dates=["datetime"])

# Normalización de datetime (timezone UTC)
def normalize_datetime(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """
    Normalize a datetime column to UTC.
    - If column is not datetime, convert to datetime.
    - If datetime values are tz-aware, convert to UTC.
    - If naive (no tz), assume local Spain timezone (Europe/Madrid) then convert to UTC.
    Returns a sorted dataframe with reset index.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in dataframe")

    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop rows where datetime conversion failed
    if df[col].isna().any():
        logging.warning("Some datetime values could not be parsed and will be dropped")
        df = df.dropna(subset=[col])

    # If tz-aware, convert directly to UTC
    if df[col].dt.tz is not None:
        df[col] = df[col].dt.tz_convert("UTC")
    else:
        # Assume the naive timestamps are in Spain local time and localize then convert to UTC
        df[col] = df[col].dt.tz_localize("Europe/Madrid").dt.tz_convert("UTC")

    df = df.sort_values(col).reset_index(drop=True)
    return df
