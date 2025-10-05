# src/model.py
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def train_model(train_df: pd.DataFrame, feature_cols: List[str], target_col: str = "target") -> RandomForestRegressor:
    """Train a RandomForest on train_df using feature_cols."""
    X = train_df[feature_cols].astype(float).fillna(0.0)
    y = train_df[target_col].astype(float)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    logger.info("Training model on %d rows, %d features", X.shape[0], X.shape[1])
    model.fit(X, y)
    return model

def save_model(model, name: str = "rf_model.joblib") -> Path:
    path = MODEL_DIR / name
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
    return path

def load_model(path: str):
    return joblib.load(path)

def evaluate(model, df: pd.DataFrame, feature_cols: List[str], target_col: str = "target"):
    """Evaluate model on df and return MAE."""
    if df.empty:
        logger.warning("Empty evaluation dataframe; skipping evaluation")
        return None
    X = df[feature_cols].astype(float).fillna(0.0)
    y = df[target_col].astype(float)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    logger.info("Evaluation MAE: %f on %d samples", mae, len(y))
    return mae

def _parse_lag_cols(feature_cols: List[str], prefix: str = "target_lag_"):
    """Return sorted list of lag column names (ascending lag number)."""
    lag_cols = [c for c in feature_cols if c.startswith(prefix)]
    def lag_num(c):
        try:
            return int(c.split("_")[-1])
        except:
            return 999999
    return sorted(lag_cols, key=lag_num)

def make_recursive_forecast(model, last_features_row: pd.Series, feature_cols: List[str],
                            horizon: int = 6, lag_prefix: str = "target_lag_") -> pd.DataFrame:
    """
    Recursive forecast for `horizon` steps.
    - last_features_row: pandas Series indexed by column name (should include feature_cols)
    - feature_cols: order of features the model expects
    Returns DataFrame with columns: datetime (placeholder), target (NaN), prediction, step
    Note: this function does NOT set final datetimes; caller will label them relative to T_observed.
    """
    cur = last_features_row.copy()
    # ensure index contains all feature_cols
    for c in feature_cols:
        if c not in cur.index:
            cur[c] = np.nan

    lag_cols = _parse_lag_cols(feature_cols, prefix=lag_prefix)
    # determine numeric order for lag columns (ascending: lag_1, lag_2, ...)
    def lag_num(col): 
        try:
            return int(col.split("_")[-1])
        except:
            return 999999
    lag_nums = [lag_num(c) for c in lag_cols]

    preds = []
    # For horizon steps
    for step in range(1, horizon + 1):
        X_df = pd.DataFrame([cur[feature_cols].astype(float).fillna(0.0).to_dict()], columns=feature_cols)
        y_pred = float(model.predict(X_df)[0])
        preds.append(y_pred)

        # Update lag columns: shift up (largest <- prev largest-1, ...), then set lag_1 = y_pred
        if lag_cols:
            # Create mapping new values
            # Example: for lag cols [lag_1, lag_24, lag_168] we want:
            # lag_168 = lag_167 (but 167 not present) -> we shift via numeric transformation:
            # simpler approach: create dict current_lags where key=lag_num -> value
            cur_lags = {lag_num(c): float(cur.get(c, np.nan)) for c in lag_cols}
            # shift: new value for k = prev value for k-1 (for k>1)
            new_lags = {}
            max_existing = max(cur_lags.keys()) if cur_lags else 0
            for k in sorted(cur_lags.keys(), reverse=True):
                if k == 1:
                    new_lags[1] = y_pred
                else:
                    new_lags[k] = cur_lags.get(k-1, np.nan)
            # write back to cur
            for c in lag_cols:
                k = lag_num(c)
                cur[c] = new_lags.get(k, np.nan)
        # Update time cyclic features if present
        if "hour" in cur.index:
            h = int(cur.get("hour", 0))
            h = (h + 1) % 24
            cur["hour"] = h
            if "hour_sin" in cur.index:
                cur["hour_sin"] = np.sin(2 * np.pi * h / 24)
            if "hour_cos" in cur.index:
                cur["hour_cos"] = np.cos(2 * np.pi * h / 24)

    # Build DataFrame of predictions with steps
    rows = []
    for i, p in enumerate(preds, start=1):
        rows.append({"step": i, "target": np.nan, "prediction": float(p)})
    out = pd.DataFrame(rows)
    return out

def build_last_features_for_forecast(raw_df: pd.DataFrame, feat_df: pd.DataFrame, feature_cols: List[str]):
    """
    Build the Series of features representing the situation at T (last observed timestamp).
    - raw_df: raw (high-frequency) dataframe (must contain datetime)
    - feat_df: processed feature dataframe (hourly features, with datetime)
    - feature_cols: list of features the model expects
    Returns pandas Series indexed by feature_cols and also includes 'datetime' = last_observed_ts
    """
    raw = raw_df.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
    last_observed = raw["datetime"].max()
    last_complete = last_observed.floor("h")

    # Try to find the features row corresponding to last_complete hour
    row = feat_df[feat_df["datetime"] == last_complete]
    if not row.empty:
        base = row.iloc[0].to_dict()
    else:
        # fallback: use last row of feat_df
        base = feat_df.sort_values("datetime").iloc[-1].to_dict()

    # If feature_cols expect partial features that we did not compute, leave NaN
    s = pd.Series({c: base.get(c, np.nan) for c in feature_cols})
    # Keep datetime of last_observed for labeling outputs
    s["datetime"] = last_observed
    return s
