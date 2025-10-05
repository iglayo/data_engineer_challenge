# scripts/run_model.py
import sys, os
from pathlib import Path
import pandas as pd
# ensure repo root on path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import train_model, save_model, evaluate, make_recursive_forecast, build_last_features_for_forecast
from src.features import train_val_split_time
from src.config import BASE_DIR

def main():
    processed = Path("data/processed")
    files = sorted(processed.glob("features_*.parquet"))
    if not files:
        raise SystemExit("No processed features found. Run scripts/process_features.py first.")

    feat_path = files[-1]
    feat = pd.read_parquet(feat_path)
    print("Loaded features:", feat_path, "shape:", feat.shape)

    # Load raw copy for T determination
    raw_files = sorted(processed.glob("raw_*.parquet"))
    if not raw_files:
        raise SystemExit("No raw copy found in data/processed. Run scripts/process_features.py first.")
    raw = pd.read_parquet(raw_files[-1])
    print("Loaded raw (for T):", raw_files[-1], "rows:", len(raw))

    # Prepare feature columns (exclude datetime and target)
    feature_cols = [c for c in feat.columns if c not in ("datetime", "target")]
    # Keep deterministic order (lag cols may be better if earlier): sort but prefer keep original order
    # We'll maintain current order from file
    print("Model feature count:", len(feature_cols))

    # Train/val split (time-based)
    train_df, val_df = train_val_split_time(feat, val_hours=24*7)
    print("Train rows:", len(train_df), "Val rows:", len(val_df))

    if train_df.empty:
        raise SystemExit("No training data after split. Check features generation.")

    model = train_model(train_df, feature_cols, target_col="target")
    save_model(model, name="rf_model.joblib")

    # Evaluate
    if not val_df.empty:
        evaluate(model, val_df, feature_cols, target_col="target")

    # Build last-features vector representing T (last observed timestamp)
    last_features = build_last_features_for_forecast(raw, feat, feature_cols)
    print("Last observed timestamp (T):", pd.to_datetime(last_features['datetime']))

    # Forecast recursively horizon=6
    forecast_df = make_recursive_forecast(model, last_features, feature_cols, horizon=6)

    # Label datetimes relative to last observed timestamp
    last_observed = pd.to_datetime(last_features['datetime'])
    last_observed = last_observed.tz_convert("UTC") if last_observed.tzinfo is not None else last_observed.tz_localize("UTC")
    forecast_df["datetime"] = [last_observed + pd.Timedelta(hours=i) for i in range(1, len(forecast_df)+1)]

    # Save predictions
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"predictions_{feat_path.stem}.parquet"
    forecast_df[["datetime", "target", "prediction"]].to_parquet(out_path, index=False)
    print("Saved predictions to", out_path)
    print(forecast_df)

if __name__ == "__main__":
    main()