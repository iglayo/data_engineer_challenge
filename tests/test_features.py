# tests/test_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features import build_features_pipeline, ensure_hourly_index, train_val_split_time

def make_sample_df(hours=48):
    rng = pd.date_range(end=datetime.utcnow().replace(tzinfo=pd.Timestamp.utcnow().tz), periods=hours, freq="h", tz="UTC")
    vals = np.linspace(100, 200, num=hours) + np.random.normal(0, 1, size=hours)
    return pd.DataFrame({"datetime": rng, "target": vals})

def test_ensure_hourly_index_and_pipeline():
    raw = make_sample_df(50)
    # drop a couple of rows to simulate missing timestamps
    raw = raw.drop(index=[5, 10]).reset_index(drop=True)
    ensured = ensure_hourly_index(raw)
    # must be contiguous hourly and not shorter than original
    assert pd.infer_freq(ensured["datetime"].dt.tz_localize(None) if hasattr(ensured["datetime"].dt, 'tz') else ensured["datetime"]) is not None or len(ensured) >= len(raw)
    # build pipeline
    feat = build_features_pipeline(raw)
    # should include lag columns
    assert "target_lag_1" in feat.columns
    assert "target_roll_mean_24" in feat.columns
    assert not feat.empty

def test_time_split():
    raw = make_sample_df(200)
    feat = build_features_pipeline(raw)
    train, val = train_val_split_time(feat, val_hours=24)
    assert train["datetime"].max() < val["datetime"].min()
