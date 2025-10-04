import pandas as pd
from src.config import RAW_DIR

def test_parquet_exists_and_has_expected_columns():
    """Check that at least one parquet exists and contains the right schema."""
    files = list(RAW_DIR.glob("*.parquet"))
    assert files, "No parquet files found in RAW_DIR"

    # Load the latest parquet
    latest_file = sorted(files)[-1]
    df = pd.read_parquet(latest_file)

    # Check required columns
    required_cols = {"datetime", "target"}
    assert required_cols.issubset(df.columns), f"Missing required columns in {latest_file}"

    # Check dataframe is not empty
    assert not df.empty, f"Parquet {latest_file} is empty"

    # Check datetime column is timezone-aware and in UTC
    assert str(df["datetime"].dt.tz) == "UTC", "Datetime column is not in UTC"
