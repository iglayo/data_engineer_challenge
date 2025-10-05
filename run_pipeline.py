from datetime import datetime, timedelta
from src.etl import fetch_esios, normalize_datetime, save_raw_parquet
import os

def main():
    # UTC para reproducibilidad; ESIOS sirve datos horario con timezone
    end = datetime.utcnow()
    start = end - timedelta(days=21)
    INDICATOR = 1293  # demanda realizada
    try:
        df = fetch_esios(INDICATOR, start, end)
    except Exception as e:
        print("API fetch failed:", e)
        # fallback: CSV de ejemplo, ruta aquí
        csv_path = "data/example_1293.csv"
        if os.path.exists(csv_path):
            from src.etl import load_local_csv
            df = load_local_csv(csv_path)
        else:
            raise

    if df.empty:
        print("No data returned.")
    else:
        df = normalize_datetime(df)
        print(df.head())
        save_raw_parquet(df, INDICATOR, start, end)

if __name__ == "__main__":
    main()