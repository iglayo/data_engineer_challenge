# scripts/process_features.py
import sys
import os
import pandas as pd
from pathlib import Path

# Asegura que la raíz del repo esté en sys.path para que "import src..." funcione
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import RAW_DIR
from src.features import build_features_pipeline, resample_to_hourly

def main():
    # 1) Localiza el parquet más reciente dentro de data/raw
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("No se encontraron ficheros parquet en data/raw")
    latest = files[-1]
    print(f" Usando fichero: {latest.name}")

    # 2) Carga el parquet en memoria (raw  — alta frecuencia)
    df_raw = pd.read_parquet(latest)
    print("\nPrimeras filas del dataset raw:")
    print(df_raw.head())
    print(f"\nRaw rows: {len(df_raw)}  | Min ts: {df_raw['datetime'].min()}  Max ts: {df_raw['datetime'].max()}")

    # 3) Preparar output dir
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Guardar copia del raw (por trazabilidad)
    raw_copy_path = out_dir / f"raw_{latest.stem}.parquet"
    df_raw.to_parquet(raw_copy_path, index=False)
    print(f"\n💾 Copia raw guardada en: {raw_copy_path}")

    # 5) Crear vista hourly agregando por hora (mean)
    try:
        hourly = resample_to_hourly(df_raw, datetime_col="datetime", value_col="target", how="mean")
    except Exception as e:
        raise SystemExit(f"Error al resamplear a hourly: {e}")

    hourly_path = out_dir / f"hourly_{latest.stem}.parquet"
    hourly.to_parquet(hourly_path, index=False)
    print(f"💾 Hourly aggregated saved: {hourly_path}")
    print(f"Hourly rows: {len(hourly)} | Min: {hourly['datetime'].min()} Max: {hourly['datetime'].max()}")

    # 6) Generar features a partir de la vista hourly
    try:
        df_features = build_features_pipeline(hourly)
    except Exception as e:
        raise SystemExit(f"Error al generar features: {e}")

    features_path = out_dir / f"features_{latest.stem}.parquet"
    df_features.to_parquet(features_path, index=False)
    print("\n Features generadas correctamente:")
    print(f"Shape final: {df_features.shape}")
    print("Columnas principales:", df_features.columns[:20].tolist())
    print(f"💾 Guardado en: {features_path}")

if __name__ == "__main__":
    main()