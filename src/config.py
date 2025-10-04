import os
from pathlib import Path

# Read token from environment variable
ESIOS_API_KEY = os.getenv("ESIOS_API_KEY")  # export ESIOS_API_KEY=...
BASE_URL = "https://api.esios.ree.es/indicators/"

# Data directories
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)