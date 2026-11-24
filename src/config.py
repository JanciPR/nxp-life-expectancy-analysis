
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Data directories
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Output directories
OUTPUT_DIR = ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Analysis period
START_YEAR = 1960
END_YEAR = 2023

# Forecasting cutoff
FORECAST_CUTOFF = 2010

# World Bank indicator codes
INDICATORS = {
    "life_expectancy_total": "SP.DYN.LE00.IN",
    "life_expectancy_male": "SP.DYN.LE00.MA.IN",
    "life_expectancy_female": "SP.DYN.LE00.FE.IN",
    "death_rate": "SP.DYN.CDRT.IN",
    "fertility_rate": "SP.DYN.TFRT.IN",
}

# Create output directories if they don't exist
for directory in [
    PROCESSED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
