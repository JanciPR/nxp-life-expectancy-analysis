"""Run the full pipeline: data checks, statistical analysis, charts and forecasts.

Usage (from the project root):
    python run_analysis.py

Raw World Bank CSV files must already be in data/raw/ (see README).
"""
import json

from src.analysis import (
    fertility_correlations,
    gender_gap_analysis,
    sankey_transitions,
    variability_analysis,
)
from src.config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR
from src.data import build_dataset
from src.model import run_forecast
from src.validation import save_validation_report, validate_dataset
from src.visuals import make_all_charts


def main():
    df = build_dataset()

    # 1. Data-quality checks
    save_validation_report(validate_dataset(df), REPORTS_DIR)

    # 2. Statistical analysis (country-level functions filter aggregates internally)
    gender = gender_gap_analysis(df)
    variability = variability_analysis(df)
    highest, lowest = fertility_correlations(df)
    _, transitions = sankey_transitions(df)

    gender.to_csv(TABLES_DIR / "gender_gap_by_income_group.csv")
    variability.to_csv(TABLES_DIR / "variability_by_income_group.csv")
    highest.head(10).to_csv(TABLES_DIR / "highest_fertility_life_expectancy_correlations.csv", index=False)
    lowest.head(10).to_csv(TABLES_DIR / "lowest_fertility_life_expectancy_correlations.csv", index=False)
    transitions.to_csv(TABLES_DIR / "sankey_transition_counts.csv", index=False)

    # 3. Charts
    make_all_charts(df)

    # 4. Forecasting from the 2010 origin, evaluated on 2011-2023
    results = run_forecast(df)
    results["evaluation"].to_csv(TABLES_DIR / "forecast_vs_actual.csv", index=False)
    results["all_years"].to_csv(TABLES_DIR / "forecast_scores_all_years.csv")
    results["ex_covid"].to_csv(TABLES_DIR / "forecast_scores_excluding_2020_2022.csv")
    results["yearly"].to_csv(TABLES_DIR / "forecast_mae_by_year.csv")
    results["by_country"].to_csv(TABLES_DIR / "forecast_mae_by_country.csv", index=False)
    results["rolling_summary"].to_csv(TABLES_DIR / "rolling_origin_validation.csv")
    (TABLES_DIR / "model_metrics.json").write_text(json.dumps(results["metrics"], indent=2))

    print("\nGender gap by income group (paired countries):\n", gender.round(2))
    print("\nVariability by income group (paired countries):\n", variability.round(2))
    print("\nHighest level correlations:\n", highest.head(10).round(2))
    print("\nLowest level correlations:\n", lowest.head(10).round(2))
    print("\nForecast scores, 2011-2023:\n", results["all_years"].round(2))
    print("\nExcluding 2020-2022:\n", results["ex_covid"].round(2))
    print("\nRolling-origin validation:\n", results["rolling_summary"].round(2))
    print("\nTables:", TABLES_DIR, "| Figures:", FIGURES_DIR, "| Reports:", REPORTS_DIR)


if __name__ == "__main__":
    main()
