
"""Data-quality checks for the World Bank life-expectancy dataset."""

import pandas as pd

from .config import END_YEAR, START_YEAR

INDICATOR_COLUMNS = [
    "life_expectancy_total",
    "life_expectancy_male",
    "life_expectancy_female",
    "death_rate",
    "fertility_rate",
]


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run structural, completeness and consistency checks."""

    # 1. Check required columns
    required = {
        "country",
        "country_code",
        "year",
        "region",
        "income_group",
        "observation_type",
        *INDICATOR_COLUMNS,
    }

    missing_columns = sorted(required - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 2. Check duplicate country-year records
    duplicate_rows = df[
        df.duplicated(
            subset=["country_code", "year"],
            keep=False
        )
    ].copy()

    # 3. Missing values by indicator
    missing_report = pd.DataFrame({
        "indicator": INDICATOR_COLUMNS,
        "missing_count": [
            df[col].isna().sum()
            for col in INDICATOR_COLUMNS
        ],
        "missing_pct": [
            df[col].isna().mean() * 100
            for col in INDICATOR_COLUMNS
        ],
    })

    # 4. Observation counts by type
    type_report = (
        df["observation_type"]
        .value_counts(dropna=False)
        .rename_axis("observation_type")
        .reset_index(name="record_count")
    )

    # 5. Missing values by observation type
    missing_by_type = (
        df.groupby("observation_type", dropna=False)[INDICATOR_COLUMNS]
        .agg(lambda column: column.isna().sum())
        .reset_index()
    )

    # 6. Year coverage
    year_coverage = (
        df.groupby("year")
        .agg(
            total_records=("country_code", "size"),
            unique_observations=("country_code", "nunique"),
        )
        .reset_index()
    )

    expected_years = set(range(START_YEAR, END_YEAR + 1))
    available_years = set(df["year"].dropna().astype(int))
    missing_years = sorted(expected_years - available_years)

    # 7. Overall summary
    summary = {
        "total_rows": len(df),
        "unique_country_codes": df["country_code"].nunique(),
        "min_year": int(df["year"].min()) if len(df) else None,
        "max_year": int(df["year"].max()) if len(df) else None,
        "expected_years": f"{START_YEAR}-{END_YEAR}",
        "missing_years": missing_years,
        "duplicate_country_year_rows": len(duplicate_rows),
        "missing_country_codes": int(df["country_code"].isna().sum()),
        "missing_country_names": int(df["country"].isna().sum()),
    }

    return {
        "summary": summary,
        "missing_report": missing_report,
        "type_report": type_report,
        "missing_by_type": missing_by_type,
        "year_coverage": year_coverage,
        "duplicate_report": duplicate_rows,
    }


def save_validation_report(report: dict, output_dir) -> None:
    """Save all data-quality reports."""

    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report["missing_report"].to_csv(
        output_dir / "missing_values_report.csv",
        index=False,
    )

    report["type_report"].to_csv(
        output_dir / "records_by_observation_type.csv",
        index=False,
    )

    report["missing_by_type"].to_csv(
        output_dir / "missing_by_observation_type.csv",
        index=False,
    )

    report["year_coverage"].to_csv(
        output_dir / "year_coverage.csv",
        index=False,
    )

    report["duplicate_report"].to_csv(
        output_dir / "duplicate_records.csv",
        index=False,
    )

    with (output_dir / "data_quality_summary.txt").open(
        "w", encoding="utf-8"
    ) as file:
        for key, value in report["summary"].items():
            file.write(f"{key}: {value}\n")
