"""Load and prepare World Bank indicator CSV files."""

import pandas as pd

from .config import (
    END_YEAR,
    INDICATORS,
    PROCESSED_DIR,
    RAW_DIR,
    START_YEAR,
)

# Map indicator names to the downloaded CSV filenames.
INDICATOR_FILES = {
    "life_expectancy_total": "life_expectancy_total.csv",
    "life_expectancy_male": "life_expectancy_male.csv",
    "life_expectancy_female": "life_expectancy_female.csv",
    "death_rate": "death_rate.csv",
    "fertility_rate": "fertility_rate.csv",
}

METADATA_FILE = "life_expectancy_country_metadata.csv"


def load_indicator(name, filename):
    """Read one World Bank indicator CSV and convert it from wide to long format."""
    file_path = RAW_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # World Bank indicator CSVs have four introductory rows.
    df = pd.read_csv(file_path, skiprows=4)
    print(f"Loaded {filename}: {df.shape}")

    year_columns = [
        str(year)
        for year in range(START_YEAR, END_YEAR + 1)
        if str(year) in df.columns
    ]
    if not year_columns:
        raise ValueError(f"No valid year columns found in {filename}")

    df = df[["Country Name", "Country Code"] + year_columns]

    df = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_columns,
        var_name="year",
        value_name=name,
    )
    df = df.rename(columns={"Country Name": "country", "Country Code": "country_code"})

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[name] = pd.to_numeric(df[name], errors="coerce")

    df = df.dropna(subset=["country_code", "year"])
    df["year"] = df["year"].astype(int)

    if df.duplicated(["country_code", "year"]).any():
        raise ValueError(f"Duplicate country-year records in {filename}")

    return df


def load_country_metadata():
    """Load region and income-group metadata and classify each record."""
    file_path = RAW_DIR / METADATA_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = pd.read_csv(file_path).rename(columns={
        "Country Code": "country_code",
        "Region": "region",
        "IncomeGroup": "income_group",
        "TableName": "metadata_country",
    })

    required_columns = {"country_code", "region", "income_group"}
    if not required_columns.issubset(metadata.columns):
        raise ValueError("Country metadata file is missing required columns.")

    metadata = metadata[
        ["country_code", "region", "income_group", "metadata_country"]
    ].drop_duplicates(subset=["country_code"])

    # Countries by default; aggregates have missing region/income group
    # or the region label "Aggregates".
    metadata["observation_type"] = "Country"
    aggregate_mask = (
        metadata["region"].isna()
        | metadata["income_group"].isna()
        | metadata["region"].eq("Aggregates")
    )
    metadata.loc[aggregate_mask, "observation_type"] = "Regional aggregate"

    # The World Bank "Not classified" record is kept separate.
    unclassified_mask = (
        metadata["metadata_country"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("not classified")
    )
    metadata.loc[unclassified_mask, "observation_type"] = "Unclassified"

    return metadata


def build_dataset(force=False):
    """Build the country-year dataset by merging all indicators and metadata.

    The result is cached in ``data/processed``. Pass ``force=True`` to rebuild
    it from the raw CSV files (needed whenever the raw files change).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / "world_bank_life_expectancy.csv"

    if output_file.exists() and not force:
        print("Loading existing processed dataset.")
        return pd.read_csv(output_file)

    indicator_frames = []
    for name, code in INDICATORS.items():
        print(f"Processing {name} ({code})")
        indicator_frames.append(load_indicator(name, INDICATOR_FILES[name]))

    data = indicator_frames[0]
    for indicator_df in indicator_frames[1:]:
        data = data.merge(
            indicator_df,
            on=["country_code", "country", "year"],
            how="outer",
            validate="one_to_one",
        )
    print("All indicators merged:", data.shape)

    data = data.merge(
        load_country_metadata(),
        on="country_code",
        how="left",
        validate="many_to_one",
    )
    data["observation_type"] = data["observation_type"].fillna("Unclassified")

    data = data[data["year"].between(START_YEAR, END_YEAR)]
    data = data.sort_values(["country_code", "year"]).reset_index(drop=True)
    data.to_csv(output_file, index=False)

    print("\nDataset successfully created!")
    print("Rows:", len(data))
    print("Columns:", len(data.columns))
    print("Saved to:", output_file)

    return data


def countries_only(df):
    """Keep sovereign-country records and drop regional aggregates and unclassified rows."""
    return df.loc[df["observation_type"].eq("Country")].copy()
