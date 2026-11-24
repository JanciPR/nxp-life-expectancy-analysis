"""Pure data helpers for the Streamlit dashboard (no Streamlit or Plotly imports)."""
import pandas as pd

from .data import countries_only

# "View by" option -> grouping column (None = one line for all selected countries)
VIEW_OPTIONS = {
    "Income group": "income_group",
    "Region": "region",
    "Country": "country",
    "Overall": None,
}

INDICATORS = {
    "Life expectancy at birth (years)": "life_expectancy_total",
    "Crude death rate (per 1,000 people)": "death_rate",
    "Total fertility rate (births per woman)": "fertility_rate",
    "Female–male life expectancy gap (years)": "gender_gap",
}

OVERALL_LABEL = "All selected countries"


def prepare(df):
    """Country-level records with a numeric year and the female-minus-male gap."""
    d = countries_only(df)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d.dropna(subset=["year"])
    d["year"] = d["year"].astype(int)
    d["gender_gap"] = d["life_expectancy_female"] - d["life_expectancy_male"]
    return d


def country_options(df, regions=(), incomes=()):
    """Countries that match the selected regions and income groups (cascading filter)."""
    d = df
    if regions:
        d = d[d["region"].isin(regions)]
    if incomes:
        d = d[d["income_group"].isin(incomes)]
    return sorted(d["country"].dropna().unique())


def apply_filters(df, regions=(), incomes=(), countries=(), years=None):
    d = df
    if regions:
        d = d[d["region"].isin(regions)]
    if incomes:
        d = d[d["income_group"].isin(incomes)]
    if countries:
        d = d[d["country"].isin(countries)]
    if years:
        d = d[d["year"].between(years[0], years[1])]
    return d


def trend(df, column, view):
    """Unweighted mean of ``column`` per year and group. Returns year, group, value, countries."""
    key = VIEW_OPTIONS[view]
    d = df.dropna(subset=[column]).copy()
    if key is None:
        key = "_group"
        d[key] = OVERALL_LABEL
    d = d.dropna(subset=[key])
    out = (d.groupby(["year", key])
             .agg(value=(column, "mean"), countries=("country_code", "nunique"))
             .reset_index()
             .rename(columns={key: "group"}))
    return out


def country_profile(df):
    """One row per country: latest available year with life expectancy and its indicators."""
    d = df.dropna(subset=["life_expectancy_total"])
    if d.empty:
        return pd.DataFrame()
    latest = d.loc[d.groupby("country_code")["year"].idxmax()]
    cols = {
        "country": "Country", "region": "Region", "income_group": "Income group",
        "year": "Latest year", "life_expectancy_total": "Life expectancy",
        "life_expectancy_female": "Female", "life_expectancy_male": "Male",
        "gender_gap": "Gap (F–M)", "death_rate": "Death rate",
        "fertility_rate": "Fertility rate",
    }
    return latest[list(cols)].rename(columns=cols).sort_values("Country").reset_index(drop=True)
