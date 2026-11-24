"""Statistical analysis: income-group gaps, variability, correlations and rank transitions.

All functions work on sovereign countries only (regional aggregates and
unclassified records are excluded) and compare 1960 with 2023 on a paired set of
countries observed in both years.
"""
import numpy as np
import pandas as pd

from .data import countries_only

START, END = 1960, 2023

BUCKETS = [
    "Very low life expectancy", "Low life expectancy",
    "Medium life expectancy", "High life expectancy",
    "Very high life expectancy",
]


def _paired_endpoints(df, value_cols):
    """Countries with income group and non-missing values in both 1960 and 2023."""
    d = countries_only(df).dropna(subset=["income_group", *value_cols])
    d = d[d["year"].isin([START, END])]
    complete = d.groupby("country_code")["year"].nunique()
    return d[d["country_code"].isin(complete[complete == 2].index)].copy()


def gender_gap_analysis(df):
    """Mean female-minus-male life-expectancy gap by income group, 1960 vs 2023."""
    d = _paired_endpoints(df, ["life_expectancy_male", "life_expectancy_female"])
    d["female_minus_male"] = d["life_expectancy_female"] - d["life_expectancy_male"]
    wide = d.pivot_table(index="income_group", columns="year", values="female_minus_male", aggfunc="mean")
    wide["change_1960_to_2023"] = wide[END] - wide[START]
    wide["absolute_change"] = wide["change_1960_to_2023"].abs()
    wide["countries"] = d.groupby("income_group")["country_code"].nunique()
    return wide.sort_values("absolute_change", ascending=False)


def variability_analysis(df):
    """Cross-country standard deviation and IQR of life expectancy by income group."""
    d = _paired_endpoints(df, ["life_expectancy_total"])

    def stats(g):
        x = g["life_expectancy_total"]
        return pd.Series({"std": x.std(), "iqr": x.quantile(0.75) - x.quantile(0.25)})

    s = d.groupby(["income_group", "year"]).apply(stats).unstack("year")
    out = pd.DataFrame(index=s.index)
    for m in ("std", "iqr"):
        out[f"{m}_1960"] = s[(m, START)]
        out[f"{m}_2023"] = s[(m, END)]
        out[f"{m}_change"] = out[f"{m}_2023"] - out[f"{m}_1960"]
    out["countries"] = d.groupby("income_group")["country_code"].nunique()
    return out.sort_values("std_change", key=np.abs, ascending=False)


def fertility_correlations(df, min_observations=10):
    """Per-country correlation of fertility with life expectancy.

    ``correlation`` uses annual levels (dominated by shared long-run trends);
    ``change_correlation`` uses year-to-year changes.
    """
    records = []
    for code, g in countries_only(df).sort_values("year").groupby("country_code"):
        pair = g[["year", "fertility_rate", "life_expectancy_total"]].dropna()
        if len(pair) < min_observations:
            continue
        level = pair["fertility_rate"].corr(pair["life_expectancy_total"])
        step = pair.assign(
            d_fert=pair["fertility_rate"].diff(),
            d_le=pair["life_expectancy_total"].diff(),
            gap=pair["year"].diff(),
        )
        step = step[step["gap"].eq(1)]          # consecutive years only
        change = step["d_fert"].corr(step["d_le"]) if len(step) >= min_observations else np.nan
        if pd.notna(level):
            records.append({"country_code": code, "country": g["country"].iloc[0],
                            "correlation": level, "change_correlation": change,
                            "observations": len(pair)})
    result = pd.DataFrame(records)
    if result.empty:
        return result, result
    return result.sort_values("correlation", ascending=False), result.sort_values("correlation")


def classify_countries(df, year):
    """Assign countries to life-expectancy quintiles for a year."""
    d = countries_only(df)
    d = d.loc[d["year"].eq(year), ["country_code", "country", "life_expectancy_total"]].dropna()
    # Rank first so ties do not collapse quantile buckets.
    d["rank"] = d["life_expectancy_total"].rank(method="first")
    d["category"] = pd.qcut(d["rank"], q=5, labels=BUCKETS)
    return d


def sankey_transitions(df):
    a, b = classify_countries(df, START), classify_countries(df, END)
    merged = a[["country_code", "country", "category"]].merge(
        b[["country_code", "category"]], on="country_code", suffixes=("_1960", "_2023"))
    counts = (merged.groupby(["category_1960", "category_2023"], observed=False)
                    .size().reset_index(name="countries"))
    return merged, counts
