"""Country-level life-expectancy forecasting from a 2010 origin.

Four models are compared on the same country-years:

* persistence     - last observed value
* drift           - last value + 10-year average annual gain
* ARIMA(1,1,0)    - per-country, linear trend, fitted on data up to the cutoff
* gradient boosting, direct multi-horizon: predicts LE(T+h) - LE(T), trained
  with life-expectancy features only and with all indicators

Leakage control: training features for an origin year T are built from data
truncated at T, and every training pair satisfies T + h <= cutoff.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import END_YEAR, FORECAST_CUTOFF, START_YEAR
from .data import countries_only

INDICATOR_COLUMNS = {
    "le": "life_expectancy_total",
    "le_m": "life_expectancy_male",
    "le_f": "life_expectancy_female",
    "fert": "fertility_rate",
    "death": "death_rate",
}

LE_ONLY = ["le", "le_slope5", "le_slope10"]
FIRST_ORIGIN = START_YEAR + 10          # features need a 10-year look-back
MAX_INTERPOLATION_GAP = 3
COVID_YEARS = (2020, 2022)

HGB_PARAMS = dict(
    max_depth=4, learning_rate=0.05, max_iter=300,
    min_samples_leaf=50, l2_regularization=1.0, random_state=42,
)

# Prediction column -> display label (order = display order)
MODELS = {
    "persistence": "Persistence",
    "drift": "Drift (10y)",
    "arima": "ARIMA(1,1,0)+drift",
    "ml_le_only": "ML, LE features only",
    "ml_full": "ML, all indicators",
}


def pivot_indicators(df, end_year=END_YEAR):
    """Return {key: year x country_code frame} of raw observations (countries only)."""
    c = countries_only(df)
    years = range(START_YEAR, end_year + 1)
    return {
        key: c.pivot(index="year", columns="country_code", values=col).reindex(years)
        for key, col in INDICATOR_COLUMNS.items()
    }


def _interpolate(frame):
    return frame.interpolate(limit=MAX_INTERPOLATION_GAP, limit_area="inside")


def features(raw, T):
    """Features known at origin year T (data truncated at T before interpolating)."""
    h = {k: _interpolate(v.loc[:T]) for k, v in raw.items()}
    le = h["le"]
    X = pd.DataFrame({
        "le": le.loc[T],
        "le_slope5": (le.loc[T] - le.loc[T - 5]) / 5,
        "le_slope10": (le.loc[T] - le.loc[T - 10]) / 10,
        "gender_gap": h["le_f"].loc[T] - h["le_m"].loc[T],
        "fertility": h["fert"].loc[T],
        "fertility_chg5": h["fert"].loc[T] - h["fert"].loc[T - 5],
        "death_rate": h["death"].loc[T],
        "death_chg5": h["death"].loc[T] - h["death"].loc[T - 5],
    })
    X.index.name = "country_code"
    return X


FULL = LE_ONLY + ["gender_gap", "fertility", "fertility_chg5", "death_rate", "death_chg5"]


def build_training_panel(raw, cutoff=FORECAST_CUTOFF, max_h=None):
    """Supervised pairs (origin T, horizon h) with T + h <= cutoff. Target = LE change."""
    max_h = max_h or (END_YEAR - cutoff)
    rows = []
    for T in range(FIRST_ORIGIN, cutoff):
        X = features(raw, T)
        for h in range(1, min(max_h, cutoff - T) + 1):
            d = X.copy()
            d["h"] = h
            d["origin"] = T
            # Labels use original observations, never interpolated values.
            d["y"] = raw["le"].loc[T + h] - raw["le"].loc[T]
            rows.append(d)
    panel = pd.concat(rows).dropna(subset=LE_ONLY + ["y"]).reset_index()
    assert (panel["origin"] + panel["h"]).max() <= cutoff, "target year leaks past cutoff"
    return panel


def fit_hgb(panel, feature_cols):
    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(panel[feature_cols + ["h"]], panel["y"])
    return model


def origin_frame(raw, T, n_horizons):
    """One row per country and horizon with baseline predictions made at origin T."""
    X = features(raw, T).dropna(subset=LE_ONLY)
    out = pd.concat([X.assign(h=h) for h in range(1, n_horizons + 1)]).reset_index()
    out["origin"] = T
    out["year"] = T + out["h"]
    out["persistence"] = out["le"]
    out["drift"] = out["le"] + out["le_slope10"] * out["h"]
    return out


def arima_forecasts(raw, cutoff=FORECAST_CUTOFF, n_horizons=END_YEAR - FORECAST_CUTOFF):
    """ARIMA(1,1,0) with linear trend per country, fitted on data up to the cutoff."""
    from statsmodels.tsa.arima.model import ARIMA

    hist = _interpolate(raw["le"].loc[:cutoff])
    origin = pd.Period(str(cutoff), freq="Y")
    rows, failures = [], []
    for code in hist.columns:
        s = hist[code].copy()
        s.index = pd.PeriodIndex(s.index.astype(int), freq="Y")
        if s.notna().sum() < 20 or pd.isna(s.loc[origin]):
            failures.append({"country_code": code, "reason": "fewer than 20 values or missing origin"})
            continue
        try:
            fc = ARIMA(s, order=(1, 1, 0), trend="t", missing="none").fit().forecast(n_horizons)
            rows += [{"country_code": code, "year": cutoff + i + 1, "arima": float(v)}
                     for i, v in enumerate(fc)]
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            failures.append({"country_code": code, "reason": str(exc)[:120]})
    return pd.DataFrame(rows, columns=["country_code", "year", "arima"]), pd.DataFrame(failures)


def forecast_from_cutoff(raw, cutoff=FORECAST_CUTOFF, end_year=END_YEAR):
    """Fit all models on data <= cutoff and forecast cutoff+1 .. end_year."""
    n_h = end_year - cutoff
    panel = build_training_panel(raw, cutoff, n_h)
    hgb_le = fit_hgb(panel, LE_ONLY)
    hgb_full = fit_hgb(panel, FULL)

    test = origin_frame(raw, cutoff, n_h)
    test["ml_le_only"] = test["le"] + hgb_le.predict(test[LE_ONLY + ["h"]])
    test["ml_full"] = test["le"] + hgb_full.predict(test[FULL + ["h"]])

    arima, failures = arima_forecasts(raw, cutoff, n_h)
    test = test.merge(arima, on=["country_code", "year"], how="left")
    return test, panel, failures


def attach_actuals(forecast, raw, countries):
    """Add observed values and country names; keep a common cohort for all models."""
    actual = raw["le"].stack().rename("actual").reset_index()
    actual.columns = ["year", "country_code", "actual"]
    out = forecast.merge(actual, on=["country_code", "year"], how="left")
    names = countries.drop_duplicates("country_code")[["country_code", "country"]]
    out = out.merge(names, on="country_code", how="left")
    cols = list(MODELS)
    return out.dropna(subset=["actual"] + cols).copy()


def score_table(evaluation, exclude_covid=False):
    d = evaluation
    if exclude_covid:
        d = d[~d["year"].between(*COVID_YEARS)]
    rows = {}
    for col, label in MODELS.items():
        e = d["actual"] - d[col]
        rows[label] = {
            "MAE": e.abs().mean(),
            "Median AE": e.abs().median(),
            "RMSE": float(np.sqrt((e ** 2).mean())),
            "Bias": float(-e.mean()),          # mean(forecast - actual)
        }
    t = pd.DataFrame(rows).T
    t["Skill vs persistence"] = 1 - t["MAE"] / t.loc["Persistence", "MAE"]
    return t.sort_values("MAE")


def rolling_origin_validation(raw, origins=(1995, 2000, 2005), horizon=5, cutoff=FORECAST_CUTOFF):
    """Expanding-window validation entirely inside the pre-cutoff period.

    Persistence, drift and both ML feature sets (ARIMA is evaluated only at the
    final origin). Hyperparameters are fixed, so this is a stability check.
    """
    panel = build_training_panel(raw, cutoff)
    frames = []
    for vo in origins:
        train = panel[panel["origin"] + panel["h"] <= vo]
        test = origin_frame(raw, vo, horizon)
        for name, cols in (("ml_le_only", LE_ONLY), ("ml_full", FULL)):
            model = fit_hgb(train, cols)
            test[name] = test["le"] + model.predict(test[cols + ["h"]])
        actual = raw["le"].stack().rename("actual").reset_index()
        actual.columns = ["year", "country_code", "actual"]
        test = test.merge(actual, on=["year", "country_code"], how="left")
        frames.append(test.dropna(subset=["actual", "persistence", "drift", "ml_le_only", "ml_full"]))
    ev = pd.concat(frames, ignore_index=True)
    labels = {k: v for k, v in MODELS.items() if k != "arima"}
    rows = {}
    for col, label in labels.items():
        e = ev["actual"] - ev[col]
        rows[label] = {"MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())), "Bias": float(-e.mean())}
    summary = pd.DataFrame(rows).T
    summary["Skill vs persistence"] = 1 - summary["MAE"] / summary.loc["Persistence", "MAE"]
    return ev, summary.sort_values("MAE")


def run_forecast(df, cutoff=FORECAST_CUTOFF, end_year=END_YEAR):
    """Full pipeline on the raw dataset. Returns a dict of results."""
    countries = countries_only(df)
    raw = pivot_indicators(df, end_year)
    forecast, panel, failures = forecast_from_cutoff(raw, cutoff, end_year)
    evaluation = attach_actuals(forecast, raw, countries)

    all_years = score_table(evaluation)
    ex_covid = score_table(evaluation, exclude_covid=True)
    yearly = evaluation.groupby("year").apply(
        lambda g: pd.Series({MODELS[c]: (g["actual"] - g[c]).abs().mean() for c in MODELS}))
    best = all_years.index[0]
    by_country = evaluation.groupby(["country_code", "country"]).apply(
        lambda g: pd.Series({MODELS[c]: (g["actual"] - g[c]).abs().mean() for c in MODELS})
    ).reset_index().sort_values(best, ascending=False)

    rolling_eval, rolling_summary = rolling_origin_validation(raw, cutoff=cutoff)

    metrics = {
        "forecast_cutoff": cutoff,
        "evaluated_country_years": int(len(evaluation)),
        "evaluated_countries": int(evaluation["country_code"].nunique()),
        "arima_failures": int(len(failures)),
        "best_model": best,
        "all_years": all_years.round(4).to_dict("index"),
        "excluding_2020_2022": ex_covid.round(4).to_dict("index"),
        "rolling_origin_validation": rolling_summary.round(4).to_dict("index"),
    }
    return {
        "forecast": forecast, "evaluation": evaluation, "metrics": metrics,
        "all_years": all_years, "ex_covid": ex_covid, "yearly": yearly,
        "by_country": by_country, "rolling_summary": rolling_summary,
        "rolling_evaluation": rolling_eval, "arima_failures": failures,
    }
