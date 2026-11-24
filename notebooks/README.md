# Notebooks

Run from the project root environment (`pip install -r requirements.txt`), top to bottom.

| Notebook | Content |
|---|---|
| `data_exploration.ipynb` | Dataset structure and coverage, data-quality checks, income-group gender gap and variability, fertility correlations, visualisations |
| `life_expectancy_forecasting.ipynb` | Forecasts for 2011–2023 from a 2010 origin: baselines, ARIMA and gradient boosting, rolling-origin validation, error analysis |

The same forecasting logic is implemented in `src/model.py` and runs end to end with `python run_analysis.py`. Both use the shared loaders in `src/data.py` and the settings in `src/config.py`.
