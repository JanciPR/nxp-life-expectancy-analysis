"""Reusable charts for the assignment (saved as standalone HTML)."""
import plotly.express as px
import plotly.graph_objects as go

from .analysis import BUCKETS, sankey_transitions
from .config import FIGURES_DIR
from .data import countries_only


def _save(fig, name):
    # CDN-hosted plotly.js keeps each HTML file small.
    fig.write_html(FIGURES_DIR / name, include_plotlyjs="cdn")


def make_all_charts(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    countries = countries_only(df)

    trend = (countries.dropna(subset=["income_group", "life_expectancy_total"])
             .groupby(["year", "income_group"], as_index=False)["life_expectancy_total"].mean())
    fig = px.line(trend, x="year", y="life_expectancy_total", color="income_group",
                  title="Life expectancy at birth by income group (unweighted country mean)",
                  labels={"life_expectancy_total": "Life expectancy (years)", "year": "Year"})
    _save(fig, "life_expectancy_by_income.html")

    map_df = countries[countries["year"].eq(2023)].dropna(subset=["life_expectancy_total"])
    fig = px.choropleth(map_df, locations="country_code", color="life_expectancy_total",
                        hover_name="country", locationmode="ISO-3",
                        color_continuous_scale="Viridis",
                        title="Life expectancy at birth by country (2023)",
                        labels={"life_expectancy_total": "Years"})
    _save(fig, "life_expectancy_world_map_2023.html")

    _, transitions = sankey_transitions(df)
    labels = BUCKETS + [f"{x} (2023)" for x in BUCKETS]
    source, target, value = [], [], []
    for _, row in transitions.iterrows():
        source.append(BUCKETS.index(str(row["category_1960"])))
        target.append(5 + BUCKETS.index(str(row["category_2023"])))
        value.append(int(row["countries"]))
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=18, thickness=18),
        link=dict(source=source, target=target, value=value))])
    fig.update_layout(title_text="Country movement across life-expectancy quintiles: 1960 to 2023")
    _save(fig, "life_expectancy_sankey.html")
