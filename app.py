"""Global Life Expectancy Explorer.

Run using:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR
from src.dashboard import (
    INDICATORS,
    VIEW_OPTIONS,
    apply_filters,
    country_options,
    country_profile,
    prepare,
    trend,
)
from src.data import build_dataset

st.set_page_config(page_title="Global Life Expectancy Explorer", page_icon="🌍", layout="wide")
st.title("🌍 Global Life Expectancy Explorer")
st.caption("World Bank indicators | 1960–2023 | sovereign countries only")


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    path = PROCESSED_DIR / "world_bank_life_expectancy.csv"
    data = pd.read_csv(path) if path.exists() else build_dataset()
    return prepare(data)


df = load_data()

# ------------------------------------------------------------- sidebar
st.sidebar.header("Filters")

selected_regions = st.sidebar.multiselect("Region", sorted(df["region"].dropna().unique()))
selected_incomes = st.sidebar.multiselect("Income group", sorted(df["income_group"].dropna().unique()))

# Country options narrow to the chosen regions / income groups.
selected_countries = st.sidebar.multiselect(
    "Country", country_options(df, selected_regions, selected_incomes),
    help="Leave empty to include every country that matches the filters above.",
)

min_year, max_year = int(df["year"].min()), int(df["year"].max())
selected_years = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

st.sidebar.header("Grouping")
view = st.sidebar.radio(
    "View charts by", list(VIEW_OPTIONS), index=0,
    help="Income group and Region average the countries in each group (unweighted). "
         "Country draws one line per country; Overall draws one line.",
)

filtered = apply_filters(df, selected_regions, selected_incomes, selected_countries, selected_years)
if filtered.empty:
    st.warning("No rows match these filters.")
    st.stop()

n_countries = filtered["country_code"].nunique()
if view == "Country" and n_countries > 15:
    st.info(f"{n_countries} countries are in view, so the lines are hard to tell apart. "
            "Select specific countries (or narrow region / income group) for a clearer comparison.")


# ------------------------------------------------------------- helpers
def trend_chart(column, title, y_label, height=420):
    data = trend(filtered, column, view)
    if data.empty:
        st.info("No data available for this indicator and selection.")
        return
    fig = px.line(
        data, x="year", y="value", color="group",
        labels={"year": "Year", "value": y_label, "group": view if view != "Overall" else "Selection"},
        hover_data={"countries": True}, title=title,
    )
    if column == "gender_gap":
        fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        height=height, hovermode="x unified",
        showlegend=not (view == "Country" and data["group"].nunique() > 15),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=50, b=70),
    )
    st.plotly_chart(fig, width="stretch")


profile = country_profile(filtered)
if profile.empty:
    st.warning("No life-expectancy observations for this selection and year range.")
    st.stop()

# ------------------------------------------------------------------ KPIs
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Countries in view", f"{n_countries}")
m2.metric("Life expectancy", f"{profile['Life expectancy'].mean():.1f} yrs")
m3.metric("Death rate", f"{profile['Death rate'].mean():.1f} per 1,000"
          if profile["Death rate"].notna().any() else "N/A")
m4.metric("Fertility rate", f"{profile['Fertility rate'].mean():.2f}"
          if profile["Fertility rate"].notna().any() else "N/A")
m5.metric("Female–male gap", f"{profile['Gap (F–M)'].mean():.1f} yrs"
          if profile["Gap (F–M)"].notna().any() else "N/A")
st.caption("Unweighted averages across countries in view, using each country's latest available year "
           f"within {selected_years[0]}–{selected_years[1]}.")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Life expectancy", "Death & fertility", "Gender gap", "Country profile", "Data",
])

# ------------------------------------------------- TAB 1: LIFE EXPECTANCY
with tab1:
    left, right = st.columns(2, gap="large")
    with left:
        trend_chart("life_expectancy_total", f"Life expectancy at birth by {view.lower()}"
                    if view != "Overall" else "Life expectancy at birth", "Life expectancy (years)", 480)

    with right:
        indicator_label = st.selectbox("Map indicator", list(INDICATORS), index=0)
        indicator = INDICATORS[indicator_label]
        map_year = selected_years[1]
        map_df = filtered[(filtered["year"] == map_year)].dropna(subset=[indicator])
        st.caption(f"Map shows {map_year}, the end of the selected year range.")
        if map_df.empty:
            st.info(f"No {map_year} data for this indicator and selection.")
        else:
            fig_map = px.choropleth(
                map_df, locations="country_code", locationmode="ISO-3", color=indicator,
                hover_name="country",
                hover_data={"country_code": False, "region": True, "income_group": True, indicator: ":.2f"},
                color_continuous_scale="RdBu" if indicator == "gender_gap" else "Viridis",
                labels={indicator: indicator_label},
            )
            fig_map.update_geos(showframe=False, showcoastlines=True, projection_type="natural earth")
            fig_map.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0),
                                  coloraxis_colorbar=dict(thickness=12, title=""))
            st.plotly_chart(fig_map, width="stretch")
    st.caption("Source: World Bank, SP.DYN.LE00.IN. Group lines are unweighted country means.")

# ----------------------------------------------- TAB 2: DEATH & FERTILITY
with tab2:
    left, right = st.columns(2, gap="large")
    with left:
        trend_chart("death_rate", "Crude death rate", "Deaths per 1,000 people")
    with right:
        trend_chart("fertility_rate", "Total fertility rate", "Births per woman")

# ------------------------------------------------------ TAB 3: GENDER GAP
with tab3:
    trend_chart("gender_gap", "Female–male life expectancy gap", "Gap (years)", 460)
    st.caption("Gap = female minus male life expectancy at birth. Positive values mean women live longer.")

    gap_latest = profile.dropna(subset=["Gap (F–M)"]).sort_values("Gap (F–M)", ascending=False)
    if not gap_latest.empty:
        st.markdown("#### Gap by country (latest available year)")
        n_show = st.slider("Countries to show at each end", 5, 25, 10) if len(gap_latest) > 10 else len(gap_latest)
        shown = gap_latest if len(gap_latest) <= 2 * n_show else pd.concat(
            [gap_latest.head(n_show), gap_latest.tail(n_show)])
        fig_bar = px.bar(
            shown.sort_values("Gap (F–M)"), x="Gap (F–M)", y="Country", color="Income group",
            orientation="h", hover_data=["Region", "Latest year"],
            labels={"Gap (F–M)": "Gap (years)"},
        )
        fig_bar.update_layout(height=max(320, 22 * len(shown)), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, width="stretch")

# ------------------------------------------------- TAB 4: COUNTRY PROFILE
with tab4:
    st.subheader("Countries in view")
    st.dataframe(
        profile, width="stretch", hide_index=True,
        column_config={
            "Life expectancy": st.column_config.NumberColumn(format="%.1f"),
            "Female": st.column_config.NumberColumn(format="%.1f"),
            "Male": st.column_config.NumberColumn(format="%.1f"),
            "Gap (F–M)": st.column_config.NumberColumn(format="%.1f"),
            "Death rate": st.column_config.NumberColumn(format="%.1f"),
            "Fertility rate": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption("Latest available year per country within the selected range.")

    st.subheader("Country detail")
    country = st.selectbox("Choose a country", profile["Country"].tolist())
    detail = filtered[filtered["country"] == country].sort_values("year")
    info = profile[profile["Country"] == country].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income group", info["Income group"])
    c2.metric("Region", info["Region"])
    c3.metric(f"Life expectancy ({int(info['Latest year'])})", f"{info['Life expectancy']:.1f} yrs")
    c4.metric("Female–male gap", f"{info['Gap (F–M)']:.1f} yrs" if pd.notna(info["Gap (F–M)"]) else "N/A")

    left, right = st.columns(2, gap="large")
    with left:
        both = detail.melt(id_vars="year", value_vars=["life_expectancy_female", "life_expectancy_male"],
                           var_name="Sex", value_name="Life expectancy").dropna()
        both["Sex"] = both["Sex"].map({"life_expectancy_female": "Female", "life_expectancy_male": "Male"})
        fig = px.line(both, x="year", y="Life expectancy", color="Sex", title=f"{country}: life expectancy by sex",
                      labels={"year": "Year", "Life expectancy": "Years"})
        fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.line(detail, x="year", y="gender_gap", title=f"{country}: female–male gap",
                      labels={"year": "Year", "gender_gap": "Gap (years)"})
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2, gap="large")
    for col, column, title, ylab in (
        (left, "death_rate", f"{country}: crude death rate", "Deaths per 1,000"),
        (right, "fertility_rate", f"{country}: fertility rate", "Births per woman"),
    ):
        with col:
            fig = px.line(detail, x="year", y=column, title=title, labels={"year": "Year", column: ylab})
            fig.update_layout(height=340, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------- TAB 5: DATA
with tab5:
    st.write(f"{len(filtered):,} observations across {n_countries} countries.")
    table = filtered.sort_values(["country", "year"]).drop(columns=["observation_type"], errors="ignore")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download filtered data", table.to_csv(index=False),
                       file_name="filtered_world_bank_data.csv", mime="text/csv")

st.divider()
st.caption("Data source: World Bank | life expectancy, death rate and fertility indicators | 1960–2023")
