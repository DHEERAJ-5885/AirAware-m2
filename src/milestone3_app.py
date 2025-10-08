# milestone3_app.py
# Milestone 3 — Air Quality Alert & 7-Day Forecast Visualization (multi-city + pollutants)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# ---------------------------------------------
# Streamlit setup
# ---------------------------------------------
st.set_page_config(page_title="AirAware - Milestone 3", layout="wide")
st.title("🟧 AirAware — Milestone 3: Air Quality Alert System")

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(path="data/air_quality.csv"):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️  File not found! Make sure `data/air_quality.csv` exists.")
    st.stop()

if df.empty:
    st.error("⚠️ No valid datetime values found in dataset.")
    st.stop()

# ---------------------------------------------
# Sidebar Controls
# ---------------------------------------------
st.sidebar.header("⚙️  Filters & Options")

cities = sorted(df["City"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect("Select City/Cities", cities, default=cities[:1])
df = df[df["City"].isin(selected_cities)]

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
available = [p for p in pollutants if p in df.columns]
selected_pollutants = st.sidebar.multiselect("Select Pollutants", available, default=["PM2.5", "PM10"])

# ✅ Safe Date Handling
if not df.empty:
    min_date = pd.to_datetime(df["datetime"].min()).date()
    max_date = pd.to_datetime(df["datetime"].max()).date()
else:
    min_date = max_date = None

if min_date and max_date:
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)]
else:
    st.warning("No valid dates available.")
    st.stop()

if df.empty:
    st.warning("No data for chosen filters.")
    st.stop()

# ---------------------------------------------
# AQI Computation Logic
# ---------------------------------------------
def compute_aqi(row):
    sub_index = []
    if "PM2.5" in row and not pd.isna(row["PM2.5"]):
        sub_index.append(row["PM2.5"])
    if "PM10" in row and not pd.isna(row["PM10"]):
        sub_index.append(row["PM10"] / 1.5)
    if "NO2" in row and not pd.isna(row["NO2"]):
        sub_index.append(row["NO2"] / 1.2)
    if "SO2" in row and not pd.isna(row["SO2"]):
        sub_index.append(row["SO2"])
    if "CO" in row and not pd.isna(row["CO"]):
        sub_index.append(row["CO"] * 60)
    if "O3" in row and not pd.isna(row["O3"]):
        sub_index.append(row["O3"])
    if not sub_index:
        return np.nan
    return np.mean(sub_index)

df["AQI"] = df.apply(compute_aqi, axis=1)

def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

df["Category"] = df["AQI"].apply(aqi_category)

alert_colors = {
    "Good": "#2ecc71",
    "Moderate": "#f1c40f",
    "Unhealthy for Sensitive": "#e67e22",
    "Unhealthy": "#e74c3c",
    "Very Unhealthy": "#8e44ad",
    "Hazardous": "#7e0023"
}

# ---------------------------------------------
# Main Dashboard Layout
# ---------------------------------------------
col1, col2 = st.columns([2, 3])

# ✅ MULTI-CITY CURRENT AIR QUALITY
with col1:
    st.markdown("### 🌆 Current Air Quality")
    current_cols = st.columns(len(selected_cities))
    for i, city in enumerate(selected_cities):
        city_data = df[df["City"] == city]
        if not city_data.empty:
            latest = city_data.iloc[-1]
            cat = latest["Category"]
            color = alert_colors.get(cat, "#bdc3c7")
            current_cols[i].markdown(
                f"""
                <div style='background-color:{color}; padding:15px; border-radius:10px; text-align:center; color:white'>
                    <b>{city}</b><br>
                    AQI: {latest['AQI']:.0f}<br>
                    <small>{cat}</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Gauge for first selected city
    if selected_cities:
        city_df = df[df["City"] == selected_cities[0]]
        if not city_df.empty:
            latest = city_df.iloc[-1]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest["AQI"],
                gauge={
                    "axis": {"range": [0, 400]},
                    "steps": [
                        {"range": [0, 50], "color": "#009966"},
                        {"range": [51, 100], "color": "#ffde33"},
                        {"range": [101, 150], "color": "#ff9933"},
                        {"range": [151, 200], "color": "#cc0033"},
                        {"range": [201, 300], "color": "#660099"},
                        {"range": [301, 400], "color": "#7e0023"},
                    ],
                },
                title={"text": f"AQI Level ({selected_cities[0]})"}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

# 7-DAY FORECAST
with col2:
    st.markdown("### 🔮 7-Day Forecast ")
    daily = df.resample("D", on="datetime").mean(numeric_only=True)
    daily["AQI_7D"] = daily["AQI"].rolling(window=7, min_periods=1).mean()

    last7 = daily.tail(7).reset_index()
    cols = st.columns(len(last7))
    for i, (_, row) in enumerate(last7.iterrows()):
        cat = aqi_category(row["AQI"])
        color = alert_colors[cat]
        date_label = row["datetime"].strftime("%b %d")
        cols[i].markdown(
            f"""
            <div style='background-color:{color}; padding:15px; border-radius:10px; text-align:center; color:white'>
                <b>{date_label}</b><br>
                AQI: {row["AQI"]:.0f}<br>
                <small>{cat}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------
# Pollutant Trend Graph
# ---------------------------------------------
st.markdown("### 📈 Pollutant Concentrations")
fig_poll = px.line(df, x="datetime", y=selected_pollutants,
                   color_discrete_sequence=px.colors.qualitative.Set2,
                   title=f"Concentration Levels in {', '.join(selected_cities)}")
st.plotly_chart(fig_poll, use_container_width=True)

# ---------------------------------------------
# Alerts Panel
# ---------------------------------------------
st.markdown("### 🚨 Active Alerts")
latest_aqi = df.groupby("City")["AQI"].last().reset_index()
latest_aqi["Category"] = latest_aqi["AQI"].apply(aqi_category)

for _, row in latest_aqi.iterrows():
    st.markdown(
        f"<div style='background-color:{alert_colors[row['Category']]};padding:8px;border-radius:6px;margin-bottom:5px;'>"
        f"<b>{row['City']}</b> – {row['Category']} (AQI: {row['AQI']:.0f})</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------
# Download processed data
# ---------------------------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Filtered Data", csv, "filtered_aqi.csv", "text/csv")

st.caption("✅ Milestone 3 Completed — Multi-City Alert & 7-Day Forecast Visualization Module")
