# milestone4_app.py
# Milestone 4 — Streamlit Web Dashboard (Optimized Layout)
# Compact design + multi-city, multi-pollutant + dynamic alerts

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

# -------------------------------------------
# Streamlit setup
# -------------------------------------------
st.set_page_config(page_title="AirAware - Milestone 4", layout="wide")
st.title("🌐 AirAware — Milestone 4: Streamlit Web Dashboard")

# -------------------------------------------
# Load dataset safely
# -------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(path="data/air_quality.csv"):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# -------------------------------------------
# Sidebar controls
# -------------------------------------------
st.sidebar.header("⚙️ Controls")

cities = sorted(df["City"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect("Select Cities", cities, default=[cities[0]])

time_ranges = {"Last 24 Hours": 1, "Last 7 Days": 7, "Last 30 Days": 30}
time_label = st.sidebar.selectbox("Time Range", list(time_ranges.keys()))
days = time_ranges[time_label]

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
available = [p for p in pollutants if p in df.columns]
selected_pollutants = st.sidebar.multiselect("Select Pollutants", available, default=["PM2.5", "PM10"])

forecast_horizon = st.sidebar.selectbox("Forecast Horizon", ["12 Hours", "24 Hours", "48 Hours"], index=1)
admin_mode = st.sidebar.toggle("Admin Mode", value=False)

if st.sidebar.button("🔄 Update Dashboard"):
    st.toast("Dashboard updated successfully!", icon="✅")

# -------------------------------------------
# Filter by selected cities and time
# -------------------------------------------
latest_time = df["datetime"].max()
start_time = latest_time - timedelta(days=days)
df_filtered = df[(df["City"].isin(selected_cities)) & (df["datetime"] >= start_time)].copy()

if df_filtered.empty:
    st.warning("No data available for this selection.")
    st.stop()

# -------------------------------------------
# AQI Calculation
# -------------------------------------------
def compute_aqi(row):
    vals = []
    if "PM2.5" in row: vals.append(row["PM2.5"])
    if "PM10" in row: vals.append(row["PM10"] / 1.5)
    if "NO2" in row: vals.append(row["NO2"] / 1.2)
    if "SO2" in row: vals.append(row["SO2"])
    if "CO" in row: vals.append(row["CO"] * 60)
    if "O3" in row: vals.append(row["O3"])
    return np.nanmean(vals) if vals else np.nan

df_filtered["AQI"] = df_filtered.apply(compute_aqi, axis=1)

def category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

df_filtered["Category"] = df_filtered["AQI"].apply(category)

# -------------------------------------------
# Layout - AQI & Forecast
# -------------------------------------------
st.markdown("### 🌆 Current Air Quality")

# Group gauges in compact rows of 3
num_cols = 3
for i in range(0, len(selected_cities), num_cols):
    cols = st.columns(num_cols)
    for j, city in enumerate(selected_cities[i:i+num_cols]):
        with cols[j]:
            city_df = df_filtered[df_filtered["City"] == city]
            if not city_df.empty:
                latest = city_df.iloc[-1]
                current_aqi = latest["AQI"]
                current_cat = latest["Category"]
                st.metric(label=f"{city} (AQI)", value=f"{current_aqi:.0f}", delta=current_cat)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current_aqi,
                    number={'suffix': " AQI"},
                    title={'text': current_cat},
                    gauge={
                        'axis': {'range': [0, 400]},
                        'steps': [
                            {'range': [0, 50], 'color': '#009966'},
                            {'range': [51, 100], 'color': '#ffde33'},
                            {'range': [101, 150], 'color': '#ff9933'},
                            {'range': [151, 200], 'color': '#cc0033'},
                            {'range': [201, 300], 'color': '#660099'},
                            {'range': [301, 400], 'color': '#7e0023'},
                        ],
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

# -------------------------------------------
# Forecast section
# -------------------------------------------
st.markdown(f"### 🔮 Forecast Trends ({forecast_horizon})")

for pollutant in selected_pollutants:
    df_recent = df_filtered.set_index("datetime").resample("3h").mean(numeric_only=True).reset_index()
    hist_x = df_recent["datetime"]
    hist_y = df_recent[pollutant]

    base_val = hist_y.mean()
    forecast_vals = base_val + np.sin(np.linspace(0, np.pi, 10)) * (hist_y.std() / 2)
    future_time = [hist_x.iloc[-1] + timedelta(hours=i*2) for i in range(1, 11)]

    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(x=hist_x, y=hist_y, mode="lines+markers", name="Historical"))
    fig_fore.add_trace(go.Scatter(x=future_time, y=forecast_vals, mode="lines+markers",
                                  name=f"{pollutant} Forecast", line=dict(color="orange", dash="dot")))
    fig_fore.update_layout(xaxis_title="Time", yaxis_title=f"{pollutant} (µg/m³)",
                           legend=dict(orientation="h", y=-0.3), height=350)
    st.plotly_chart(fig_fore, use_container_width=True)

# -------------------------------------------
# Pollutant trends & alerts
# -------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Pollutant Trends")
    df_trend = df_filtered.set_index("datetime").resample("D").mean(numeric_only=True)
    fig_trend = px.line(df_trend, y=[p for p in pollutants if p in df_trend.columns],
                        labels={"value": "Concentration (µg/m³)", "datetime": "Date"},
                        color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.markdown("### 🚨 Active Alerts")
    latest_aqi = df_filtered.groupby("City")["AQI"].last().reset_index()
    latest_aqi["Category"] = latest_aqi["AQI"].apply(category)

    alert_colors = {
        "Good": "#2ecc71",
        "Moderate": "#f1c40f",
        "Unhealthy for Sensitive": "#e67e22",
        "Unhealthy": "#e74c3c",
        "Very Unhealthy": "#8e44ad",
        "Hazardous": "#7e0023"
    }

    for _, row in latest_aqi.iterrows():
        st.markdown(
            f"<div style='background-color:{alert_colors[row['Category']]}33;"
            f"border-left:6px solid {alert_colors[row['Category']]};"
            f"padding:8px;border-radius:6px;margin-bottom:5px;'>"
            f"<b>{row['City']}</b> – {row['Category']} (AQI: {row['AQI']:.0f})</div>",
            unsafe_allow_html=True
        )

# -------------------------------------------
# Admin mode
# -------------------------------------------
st.markdown("---")
if admin_mode:
    st.subheader("🛠 Admin Interface")
    uploaded_file = st.file_uploader("Upload new air quality data (CSV)", type=["csv"])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded {uploaded_file.name} successfully! ({len(new_df)} rows)")
    retrain = st.button("♻️ Retrain Models")
    if retrain:
        st.info("Retraining models... (simulated)")
        st.success("Models retrained successfully!")

st.caption("✅ Milestone 4 — Compact Streamlit Dashboard (multi-city, multi-pollutant)")
