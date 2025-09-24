# milestone1_app.py
# Streamlit interactive dashboard for Milestone 1 (Data Preprocessing & EDA)
# Works with: air_quality.csv (multi-city dataset with pollutants + Temperature)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="AirAware - Milestone 1", layout="wide")
st.title("🌍 AirAware — Milestone 1: Data Preprocessing & EDA")

# --- Load Data ---
@st.cache_data(show_spinner=True)
def load_data(path="data/air_quality.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Put your CSV there and retry.")

    df = pd.read_csv(path)

    # detect datetime column
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    date_col = date_cols[0] if date_cols else df.columns[0]
    df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")

    # rename important columns
    rename_map = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("pm25", "pm2.5", "pm_2_5"):
            rename_map[c] = "PM2.5"
        if low in ("pm10", "pm_10"):
            rename_map[c] = "PM10"
        if low == "no2":
            rename_map[c] = "NO2"
        if low == "so2":
            rename_map[c] = "SO2"
        if low == "co":
            rename_map[c] = "CO"
        if low in ("o3", "ozone"):
            rename_map[c] = "O3"
        if low in ("temp", "temperature"):
            rename_map[c] = "Temperature"
        if low == "city":
            rename_map[c] = "City"

    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df

# Load dataset
with st.spinner("Loading dataset..."):
    try:
        df = load_data("data/air_quality.csv")
    except Exception as e:
        st.error(str(e))
        st.stop()

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Data Controls")

# Date range
min_date = df.index.min().date()
max_date = df.index.max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Aggregation & smoothing
agg_option = st.sidebar.selectbox("Aggregation frequency", ["Hourly", "Daily", "Weekly"])
rolling_window = st.sidebar.slider("Rolling window (periods, 0=no smoothing)", 0, 30, 7)

# City filter
if "City" in df.columns:
    cities = df["City"].dropna().unique().tolist()
    chosen_loc = st.sidebar.selectbox("Select City", ["All"] + cities)
    if chosen_loc != "All":
        df = df[df["City"] == chosen_loc]

# Variables selection
numeric_cols = df.select_dtypes(include="number").columns.tolist()
pollutant_priority = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature"]
available = [c for c in pollutant_priority if c in numeric_cols]
selected = st.sidebar.multiselect("Select variable(s)", available,
                                  default=["PM2.5"] if "PM2.5" in available else available[:1])
show_table = st.sidebar.checkbox("Show processed data table", value=False)

# --- Filter by date ---
start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df_range = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

if df_range.empty:
    st.warning("No data available for the chosen range. Choose a wider range.")
    st.stop()

# --- Data Quality ---
st.markdown("## 🧹 Data Quality Report")
st.write(f"**Total rows:** {len(df)}")
st.write(f"**Date range:** {df.index.min().date()} → {df.index.max().date()}")

missing = df.isna().sum()
if missing.sum() > 0:
    st.write("**Missing values per column:**")
    st.dataframe(missing[missing > 0])
else:
    st.success("No missing values found ✅")

dup_count = df.duplicated().sum()
if dup_count > 0:
    st.warning(f"⚠️ Found {dup_count} duplicate rows")
else:
    st.success("No duplicate rows ✅")

# --- Resampling ---
# ✅ FIXED: use lowercase "h" instead of "H"
freq_map = {"Hourly": "h", "Daily": "D", "Weekly": "W"}
freq = freq_map.get(agg_option, "D")
numeric_df = df_range.select_dtypes(include="number")
df_resampled = numeric_df.resample(freq).mean()

# Rolling smoothing
if rolling_window > 0 and selected:
    df_plot = df_resampled[selected].rolling(window=rolling_window, min_periods=1).mean()
else:
    df_plot = df_resampled[selected].copy() if selected else df_resampled.copy()

# --- Statistical Summary ---
st.markdown("## 📊 Statistical Summary")
if not df_resampled.empty:
    st.dataframe(df_resampled.describe().T.style.format("{:.2f}"))

# --- Summary Metrics ---
st.markdown("## 📌 Summary Metrics")
if selected:
    cols = st.columns(min(len(selected), 4))
    for i, var in enumerate(selected[:4]):
        series = df_resampled[var].dropna()
        if series.empty:
            mean_v = min_v = max_v = "NA"
        else:
            mean_v = f"{series.mean():.2f}"
            min_v = f"{series.min():.2f}"
            max_v = f"{series.max():.2f}"
        cols[i % len(cols)].metric(label=f"{var} (mean)", value=mean_v,
                                   delta=f"min {min_v} / max {max_v}")

# --- Time Series ---
st.markdown("## 📈 Time Series")
if selected:
    fig = px.line(df_plot.reset_index(), x="datetime", y=selected,
                  labels={"value": "Value", "datetime": "Date"})
    st.plotly_chart(fig, use_container_width=True)

# --- Distribution & Correlation ---
left, right = st.columns([2, 1])

with left:
    st.markdown("### 📊 Distribution (Histogram)")
    if selected:
        var = selected[0]
        fig_h = px.histogram(df_resampled, x=var, nbins=50,
                             title=f"{var} distribution ({agg_option})")
        st.plotly_chart(fig_h, use_container_width=True)

with right:
    st.markdown("### 🔥 Correlation Heatmap")
    if len(selected) >= 2:
        corr = df_resampled[selected].corr()
    elif len(selected) == 1:
        corr = df_resampled[[selected[0]]].corr()
    else:
        corr = pd.DataFrame()

    if not corr.empty:
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="Greens", zmid=0))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Please select at least one variable to see correlation.")

# --- Processed Data ---
if show_table:
    st.markdown("### 📋 Processed (resampled) data")
    st.dataframe(df_resampled[selected].reset_index().head(200)
                 if selected else df_resampled.reset_index().head(200))

# --- Download/Save ---
st.markdown("## 💾 Download / Save")
if not df_resampled.empty:
    csv_bytes = df_resampled.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download processed CSV",
                       data=csv_bytes,
                       file_name="processed_air_quality.csv",
                       mime="text/csv")
    if st.button("Save processed CSV to data/processed_air_quality.csv"):
        out_path = "data/processed_air_quality.csv"
        df_resampled.to_csv(out_path)
        st.success(f"Saved processed CSV to {out_path}")

st.markdown("---")
st.caption(f"Data range: {df.index.min().date()} → {df.index.max().date()} — Total rows: {len(df)}")
