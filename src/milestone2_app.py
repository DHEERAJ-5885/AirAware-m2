# src/milestone2_app.py
"""
Milestone 2: Model Training & Evaluation (ARIMA / Prophet / LSTM)
Use the same dataset: data/air_quality.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import pickle
from datetime import timedelta

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# sklearn metrics & preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# try imports that are heavier: show friendly error if missing
_missing = []
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    _missing.append("statsmodels")
try:
    from prophet import Prophet
except Exception:
    _missing.append("prophet")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    _missing.append("tensorflow")

if _missing:
    st.error(
        "Missing Python packages: " + ", ".join(_missing)
        + ".\nPlease install them into your environment (pip install -r requirements.txt)."
    )
    st.stop()

st.set_page_config(page_title="AirAware - Milestone 2", layout="wide")
st.title("🟩 AirAware — Milestone 2: Model Training & Evaluation")

# --------------------------
# Helper: load dataset (same logic as M1)
# --------------------------
@st.cache_data(show_spinner=True)
def load_data(path="data/air_quality.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Put your CSV there and retry.")
    df = pd.read_csv(path)
    # detect datetime column
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    date_col = date_cols[0] if date_cols else df.columns[0]
    df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    # rename key pollutants
    rename_map = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("pm25", "pm2.5", "pm_2_5"): rename_map[c] = "PM2.5"
        if low in ("pm10", "pm_10"): rename_map[c] = "PM10"
        if low == "no2": rename_map[c] = "NO2"
        if low == "so2": rename_map[c] = "SO2"
        if low == "co": rename_map[c] = "CO"
        if low in ("o3", "ozone"): rename_map[c] = "O3"
        if low in ("temp", "temperature"): rename_map[c] = "Temperature"
        if low == "city": rename_map[c] = "City"
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df

df = load_data("data/air_quality.csv")

# Sidebar controls
st.sidebar.header("Modeling Controls")
min_date, max_date = df.index.min().date(), df.index.max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# aggregation choice - we use hourly (h) or daily (D) resampling; LSTM expects regular intervals
agg_option = st.sidebar.selectbox("Aggregation freq (resample)", ["Hourly", "Daily"])
freq_map = {"Hourly":"h", "Daily":"D"}
freq = freq_map[agg_option]

# pollutants to train on
numeric_cols = df.select_dtypes(include="number").columns.tolist()
pollutant_priority = ["PM2.5","PM10","NO2","SO2","CO","O3"]
available = [c for c in pollutant_priority if c in numeric_cols]
if not available:
    st.error("No pollutant columns found in dataset.")
    st.stop()

# Default: select only one pollutant by default (first available). User can select more explicitly.
pollutants = st.sidebar.multiselect("Pollutants to model (select one or more)", available, default=[available[0]])

# train/test split fraction
test_frac = st.sidebar.slider("Test fraction (for evaluation)", 0.1, 0.4, 0.2)
# LSTM params
lstm_epochs = st.sidebar.number_input("LSTM epochs", min_value=1, max_value=200, value=20)
lstm_batch = st.sidebar.number_input("LSTM batch size", min_value=1, max_value=256, value=16)
lstm_units = st.sidebar.number_input("LSTM units", min_value=8, max_value=256, value=50)

# Forecast horizon selection (in number of periods of chosen freq)
horizon_input = st.sidebar.selectbox("Forecast horizon (units match aggregation)", ["1","3","6","12","24","48"])
horizon = int(horizon_input)

# Run training button
train_button = st.sidebar.button("Train models (ARIMA / Prophet / LSTM)")

# Where to save models
MODEL_DIR = "models_m2"
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------
# Utility functions
# --------------------------
def train_test_split_series(series, test_frac):
    # series: pd.Series indexed by datetime
    n = len(series)
    test_n = max(1, int(n * test_frac))
    train = series.iloc[: n - test_n]
    test = series.iloc[n - test_n :]
    return train, test

def eval_metrics(true, pred):
    # ensure numpy arrays
    true = np.array(true).astype(float)
    pred = np.array(pred).astype(float)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    return float(rmse), float(mae)

# --------------------------
# Model training functions
# --------------------------
# NOTE: don't cache the fit functions (models are complex objects)
def fit_arima(train_series):
    try:
        model = ARIMA(train_series, order=(5,1,0))
        fitted = model.fit()
        return fitted
    except Exception as e:
        st.warning("ARIMA fit failed: " + str(e))
        return None

def fit_prophet(train_series):
    # Prophet expects a df with ds, y
    dfp = train_series.reset_index()
    dfp.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(dfp)
    return m

def create_lstm_dataset(series, lookback=24):
    # produce X,y arrays for supervised learning (trained only on training series)
    arr = series.values.reshape(-1,1).astype(float)
    scaler = MinMaxScaler()
    arr_s = scaler.fit_transform(arr)
    X, y = [], []
    for i in range(lookback, len(arr_s)):
        X.append(arr_s[i-lookback:i, 0])
        y.append(arr_s[i, 0])
    if len(X) == 0:
        return np.empty((0,lookback,1)), np.empty((0,)), scaler
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def fit_lstm(train_series, lookback=24, epochs=20, batch_size=16, units=50):
    X, y, scaler = create_lstm_dataset(train_series, lookback=lookback)
    if len(X) == 0:
        return None, None, None
    model = Sequential()
    model.add(LSTM(units, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    return model, scaler, lookback

def lstm_forecast(model, scaler, history_series, lookback, periods):
    # history_series: pd.Series (train) used to initialize the last lookback values
    arr = history_series.values.reshape(-1,1).astype(float)
    arr_s = scaler.transform(arr)  # scaler was fit on train series in fit_lstm
    # last lookback sequence from training history
    last_seq = arr_s[-lookback:].reshape(1, lookback, 1)
    preds_s = []
    for _ in range(periods):
        p = model.predict(last_seq, verbose=0)[0,0]
        preds_s.append(p)
        # slide
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0,-1,0] = p
    # invert scale
    preds_s = np.array(preds_s).reshape(-1,1)
    preds = scaler.inverse_transform(preds_s).flatten()
    return preds

# --------------------------
# Training flow when user clicks
# --------------------------
results_store = {}  # will collect metrics & forecasts

if train_button:
    st.info("Training started. This may take several minutes (LSTM training is the slowest).")
    # iterate pollutants chosen
    for pollutant in pollutants:
        st.write(f"---\n### Training for pollutant: {pollutant}")
        series = df[pollutant].dropna()
        # resample to chosen freq and interpolate
        series = series.resample(freq).mean().interpolate()
        series.name = pollutant
        if series.empty or len(series) < 50:
            st.warning(f"Not enough data for {pollutant}. Skipping.")
            continue

        # split
        train_s, test_s = train_test_split_series(series, test_frac)
        results_store[pollutant] = {}

        # ARIMA
        st.write("Fitting ARIMA...")
        arima_model = fit_arima(train_s)
        arima_fore = None
        arima_rmse = arima_mae = None
        if arima_model is not None:
            try:
                arima_fore = arima_model.forecast(steps=len(test_s))
                arima_rmse, arima_mae = eval_metrics(test_s.values, np.array(arima_fore))
                # save model
                pickle.dump(arima_model, open(os.path.join(MODEL_DIR, f"arima_{pollutant}.pkl"), "wb"))
            except Exception as e:
                st.warning("ARIMA forecast failed: " + str(e))

        # Prophet
        st.write("Fitting Prophet...")
        prophet_model = None
        prop_fore = None
        prop_rmse = prop_mae = None
        try:
            prophet_model = fit_prophet(train_s)
            future = prophet_model.make_future_dataframe(periods=len(test_s), freq=freq)
            forecast = prophet_model.predict(future)
            prop_fore = forecast[['ds','yhat']].tail(len(test_s))['yhat'].values
            prop_rmse, prop_mae = eval_metrics(test_s.values, prop_fore)
            # save prophet
            with open(os.path.join(MODEL_DIR, f"prophet_{pollutant}.pkl"), "wb") as f:
                pickle.dump(prophet_model, f)
        except Exception as e:
            st.warning("Prophet failed: " + str(e))

        # LSTM
        st.write("Fitting LSTM...")
        lstm_model = None
        lstm_scaler = None
        lstm_lookback = 24 if freq == 'h' else 7  # 24 hours lookback if hourly, else 7 days
        lstm_fore = None
        lstm_rmse = lstm_mae = None
        try:
            lstm_model, lstm_scaler, lookback = fit_lstm(train_s, lookback=lstm_lookback,
                                                         epochs=int(lstm_epochs),
                                                         batch_size=int(lstm_batch),
                                                         units=int(lstm_units))
            if lstm_model is not None:
                # forecast for length of test set using only training history to initialize
                lstm_preds = lstm_forecast(lstm_model, lstm_scaler, train_s, lookback, len(test_s))
                lstm_fore = pd.Series(lstm_preds, index=test_s.index)
                lstm_rmse, lstm_mae = eval_metrics(test_s.values, lstm_fore.values)
                # save model (Keras)
                try:
                    lstm_model.save(os.path.join(MODEL_DIR, f"lstm_{pollutant}"))
                    # save scaler as well
                    with open(os.path.join(MODEL_DIR, f"lstm_scaler_{pollutant}.pkl"), "wb") as sf:
                        pickle.dump(lstm_scaler, sf)
                except Exception:
                    # saving model might fail in some environments; warn but continue
                    st.warning("Saving LSTM model/scaler failed (this is non-blocking).")
        except Exception as e:
            st.warning("LSTM failed: " + str(e))

        # Collect
        results_store[pollutant]['ARIMA'] = {'rmse': arima_rmse, 'mae': arima_mae, 'forecast': (arima_fore if arima_fore is not None else None)}
        results_store[pollutant]['Prophet'] = {'rmse': prop_rmse, 'mae': prop_mae, 'forecast': (prop_fore if prop_fore is not None else None)}
        results_store[pollutant]['LSTM'] = {'rmse': lstm_rmse, 'mae': lstm_mae, 'forecast': (lstm_fore if lstm_fore is not None else None)}

    st.success("Training completed.")
    st.session_state["results_store"] = results_store

# --------------------------
# If models exist in session state, show dashboard
# --------------------------
results_store = st.session_state.get("results_store", None)
if results_store:
    # Create performance dataframe
    perf_rows = []
    for pollutant, models in results_store.items():
        for mname, r in models.items():
            perf_rows.append({
                "Pollutant": pollutant,
                "Model": mname,
                "RMSE": r.get('rmse') if r.get('rmse') is not None else np.nan,
                "MAE": r.get('mae') if r.get('mae') is not None else np.nan
            })
    perf_df = pd.DataFrame(perf_rows)

    # Top row: Model performance bar chart
    st.markdown("## Model Performance (RMSE)")
    if not perf_df.empty:
        fig = px.bar(perf_df, x='Pollutant', y='RMSE', color='Model', barmode='group',
                     title="RMSE by Model and Pollutant")
        st.plotly_chart(fig, use_container_width=True)

    # Choose pollutant to inspect forecasts
    inspect_pollutant = st.selectbox("Select pollutant to inspect forecast", list(results_store.keys()))
    inspect_model = st.selectbox("Select model for forecast display", ["ARIMA", "Prophet", "LSTM"])
    # build forecast plot (actual vs predicted for test + future horizon)
    s = df[inspect_pollutant].dropna().resample(freq).mean().interpolate()
    train_s, test_s = train_test_split_series(s, test_frac)
    chosen_info = results_store[inspect_pollutant].get(inspect_model, {})
    test_fore = chosen_info.get('forecast', None)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name='Actual'))
    if test_fore is not None:
        test_idx = test_s.index
        if hasattr(test_fore, 'values'):
            pred_vals = np.array(test_fore)
        else:
            pred_vals = np.array(test_fore)
        fig2.add_trace(go.Scatter(x=test_idx, y=pred_vals, mode='lines', name=f'{inspect_model} (test forecast)'))

    # future horizon prediction
    future_idx = None
    future_vals = None
    try:
        if inspect_model == "ARIMA":
            pfile = os.path.join(MODEL_DIR, f"arima_{inspect_pollutant}.pkl")
            if os.path.exists(pfile):
                fitted = pickle.load(open(pfile, "rb"))
                fut = fitted.forecast(steps=horizon)
                future_idx = pd.date_range(start=s.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
                future_vals = np.array(fut)
        elif inspect_model == "Prophet":
            pfile = os.path.join(MODEL_DIR, f"prophet_{inspect_pollutant}.pkl")
            if os.path.exists(pfile):
                model = pickle.load(open(pfile, "rb"))
                future = model.make_future_dataframe(periods=horizon, freq=freq)
                pred = model.predict(future)
                fut = pred[['ds','yhat']].tail(horizon)
                future_idx = pd.to_datetime(fut['ds']).values
                future_vals = fut['yhat'].values
        elif inspect_model == "LSTM":
            mfolder = os.path.join(MODEL_DIR, f"lstm_{inspect_pollutant}")
            scaler_file = os.path.join(MODEL_DIR, f"lstm_scaler_{inspect_pollutant}.pkl")
            if os.path.exists(mfolder) and os.path.exists(scaler_file):
                # try to load Keras model and scaler to forecast horizon (may not work in all environments)
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(mfolder)
                    with open(scaler_file, "rb") as sf:
                        scaler = pickle.load(sf)
                    # use last lookback from combined series
                    lookback = 24 if freq=='h' else 7
                    future_vals = lstm_forecast(model, scaler, s, lookback, horizon)
                    future_idx = pd.date_range(start=s.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
                except Exception:
                    future_idx = None
                    future_vals = None
    except Exception:
        future_idx = None
        future_vals = None

    if future_idx is not None and future_vals is not None:
        fig2.add_trace(go.Scatter(x=future_idx, y=future_vals, mode='lines+markers', name=f'{inspect_model} (future {horizon})', line=dict(dash='dash')))
    fig2.update_layout(title=f"Actual & Forecast for {inspect_pollutant} ({inspect_model})",
                       xaxis_title="Date", yaxis_title=f"{inspect_pollutant}")
    st.plotly_chart(fig2, use_container_width=True)

    # Best model by pollutant table (by RMSE)
    table_rows = []
    for pollutant, models in results_store.items():
        best_m = None
        best_rmse = np.inf
        for mname, r in models.items():
            rm = r.get('rmse')
            if rm is None or np.isnan(rm):
                continue
            if rm < best_rmse:
                best_rmse = rm
                best_m = mname
        status = "Active" if best_m is not None else "No model"
        table_rows.append({"Pollutant": pollutant, "Best Model": best_m if best_m else "-", "RMSE": round(best_rmse,3) if best_m else np.nan, "Status": status})
    best_df = pd.DataFrame(table_rows)
    st.markdown("### Best Model by Pollutant")
    st.dataframe(best_df)

    # Forecast accuracy vs horizon (simple simulation using collected RMSEs)
    st.markdown("### Forecast Accuracy vs Horizon (RMSE)")
    horizons = [1,3,6,12,24] if freq=='h' else [1,3,7,14,30]
    acc_rows = []
    for mname in ["ARIMA", "Prophet", "LSTM"]:
        for h in horizons:
            per_poll_rmses = []
            for pollutant in results_store.keys():
                entry = results_store[pollutant].get(mname, {})
                rm = entry.get('rmse')
                if rm is None or np.isnan(rm):
                    continue
                per_poll_rmses.append(rm)
            if per_poll_rmses:
                acc_rows.append({"Model": mname, "Horizon": h, "RMSE": np.mean(per_poll_rmses)})
    acc_df = pd.DataFrame(acc_rows)
    if not acc_df.empty:
        fig_acc = px.line(acc_df, x='Horizon', y='RMSE', color='Model', markers=True, title="RMSE vs Horizon (approx)")
        st.plotly_chart(fig_acc, use_container_width=True)

    # download results
    buf = io.BytesIO()
    best_df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download best-models CSV", data=buf, file_name="best_models.csv")

else:
    st.info("No trained models in session. Use the sidebar 'Train models' button to run ARIMA, Prophet, and LSTM on selected pollutants.")
    st.caption("Training creates and saves models under ./models_m2/")
