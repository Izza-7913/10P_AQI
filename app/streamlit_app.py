"""
AQI Predictor Dashboard — Merged Edition
Models loaded from MongoDB GridFS (works locally AND on Streamlit Cloud).
Features: Dual API data, per-day-ahead models, ensemble predictions,
          model comparison, SHAP explainability, dark theme.
"""

import os
import sys
import pickle
import json
import gridfs
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pymongo import MongoClient, errors as pymongo_errors
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
CITY      = os.getenv("CITY_NAME", "Karachi")

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
    "aqi_change_rate", "aqi_rolling_mean_3", "aqi_rolling_std_3",
    "pm25_pm10_ratio", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "humidity_temp_interaction", "heat_index", "wind_speed_normalized",
    "dewpoint", "temp_dewpoint_diff", "abs_humidity",
    "pressure_change_1h", "pressure_change_3h", "pressure_change_6h",
    "wind_effect_x", "wind_effect_y", "wind_speed_sq",
    "cloud_visibility_ratio",
    "is_morning_rush", "is_evening_rush",
]

AQI_COLORS = {
    "Good":                           "#00e400",
    "Moderate":                       "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy":                      "#ff0000",
    "Very Unhealthy":                 "#8f3f97",
    "Hazardous":                      "#7e0023",
}

AQI_BG_COLORS = {
    "Good":                           "#00e40022",
    "Moderate":                       "#ffff0022",
    "Unhealthy for Sensitive Groups": "#ff7e0022",
    "Unhealthy":                      "#ff000022",
    "Very Unhealthy":                 "#8f3f9722",
    "Hazardous":                      "#7e002322",
}


def aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


def get_aqi_color(aqi: float) -> str:
    cat = aqi_category(aqi)
    return AQI_COLORS.get(cat, "#cccccc")


def get_aqi_bg_color(aqi: float) -> str:
    cat = aqi_category(aqi)
    return AQI_BG_COLORS.get(cat, "#cccccc22")


# ── Data / model loaders ───────────────────────────────────────────────────────

@st.cache_resource(ttl=3600)
def load_models():
    """Load Day+1, Day+2, Day+3 models from MongoDB GridFS. Fallback to local."""
    models = {}
    
    if not MONGO_URI:
        st.warning("⚠️ MONGODB_URI not set. Checking for local models...")
        # Try local fallback
        for day in [1, 2, 3]:
            local = f"models/model_day_{day}.pkl"
            if os.path.exists(local):
                with open(local, "rb") as f:
                    models[day] = pickle.load(f)
        return models

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        db = client["aqi_db"]
        fs = gridfs.GridFS(db, collection="models")

        for day in [1, 2, 3]:
            grid_out = fs.find_one({"filename": f"model_day_{day}"})
            if grid_out:
                models[day] = pickle.loads(grid_out.read())
                st.success(f"✅ Loaded Day+{day} model from MongoDB GridFS")
            else:
                local = f"models/model_day_{day}.pkl"
                if os.path.exists(local):
                    with open(local, "rb") as f:
                        models[day] = pickle.load(f)
                    st.info(f"ℹ️ Loaded Day+{day} model from local file")

        client.close()
    except pymongo_errors.ServerSelectionTimeoutError as e:
        st.error(f"❌ Cannot connect to MongoDB: {e}")
        st.info("Loading local model fallbacks if available...")
        for day in [1, 2, 3]:
            local = f"models/model_day_{day}.pkl"
            if os.path.exists(local):
                with open(local, "rb") as f:
                    models[day] = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")

    return models


@st.cache_data(ttl=3600)
def load_recent_features():
    if not MONGO_URI:
        st.warning("⚠️ MONGODB_URI not set. No data available.")
        return pd.DataFrame()

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        col = client["aqi_db"][f"features_{CITY.lower()}"]
        cutoff = datetime.utcnow() - timedelta(hours=168)
        docs = list(col.find({"timestamp": {"$gte": cutoff}}, {"_id": 0}))
        client.close()
        
        if not docs:
            st.warning("⚠️ No data found in MongoDB for the last 7 days.")
            return pd.DataFrame()
            
        df = pd.DataFrame(docs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)
    except pymongo_errors.ServerSelectionTimeoutError as e:
        st.error(f"❌ Cannot connect to MongoDB: {e}")
        st.info("Please check: 1) MongoDB Atlas IP whitelist, 2) Streamlit Cloud secrets")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading features: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def load_metrics():
    if os.path.exists("models/metrics.json"):
        try:
            with open("models/metrics.json") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load metrics.json: {e}")
    return {}


def make_predictions(df: pd.DataFrame, models: dict) -> dict:
    """Generate predictions for Day+1, Day+2, Day+3."""
    if df.empty or not models:
        return {}

    valid_rows = df[FEATURE_COLS].dropna()
    if valid_rows.empty:
        st.warning("⚠️ No valid feature rows found for prediction.")
        return {}

    X = valid_rows.iloc[-1].values.reshape(1, -1)
    predictions = {}

    for day, model in models.items():
        try:
            pred_aqi = float(model.predict(X)[0])
            pred_aqi = max(0, min(500, pred_aqi))
            predictions[day] = {
                "aqi":      round(pred_aqi),
                "category": aqi_category(pred_aqi),
                "date":     (datetime.utcnow() + timedelta(days=day)).strftime("%A, %b %d"),
                "color":    get_aqi_color(pred_aqi),
                "bg":       get_aqi_bg_color(pred_aqi),
            }
        except Exception as e:
            st.warning(f"Prediction failed for Day+{day}: {e}")

    return predictions


# ── Page config & styling ─────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"AQI Predictor — {CITY}",
    page_icon="🌬️",
    layout="wide",
)

st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        [data-testid="stMetricValue"] {
            font-size: 2rem !important; font-weight: 700 !important; color: #ffffff !important;
        }
        [data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #a0a0a0 !important; }
        div[data-testid="stMetric"] {
            background-color: #1e2330 !important;
            border: 1px solid #2d3748 !important;
            border-radius: 12px !important; padding: 16px !important;
        }
        h1, h2, h3 { color: #e2e8f0 !important; }
        p, span, div { color: #cbd5e1 !important; }
        [data-testid="stSidebar"] { background-color: #161b22 !important; }
        .stButton > button {
            background-color: #2563eb !important; color: white !important; border-radius: 8px !important;
        }
        .streamlit-expanderHeader {
            color: #e2e8f0 !important; background-color: #1e2330 !important; border-radius: 8px !important;
        }
        .streamlit-expanderContent {
            background-color: #161b22 !important; border-radius: 0 0 8px 8px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title(f"🌬️ AQI Predictor — {CITY}")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Load data with error handling
with st.spinner("Loading models and data..."):
    models = load_models()
    df = load_recent_features()
    metrics = load_metrics()

# Debug info (remove in production)
with st.expander("🔧 Debug Info"):
    st.write(f"MONGO_URI set: {bool(MONGO_URI)}")
    st.write(f"Models loaded: {list(models.keys())}")
    st.write(f"Data rows: {len(df)}")
    if not df.empty:
        st.write(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        st.write(f"Columns: {list(df.columns)}")

# ── Current AQI ────────────────────────────────────────────────────────────────

st.subheader("📊 Current Air Quality")

if not df.empty and "aqi" in df.columns:
    recent_aqi = df["aqi"].dropna()
    if not recent_aqi.empty:
        current_aqi = int(recent_aqi.iloc[-1])
        current_category = aqi_category(current_aqi)
        current_color = get_aqi_color(current_aqi)
        current_bg = get_aqi_bg_color(current_aqi)

        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='text-align:center;padding:24px;border-radius:16px;
                        background:{current_bg};border:2px solid {current_color}'>
                <h1 style='color:{current_color};font-size:80px;margin:0'>{current_aqi}</h1>
                <h3 style='margin:4px 0;color:#e2e8f0'>Current AQI</h3>
                <p style='font-size:18px;margin:0;color:{current_color};font-weight:600'>{current_category}</p>
            </div>
            """, unsafe_allow_html=True)

        if current_aqi > 150:
            st.error(
                f"⚠️ **Air quality alert!** AQI is {current_aqi} ({current_category}). "
                "Limit outdoor activity and wear a mask if going outside."
            )
    else:
        st.warning("⚠️ AQI column exists but has no valid values.")
else:
    st.warning("⚠️ No current AQI data available. Run the feature pipeline to populate data.")

# ── 3-Day Forecast ─────────────────────────────────────────────────────────────

st.subheader("📅 3-Day AQI Forecast")

preds = make_predictions(df, models)

if preds:
    cols = st.columns(3)
    for i, (day, info) in enumerate(preds.items()):
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center;padding:20px;border-radius:12px;
                        background:{info["bg"]};border:2px solid {info["color"]}'>
                <p style='margin:0;font-weight:bold;color:#a0a0a0'>{info["date"]}</p>
                <h2 style='color:{info["color"]};margin:6px 0;font-size:48px'>{info["aqi"]}</h2>
                <p style='margin:0;font-size:13px;color:{info["color"]};font-weight:500'>{info["category"]}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("No predictions available. Ensure models are trained and data is loaded.")

# ── Historical AQI chart ───────────────────────────────────────────────────────

st.subheader("📈 AQI — Last 7 Days")

if not df.empty and "aqi" in df.columns and "timestamp" in df.columns:
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["aqi"],
            mode="lines+markers", name="AQI",
            line=dict(color="#4f8ef7", width=2),
            marker=dict(size=3),
        ))
        # 24h rolling mean
        if len(df) >= 24:
            df["aqi_rolling_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["aqi_rolling_24h"],
                mode="lines", name="24h Rolling Mean",
                line=dict(color="#ef4444", width=2),
            ))
        thresholds = [
            (100, "#cccc00", "Moderate"),
            (150, "#ff7e00", "Unhealthy"),
            (200, "#ff0000", "Very Unhealthy"),
        ]
        for val, color, label in thresholds:
            fig.add_hline(
                y=val, line_dash="dot", line_color=color,
                annotation_text=label, annotation_position="right",
                annotation_font_color=color,
            )
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time", yaxis_title="AQI",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e2330"),
            yaxis=dict(gridcolor="#1e2330"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering AQI chart: {e}")
else:
    st.info("No historical AQI data available. Run the feature pipeline to fetch data.")

# ── Pollutant breakdown ────────────────────────────────────────────────────────

st.subheader("🔬 Pollutant Levels (Last 7 Days)")

if not df.empty:
    available = [c for c in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"] if c in df.columns]
    if available and "timestamp" in df.columns:
        try:
            fig2 = px.line(
                df, x="timestamp", y=available,
                labels={"value": "µg/m³", "variable": "Pollutant"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#1e2330"),
                yaxis=dict(gridcolor="#1e2330"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering pollutant chart: {e}")
    else:
        st.info("No pollutant data columns found in the dataset.")
else:
    st.info("No data available for pollutant breakdown.")

# ── Weather conditions ────────────────────────────────────────────────────────

st.subheader("🌤️ Current Weather Conditions")

if not df.empty:
    latest = df.iloc[-1]
    
    # Fix visibility unit: Open-Meteo returns meters, convert to km
    visibility_val = latest.get('visibility', 'N/A')
    if pd.notna(visibility_val) and isinstance(visibility_val, (int, float)):
        visibility_km = visibility_val / 1000
        visibility_str = f"{visibility_km:.1f} km"
    else:
        visibility_str = "N/A"
    
    wcols = st.columns(4)
    weather_metrics = [
        ("🌡️ Temperature", f"{latest.get('temperature_2m', 'N/A')}°C"),
        ("💧 Humidity", f"{latest.get('relative_humidity_2m', 'N/A')}%"),
        ("💨 Wind Speed", f"{latest.get('wind_speed_10m', 'N/A')} km/h"),
        ("📊 Pressure", f"{latest.get('surface_pressure', 'N/A')} hPa"),
        ("☁️ Cloud Cover", f"{latest.get('cloud_cover', 'N/A')}%"),
        ("👁️ Visibility", visibility_str),
        ("🌧️ Precipitation", f"{latest.get('precipitation', 'N/A')} mm"),
        ("🧭 Wind Direction", f"{latest.get('wind_direction_10m', 'N/A')}°"),
    ]
    for i, (label, value) in enumerate(weather_metrics):
        with wcols[i % 4]:
            st.metric(label, value)
else:
    st.warning("No weather data available.")

# ── Model performance ──────────────────────────────────────────────────────────

st.subheader("📊 Model Performance")

if metrics:
    try:
        rows = []
        for horizon, m in metrics.items():
            rows.append({
                "Forecast Horizon": horizon.replace("day_", "Day +"),
                "Best Model":       m.get("best_model", "—"),
                "RMSE":             m.get("rmse"),
                "MAE":              m.get("mae"),
                "R²":               m.get("r2"),
                "CV Std":           m.get("cv_std"),
            })
        perf_df = pd.DataFrame(rows)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        # Model comparison bar chart per horizon
        st.subheader("🔍 Model Comparison by Horizon")
        for horizon, m in metrics.items():
            if "all_models" in m and m["all_models"]:
                try:
                    comp_df = pd.DataFrame([
                        {"Model": k, "CV RMSE": v}
                        for k, v in m["all_models"].items()
                    ]).sort_values("CV RMSE")

                    fig_comp = px.bar(
                        comp_df, x="Model", y="CV RMSE",
                        title=f"{horizon.replace('day_', 'Day +')} — Model CV RMSE Comparison",
                        color="CV RMSE", color_continuous_scale="RdYlGn_r",
                    )
                    fig_comp.update_layout(
                        height=350,
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font=dict(color="#e2e8f0"),
                        xaxis=dict(gridcolor="#1e2330"),
                        yaxis=dict(gridcolor="#1e2330"),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering model comparison for {horizon}: {e}")
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")
else:
    st.info("Run the training pipeline to populate metrics.")

# ── AQI guide ─────────────────────────────────────────────────────────────────

with st.expander("ℹ️ AQI Reference Guide"):
    guide = pd.DataFrame([
        ("0–50",   "Good",                           "Air quality is satisfactory"),
        ("51–100", "Moderate",                       "Acceptable; some pollutants may affect sensitive people"),
        ("101–150","Unhealthy for Sensitive Groups", "General public unaffected; sensitive groups at risk"),
        ("151–200","Unhealthy",                      "Everyone may experience health effects"),
        ("201–300","Very Unhealthy",                 "Health alert — serious effects for everyone"),
        ("301–500","Hazardous",                      "Emergency conditions — entire population at risk"),
    ], columns=["AQI", "Category", "Health Implication"])
    st.dataframe(guide, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.write("📊 Data: Open-Meteo Air Quality + Weather APIs")
with col2:
    st.write(f"🕐 Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
with col3:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()