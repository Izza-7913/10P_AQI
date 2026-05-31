"""
Configuration file for AQI Predictor
Merged approach: Dual API fetching + Per-day-ahead models + Full ensemble
KARACHI ONLY - Single city focus
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
CITY_NAME   = os.getenv("CITY_NAME", "Karachi")
CITY_LAT    = float(os.getenv("CITY_LAT", 24.8607))
CITY_LON    = float(os.getenv("CITY_LON", 67.0011))

# DagsHub / MLflow
DAGSHUB_USER  = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Target city
TARGET_CITIES = [
    {"name": CITY_NAME, "lat": CITY_LAT, "lon": CITY_LON},
]

# Feature columns used for training (merged weather + pollutant features)
FEATURE_COLS = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
    "aqi_change_rate", "aqi_rolling_mean_3", "aqi_rolling_std_3",
    "pm25_pm10_ratio", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    # Enhanced weather features
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "humidity_temp_interaction", "heat_index", "wind_speed_normalized",
    "dewpoint", "temp_dewpoint_diff", "abs_humidity",
    "pressure_change_1h", "pressure_change_3h", "pressure_change_6h",
    "wind_effect_x", "wind_effect_y", "wind_speed_sq",
    "cloud_visibility_ratio",
    "is_morning_rush", "is_evening_rush",
]

# AQI Categories (EPA colors)
AQI_CATEGORIES = {
    (0, 50):   ("Good", "#00e400"),
    (51, 100): ("Moderate", "#ffff00"),
    (101, 150):("Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200):("Unhealthy", "#ff0000"),
    (201, 300):("Very Unhealthy", "#8f3f97"),
    (301, 500):("Hazardous", "#7e0023"),
}

ALERT_AQI_THRESHOLD = 150

# Model Configuration
MODELS_TO_TRAIN = {
    "random_forest": {
        "type": "sklearn",
        "params": {
            "n_estimators": 200, "max_depth": 15,
            "min_samples_leaf": 5, "random_state": 42, "n_jobs": -1,
        }
    },
    "gradient_boosting": {
        "type": "sklearn",
        "params": {
            "n_estimators": 200, "max_depth": 4, "random_state": 42,
        }
    },
    "xgboost": {
        "type": "sklearn",
        "params": {
            "n_estimators": 200, "max_depth": 6,
            "learning_rate": 0.05, "random_state": 42,
            "verbosity": 0, "n_jobs": -1,
        }
    },
}

# Forecast horizons
FORECAST_HORIZONS = [1, 2, 3]

# Data Validation
AQI_MIN = 0
AQI_MAX = 500

# Dashboard
DASHBOARD_REFRESH_INTERVAL = 300

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
