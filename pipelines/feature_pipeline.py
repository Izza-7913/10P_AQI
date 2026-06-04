"""
Feature Pipeline
----------------
Runs every hour via GitHub Actions.
1. Fetches current air quality + weather from Open-Meteo (dual APIs).
2. Engineers rich features (time, derived, lags, cyclical, weather trends).
3. Upserts into MongoDB city-specific collection.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()

MONGO_URI  = os.getenv("MONGODB_URI")
LAT        = float(os.getenv("CITY_LAT", 24.8607))
LON        = float(os.getenv("CITY_LON", 67.0011))
CITY       = os.getenv("CITY_NAME", "Karachi")

MAX_RETRIES = 3
TIMEOUT_SECONDS = 30


def fetch_current_data(lat: float, lon: float) -> dict:
    """Fetch latest hourly data from Open-Meteo air quality + weather APIs."""

    # Air quality API
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone",
        "timezone": "auto", "past_days": 1, "forecast_days": 1,
    }
    # Weather API
    wx_url = "https://api.open-meteo.com/v1/forecast"
    wx_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,"
                  "surface_pressure,cloud_cover,visibility,wind_direction_10m",
        "timezone": "auto", "past_days": 1, "forecast_days": 1,
    }

    # Retry with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            aq_resp = requests.get(aq_url, params=aq_params, timeout=TIMEOUT_SECONDS).json()
            wx_resp = requests.get(wx_url, params=wx_params, timeout=TIMEOUT_SECONDS).json()
            return {"air_quality": aq_resp, "weather": wx_resp}
        except requests.exceptions.ReadTimeout:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[Feature Pipeline] Open-Meteo timeout (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"[Feature Pipeline] Open-Meteo error: {e} (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_to_dataframe(raw: dict) -> pd.DataFrame:
    """Merge air quality and weather data on timestamp."""
    aq = raw["air_quality"]["hourly"]
    wx = raw["weather"]["hourly"]

    df_aq = pd.DataFrame({
        "timestamp":        aq["time"],
        "pm2_5":            aq["pm2_5"],
        "pm10":             aq["pm10"],
        "nitrogen_dioxide": aq["nitrogen_dioxide"],
        "ozone":            aq["ozone"],
    })

    df_wx = pd.DataFrame({
        "timestamp":              wx["time"],
        "temperature_2m":         wx["temperature_2m"],
        "relative_humidity_2m":   wx["relative_humidity_2m"],
        "wind_speed_10m":         wx["wind_speed_10m"],
        "precipitation":          wx["precipitation"],
        "surface_pressure":       wx.get("surface_pressure", [1013]*len(wx["time"])),
        "cloud_cover":            wx.get("cloud_cover", [0]*len(wx["time"])),
        "visibility":             wx.get("visibility", [10]*len(wx["time"])),
        "wind_direction_10m":     wx.get("wind_direction_10m", [0]*len(wx["time"])),
    })

    # Merge on timestamp so mismatched lengths don\'t crash
    df = pd.merge(df_aq, df_wx, on="timestamp", how="inner")
    return df.dropna(subset=["pm2_5"])


def pm25_to_aqi(pm25: float) -> float:
    """Converts PM2.5 concentration (\u03bcg/m\u00b3) to US AQI."""
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for (c_lo, c_hi, i_lo, i_hi) in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo)
    return 500.0


def aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rich feature engineering: time, derived, lags, cyclical, weather trends."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Time-based features ---
    df["hour"]         = df["timestamp"].dt.hour
    df["day"]          = df["timestamp"].dt.day
    df["month"]        = df["timestamp"].dt.month
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # --- Cyclical encodings ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Rush hour flags (Karachi) ---
    df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 10)).astype(int)
    df["is_evening_rush"] = ((df["hour"] >= 17) & (df["hour"] <= 20)).astype(int)

    # --- AQI calculation ---
    df["aqi"] = df["pm2_5"].apply(pm25_to_aqi)

    # --- Derived AQI features ---
    df["aqi_change_rate"]    = df["aqi"].diff().fillna(0)
    df["aqi_rolling_mean_3"] = df["aqi"].rolling(3, min_periods=1).mean()
    df["aqi_rolling_std_3"]  = df["aqi"].rolling(3, min_periods=1).std().fillna(0)
    df["pm25_pm10_ratio"]    = (df["pm2_5"] / (df["pm10"] + 1e-5)).round(4)

    # --- Lag features ---
    for lag in [1, 2, 3]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag).bfill()

    # --- Weather-derived features ---
    df["humidity_temp_interaction"] = df["relative_humidity_2m"] * df["temperature_2m"]
    df["heat_index"] = df["temperature_2m"] + 0.05 * df["relative_humidity_2m"]
    df["wind_speed_normalized"] = df["wind_speed_10m"].apply(lambda x: min(x / 100, 1.0) if pd.notna(x) else 0)

    # Dewpoint approximation
    a, b = 17.27, 237.7
    gamma = (a * df["temperature_2m"]) / (b + df["temperature_2m"]) + np.log(df["relative_humidity_2m"] / 100.0)
    df["dewpoint"] = (b * gamma) / (a - gamma)
    df["temp_dewpoint_diff"] = df["temperature_2m"] - df["dewpoint"]
    df["abs_humidity"] = (df["relative_humidity_2m"] / 100) * 2.17e17 * np.exp(-a * df["temperature_2m"] / (b + df["temperature_2m"])) / (273.15 + df["temperature_2m"])

    # Pressure changes
    df["pressure_change_1h"] = df["surface_pressure"].diff(1).fillna(0)
    df["pressure_change_3h"] = df["surface_pressure"].diff(3).fillna(0)
    df["pressure_change_6h"] = df["surface_pressure"].diff(6).fillna(0)

    # Wind effect vectors
    wind_rad = np.radians(df["wind_direction_10m"].fillna(0))
    df["wind_effect_x"] = df["wind_speed_10m"] * np.cos(wind_rad)
    df["wind_effect_y"] = df["wind_speed_10m"] * np.sin(wind_rad)
    df["wind_speed_sq"] = df["wind_speed_10m"] ** 2

    # Cloud-visibility ratio
    df["cloud_visibility_ratio"] = df["cloud_cover"] / (df["visibility"] + 1)

    # --- Targets: AQI N days ahead ---
    for day_ahead in [1, 2, 3]:
        df[f"target_day_{day_ahead}"] = df["aqi"].shift(-24 * day_ahead)

    return df


def upsert_to_mongodb(df: pd.DataFrame, city: str):
    """Upsert features into MongoDB city-specific collection."""
    client = MongoClient(MONGO_URI)
    db     = client["aqi_db"]
    col    = db[f"features_{city.lower()}"]

    operations = []
    for _, row in df.iterrows():
        doc = row.to_dict()
        doc["city"] = city
        # Convert NaN to None for MongoDB
        for k, v in doc.items():
            if pd.isna(v):
                doc[k] = None
        doc["timestamp"] = pd.Timestamp(doc["timestamp"]).to_pydatetime()
        operations.append(
            UpdateOne(
                {"timestamp": doc["timestamp"], "city": city},
                {"$set": doc},
                upsert=True,
            )
        )

    if operations:
        result = col.bulk_write(operations)
        print(f"[Feature Pipeline] Upserted {result.upserted_count} | Modified {result.modified_count}")

    client.close()


def run():
    print(f"[{datetime.now(timezone.utc)}] Running feature pipeline for {CITY}...")
    raw = fetch_current_data(LAT, LON)
    df  = parse_to_dataframe(raw)
    df  = compute_features(df)
    upsert_to_mongodb(df, CITY)
    print(f"Done. Processed {len(df)} rows.")


if __name__ == "__main__":
    run()
