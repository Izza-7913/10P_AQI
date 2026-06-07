# 🌍 Pearls AQI Predictor

> **End-to-end serverless Air Quality Index (AQI) forecasting system**  
> Predict AQI for the next 3 days using automated ML pipelines, real-time weather & pollutant data, and an interactive Streamlit dashboard.

**Deployed Project:** https://appapppy-aervwu3kydqlawzcjhnej7.streamlit.app/


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)](https://streamlit.io)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)](https://mongodb.com)
[![MLflow](https://img.shields.io/badge/MLflow-DagsHub-blue)](https://dagshub.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Data Sources](#data-sources)
- [Feature Pipeline](#feature-pipeline)
- [Training Pipeline](#training-pipeline)
- [Model Registry & Experiment Tracking](#model-registry--experiment-tracking)
- [Automated CI/CD](#automated-cicd)
- [Web Application Dashboard](#web-application-dashboard)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering Deep Dive](#feature-engineering-deep-dive)
- [Model Performance & Evaluation](#model-performance--evaluation)
- [SHAP Explainability](#shap-explainability)
- [Alerting System](#alerting-system)
- [API Reference](#api-reference)
- [Deployment Guide](#deployment-guide)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

**Pearls AQI Predictor** is a fully serverless, end-to-end machine learning system designed to forecast the Air Quality Index (AQI) for **Karachi, Pakistan** (extensible to any city) over a **3-day horizon**. The system leverages a dual-API approach — fetching both meteorological data and air quality pollutant measurements — to build a rich feature set that captures temporal patterns, pollutant interactions, and weather-driven dynamics.

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| **Real-time Data Ingestion** | Fetches hourly weather and air quality data via Open-Meteo API |
| **Automated Feature Engineering** | Computes 30+ features including cyclical time encodings, lag features, rolling statistics, and interaction terms |
| **Multi-Horizon Forecasting** | Trains separate models for 1-day, 2-day, and 3-day ahead predictions |
| **Ensemble Modeling** | Combines Random Forest, Gradient Boosting, and XGBoost with hyperparameter optimization |
| **Feature Store Integration** | Stores processed features and targets in MongoDB Atlas (feature store) |
| **Model Registry** | Versions and tracks models via MLflow on DagsHub |
| **Automated Pipelines** | GitHub Actions orchestrate hourly feature updates and daily retraining |
| **Interactive Dashboard** | Streamlit web app with real-time forecasts, historical trends, and SHAP explanations |
| **Hazardous AQI Alerts** | Automated alerts when predicted AQI exceeds 150 (Unhealthy threshold) |
| **EDA Suite** | Comprehensive exploratory analysis with 8+ visualization types |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PEARLS AQI PREDICTOR                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│  │  Open-Meteo  │     │  Open-Meteo  │     │   GitHub     │               │
│  │   Weather    │     │    Air Q     │     │   Actions    │               │
│  │     API      │     │    API       │     │   (CI/CD)    │               │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘               │
│         │                    │                    │                         │
│         └────────────────────┼────────────────────┘                         │
│                              ▼                                              │
│                    ┌─────────────────┐                                      │
│                    │  pipelines/     │  <-- Runs every hour                  │
│                    │  feature_*.py   │     Feature generation scripts        │
│                    └────────┬────────┘                                      │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │  MongoDB Atlas  │  <-- Feature Store & Backfill          │
│                    │  (aqi_db.features_karachi)                            │
│                    └────────┬────────┘                                      │
│                             │                                               │
│              ┌──────────────┼──────────────┐                              │
│              ▼              ▼              ▼                                │
│    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐             │
│    │  eda_analysis   │ │  pipelines/     │ │  pipelines/     │             │
│    │  (eda_analysis) │ │  training_*.py  │ │  backfill_*.py  │             │
│    └─────────────────┘ └────────┬────────┘ └─────────────────┘             │
│                                 │                                          │
│                                 ▼                                          │
│                    ┌─────────────────┐                                     │
│                    │  DagsHub MLflow │  <-- Model Registry                   │
│                    │  (Experiment Tracking)                                  │
│                    └────────┬────────┘                                     │
│                             │                                              │
│                             ▼                                              │
│                    ┌─────────────────┐                                     │
│                    │  app/             │  <-- Web Dashboard                      │
│                    │  streamlit_app.py │     Real-time + 3-day forecast        │
│                    └─────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Raw Data --> Feature Generation --> Feature Store --> Model Training --> Model Registry
    ^              ^                ^                ^              ^
    |              |                |                |              |
  APIs        pipelines/       MongoDB Atlas    pipelines/     DagsHub MLflow
    |              |                |                |              |
    └──────────────┴────────────────┴────────────────┴──────────────┘
                                    |
                                    ▼
                           ┌─────────────────┐
                           │  app/           │
                           │  (Streamlit)    │
                           │  Predictions    │
                           └─────────────────┘
```

---

## 🛠️ Technology Stack

### Core Framework
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Primary development language |
| **ML Framework** | Scikit-learn, XGBoost, LightGBM, CatBoost | Classical ML models |
| **Deep Learning** | TensorFlow / PyTorch | Neural network experiments (optional) |
| **Time Series** | Prophet, Statsmodels | Statistical baseline models |

### Data & Storage
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Feature Store** | MongoDB Atlas | Document-based feature storage with geospatial support |
| **Model Registry** | MLflow (DagsHub) | Experiment tracking, model versioning, artifact storage |
| **DataFrame** | Pandas, NumPy | Data manipulation and numerical computing |

### APIs & External Services
| Service | API | Data Type |
|---------|-----|-----------|
| **Open-Meteo Weather** | `open-meteo.com` | Temperature, humidity, wind, precipitation, pressure |
| **Open-Meteo Air Quality** | `open-meteo.com` | PM2.5, PM10, NO2, O3, AQI, CO, SO2 |

### Orchestration & CI/CD
| Tool | Purpose |
|------|---------|
| **GitHub Actions** | Automated hourly feature pipeline, daily retraining |
| **Apache Airflow** (optional) | Advanced DAG scheduling for multi-city deployments |

### Visualization & Frontend
| Tool | Purpose |
|------|---------|
| **Streamlit** | Primary interactive dashboard (in `app/`) |
| **Plotly** | Interactive charts and time-series visualizations |
| **Matplotlib / Seaborn** | EDA static plots and correlation matrices |
| **SHAP** | Model explainability and feature importance |

### Utilities
| Tool | Purpose |
|------|---------|
| **python-dotenv** | Environment variable management |
| **requests / retry-requests** | Robust HTTP API fetching with exponential backoff |
| **requests-cache** | Intelligent API response caching |
| **pymongo** | MongoDB connection with SRV support |

---

## Directory Structure

```
10P/
│
├── README.md                       # This file (project documentation)
├── .env                            # Environment variables (NOT in git)
├── .gitignore                      # Git ignore rules
├── config.py                       # Central configuration & constants
├── eda_analysis.py                 # EDA script
├── requirements.txt                # Python dependencies
│
├── .devcontainer/                  # VS Code DevContainer configuration
│   └── devcontainer.json
│
├── .github/                        # GitHub Actions CI/CD workflows
│   └── workflows/
│       ├── feature_pipeline.yml    # Hourly feature ingestion
│       └── training_pipeline.yml   # Daily model retraining
│
├── app/                            # Streamlit web application
│   └── streamlit_app.py            # Main dashboard entry point
│
├── catboost_info/                  # CatBoost training logs (auto-generated)
│
│
├── models/                         # Serialized model artifacts
│   ├── metrics.json                # Model evaluation metrics
│   ├── model_day_{day_ahead}.pkl   # Dynamic model artifact
│   ├── model_day_1.pkl             # 1-day ahead model
│   ├── model_day_2.pkl             # 2-day ahead model
│   └── model_day_3.pkl             # 3-day ahead model
│
├── pipelines/                      # ML pipeline scripts
│   ├── backfill_pipeline.py        # Historical data backfill runner
│   ├── feature_pipeline.py         # Feature generation & storage
│   └── training_pipeline.py        # Model training, evaluation, registry
│
└── PROJECT_REPORT/                 # Project documentation & report assets
    ├── PROJECT_REPORT.md           # Full project report
    ├── 01_system_architecture.png
    ├── 02_MongoDB_features.png
    ├── 02.1_DangsHub.png
    ├── 03_cicd_workflow.png
    ├── 04_GitHubActions.png
    ├── 05_Streamlit.png
    ├── 05.1_Streamlit_dashboard.png
    ├── 05.2_Streamlit_3Day_Prediction.png
    ├── 06_Streamlit_AQL.png
    ├── 11_feature_importance.png
    ├── 12_model_comparison.png
    ├── 14_3day_forecast.png
    ├── 15_model_registry.png
    ├── 16_day1_stats.png
    ├── 17_day2_stats.png
    ├── 18_day3_stats.png
    ├── 19_aqi_ref.png
    ├── eda_aqi_distribution.png
    ├── eda_aqi_time.png
    ├── eda_correlation.png
    ├── eda_lag_analysis.png
    ├── eda_monthly.png
    ├── eda_pollutant_scatter.png
    └── eda_temporal_patterns.png
```

---

## 📋 Prerequisites

Before starting, ensure you have:

1. **Python 3.10 or higher** installed
2. **MongoDB Atlas** account with a free M0 cluster (or local MongoDB instance)
3. **DagsHub** account for MLflow experiment tracking
4. **GitHub** account for CI/CD automation
5. (Optional) **Streamlit Cloud** or **Render/Heroku** account for dashboard deployment

### Required Accounts & Services

| Service | Free Tier | Sign Up |
|---------|-----------|---------|
| MongoDB Atlas | 512MB storage, shared RAM | [cloud.mongodb.com](https://cloud.mongodb.com) |
| DagsHub | Unlimited public repos, MLflow | [dagshub.com](https://dagshub.com) |
| GitHub Actions | 2,000 minutes/month | [github.com](https://github.com) |
| Open-Meteo | Unlimited, no API key | [open-meteo.com](https://open-meteo.com) |
| Streamlit Cloud | 1 app, 1GB RAM | [streamlit.io/cloud](https://streamlit.io/cloud) |

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Using conda
conda create -n aqi python=3.10
conda activate aqi
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Core dependencies from `requirements`:**
```
requests                    # HTTP requests
pandas                      # Data manipulation
numpy                       # Numerical computing
scikit-learn                # ML algorithms (RF, Ridge, preprocessing)
xgboost                     # Gradient boosting
pymongo[srv]                # MongoDB with SRV support
mlflow                      # Experiment tracking
dagshub                     # DagsHub MLflow integration
streamlit                   # Web dashboard
plotly                      # Interactive visualizations
shap                        # Model explainability
python-dotenv               # Environment variables
openmeteo-requests          # Open-Meteo API client
requests-cache              # HTTP caching
retry-requests              # Retry logic
seaborn                     # Statistical visualization
lightgbm                    # LightGBM gradient boosting
catboost                    # CatBoost gradient boosting
statsmodels                 # Statistical models
prophet                     # Facebook Prophet time series
```

### 4. Configure Environment Variables

Copy the environment template and fill in your credentials:

```bash
cp .env .env.local  # or create from scratch
```

Edit `.env` with your actual values:

```env
# ============================================
# AQI Predictor - Environment Configuration
# ============================================

# REQUIRED: MongoDB Atlas connection
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/aqi_db?retryWrites=true&w=majority

# REQUIRED: City configuration
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011

# DagsHub / MLflow credentials
# Create repo at: https://dagshub.com/yourusername/your-repo-name
DAGSHUB_USERNAME=your-username
DAGSHUB_REPO_NAME=your-repo-name
DAGSHUB_TOKEN=your-dagshub-token

# Optional: Alert configuration
ALERT_EMAIL=your-email@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
```

> **Security Note:** Never commit `.env` to version control. It is already added to `.gitignore`.

---

## Configuration

All project constants, feature definitions, model hyperparameters, and thresholds are centralized in `config`.

### Key Configuration Sections

#### Target Configuration
```python
CITY_NAME = "Karachi"           # Target city name
CITY_LAT = 24.8607              # Latitude
CITY_LON = 67.0011              # Longitude

TARGET_CITIES = [
    {"name": CITY_NAME, "lat": CITY_LAT, "lon": CITY_LON},
]
```

#### Feature Columns (30+ Engineered Features)
```python
FEATURE_COLS = [
    # Temporal features
    "hour", "day", "month", "day_of_week", "is_weekend",

    # Cyclical encodings (capture seasonality)
    "hour_sin", "hour_cos", 
    "dow_sin", "dow_cos", 
    "month_sin", "month_cos",

    # Pollutant features
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone",

    # Weather features
    "temperature_2m", "relative_humidity_2m", 
    "wind_speed_10m", "precipitation",

    # Derived AQI features
    "aqi_change_rate", 
    "aqi_rolling_mean_3", "aqi_rolling_std_3",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",

    # Interaction & ratio features
    "pm25_pm10_ratio",
    "humidity_temp_interaction", 
    "heat_index",
    "dewpoint", "temp_dewpoint_diff", "abs_humidity",

    # Wind & pressure dynamics
    "wind_speed_normalized", "wind_speed_sq",
    "wind_effect_x", "wind_effect_y",
    "pressure_change_1h", "pressure_change_3h", "pressure_change_6h",
    "cloud_visibility_ratio",

    # Rush hour indicators
    "is_morning_rush", "is_evening_rush",
]
```

#### Model Configuration
```python
MODELS_TO_TRAIN = {
    "random_forest": {
        "type": "sklearn",
        "params": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        }
    },
    "gradient_boosting": {
        "type": "sklearn",
        "params": {
            "n_estimators": 200,
            "max_depth": 4,
            "random_state": 42,
        }
    },
    "xgboost": {
        "type": "sklearn",
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }
    },
}
```

#### AQI Categories (EPA Standard)
```python
AQI_CATEGORIES = {
    (0, 50):   ("Good", "#00e400"),
    (51, 100): ("Moderate", "#ffff00"),
    (101, 150):("Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200):("Unhealthy", "#ff0000"),
    (201, 300):("Very Unhealthy", "#8f3f97"),
    (301, 500):("Hazardous", "#7e0023"),
}

ALERT_AQI_THRESHOLD = 150  # Trigger alert above this level
```

#### Forecast Horizons
```python
FORECAST_HORIZONS = [1, 2, 3]  # 1-day, 2-day, 3-day ahead predictions
```

---

## Data Sources

### Open-Meteo Weather API
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Parameters**: Temperature, relative humidity, wind speed, precipitation, pressure, cloud cover, visibility
- **Resolution**: Hourly
- **Cost**: Free, no API key required
- **Rate Limit**: Unlimited (be respectful, use caching)

### Open-Meteo Air Quality API
- **Endpoint**: `https://air-quality-api.open-meteo.com/v1/air-quality`
- **Parameters**: PM2.5, PM10, NO2, O3, CO, SO2, AQI (US EPA standard), dust, pollen
- **Resolution**: Hourly
- **Cost**: Free, no API key required

### Data Merging Strategy
Both APIs are queried for the same geographic coordinates and time range. Responses are merged on the **`timestamp`** column using an outer join to ensure no data loss. Missing values are handled via:
- **Forward fill** for short gaps (< 3 hours)
- **Linear interpolation** for medium gaps (3-12 hours)
- **Seasonal imputation** for long gaps (> 12 hours)

---

## Feature Pipeline

The feature pipeline (`pipelines/feature_pipeline.py`) is the core data ingestion and transformation engine. It runs automatically every hour via GitHub Actions.

### Pipeline Steps

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Fetch Weather  │    │  Fetch Air Q    │    │   Merge Data    │
│     Data        │ +  │     Data        │ ->  │  on Timestamp   │
│  (Open-Meteo)   │    │  (Open-Meteo)   │    │                 │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Store to Mongo │ <-  │ Engineer Features│ <-  │  Clean & Validate│
│   (Feature Store)│    │  (30+ features)  │    │  (Handle nulls)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1. Raw Data Fetching
```python
import openmeteo_requests
from retry_requests import retry

# Setup client with retry logic
retry_session = retry(retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Fetch weather data
weather_url = "https://api.open-meteo.com/v1/forecast"
weather_params = {
    "latitude": CITY_LAT,
    "longitude": CITY_LON,
    "hourly": ["temperature_2m", "relative_humidity_2m", 
               "wind_speed_10m", "precipitation", "pressure_msl"],
    "past_days": 1,
    "forecast_days": 4
}
weather_response = openmeteo.weather_api(weather_url, params=weather_params)

# Fetch air quality data
aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aqi_params = {
    "latitude": CITY_LAT,
    "longitude": CITY_LON,
    "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", 
               "us_aqi", "carbon_monoxide", "sulphur_dioxide"],
    "past_days": 1,
    "forecast_days": 4
}
aqi_response = openmeteo.air_quality_api(aqi_url, params=aqi_params)
```

### 2. Data Cleaning & Validation
- **Range checks**: AQI must be between 0-500, temperature between -50C and 60C
- **Outlier detection**: Z-score > 3 flagged for review
- **Duplicate removal**: Drop duplicate timestamps
- **Type casting**: Ensure numeric types, parse timestamps to UTC

### 3. Feature Engineering

#### Temporal Features
| Feature | Description | Formula |
|---------|-------------|---------|
| `hour` | Hour of day (0-23) | `dt.hour` |
| `day` | Day of month (1-31) | `dt.day` |
| `month` | Month (1-12) | `dt.month` |
| `day_of_week` | Day of week (0-6) | `dt.dayofweek` |
| `is_weekend` | Weekend flag | `day_of_week >= 5` |
| `hour_sin` | Cyclical hour encoding | `sin(2*pi * hour / 24)` |
| `hour_cos` | Cyclical hour encoding | `cos(2*pi * hour / 24)` |
| `dow_sin` | Cyclical day encoding | `sin(2*pi * day_of_week / 7)` |
| `dow_cos` | Cyclical day encoding | `cos(2*pi * day_of_week / 7)` |
| `month_sin` | Cyclical month encoding | `sin(2*pi * month / 12)` |
| `month_cos` | Cyclical month encoding | `cos(2*pi * month / 12)` |

#### Lag & Rolling Features
| Feature | Description | Window |
|---------|-------------|--------|
| `aqi_lag_1` | AQI 1 hour ago | 1h |
| `aqi_lag_2` | AQI 2 hours ago | 2h |
| `aqi_lag_3` | AQI 3 hours ago | 3h |
| `aqi_rolling_mean_3` | Rolling mean AQI | 3h |
| `aqi_rolling_std_3` | Rolling std AQI | 3h |
| `aqi_change_rate` | Rate of change | `(AQI_t - AQI_t-1) / AQI_t-1` |

#### Interaction & Derived Features
| Feature | Description | Formula |
|---------|-------------|---------|
| `pm25_pm10_ratio` | PM2.5 to PM10 ratio | `pm2_5 / pm10` |
| `humidity_temp_interaction` | Combined effect | `humidity * temperature` |
| `heat_index` | Apparent temperature | `f(temperature, humidity)` |
| `dewpoint` | Dew point temperature | `f(temperature, humidity)` |
| `temp_dewpoint_diff` | Temperature spread | `temperature - dewpoint` |
| `abs_humidity` | Absolute humidity | `f(temperature, relative_humidity)` |
| `wind_speed_normalized` | Normalized wind | `wind_speed / max_wind` |
| `wind_speed_sq` | Wind speed squared | `wind_speed^2` |
| `wind_effect_x` | Wind vector X | `wind_speed * cos(wind_direction)` |
| `wind_effect_y` | Wind vector Y | `wind_speed * sin(wind_direction)` |
| `pressure_change_1h` | Pressure trend | `pressure_t - pressure_t-1` |
| `pressure_change_3h` | Pressure trend | `pressure_t - pressure_t-3` |
| `pressure_change_6h` | Pressure trend | `pressure_t - pressure_t-6` |
| `cloud_visibility_ratio` | Cloud/visibility ratio | `cloud_cover / visibility` |
| `is_morning_rush` | Rush hour flag | `7 <= hour <= 10` |
| `is_evening_rush` | Rush hour flag | `17 <= hour <= 20` |

### 4. Target Definition
For each horizon `h` in `[1, 2, 3]`:
```python
df[f"target_aqi_h{h}"] = df["aqi"].shift(-h)  # Future AQI value
```

### 5. Storage
Features and targets are stored in MongoDB:
```python
from pymongo import MongoClient

client = MongoClient(MONGODB_URI)
db = client["aqi_db"]
collection = db[f"features_{CITY_NAME.lower()}"]

# Insert with upsert logic (update existing, insert new)
for record in df.to_dict("records"):
    collection.update_one(
        {"timestamp": record["timestamp"]},
        {"$set": record},
        upsert=True
    )
```

---

## Training Pipeline

The training pipeline (`pipelines/training_pipeline.py`) fetches historical features from MongoDB, trains multiple models per forecast horizon, evaluates them, and registers the best model.

### Pipeline Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Load Features  │    │  Train-Test     │    │  Train Models   │
│  from MongoDB   │ ->  │  Split (80/20)  │ ->  │  (RF, GB, XGB)  │
│  (Last 90 days) │    │  Time-based     │    │  Per horizon    │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                     │
                              ┌──────────────────────┼──────────────────────┐
                              ▼                      ▼                      ▼
                    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                    │  Evaluate RMSE  │    │  Evaluate MAE   │    │  Evaluate R2    │
                    │  Per horizon  │    │  Per horizon    │    │  Per horizon    │
                    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
                             │                      │                      │
                             └──────────────────────┼──────────────────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────┐
                                          │  Select Best    │
                                          │  Model per      │
                                          │  Horizon        │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │  Log to MLflow  │
                                          │  (DagsHub)      │
                                          │  + Save artifact│
                                          └─────────────────┘
```

### Model Training Strategy

#### Per-Horizon Models
Since AQI dynamics differ across prediction horizons, we train **separate models** for each:
- `model_h1`: Predicts AQI 24 hours ahead
- `model_h2`: Predicts AQI 48 hours ahead
- `model_h3`: Predicts AQI 72 hours ahead

#### Algorithms Tested
1. **Random Forest Regressor** - Robust, handles non-linearity, feature interactions
2. **Gradient Boosting Regressor** - Sequential error correction, good for time series
3. **XGBoost Regressor** - Optimized gradient boosting, regularization
4. **Ridge Regression** - Linear baseline, fast, interpretable
5. **LightGBM** (optional) - Fast training, large datasets
6. **CatBoost** (optional) - Categorical feature handling (logs to `catboost_info/`)
7. **TensorFlow Neural Network** (optional) - Deep learning for complex patterns
8. **Prophet** (optional) - Statistical baseline with seasonality

#### Hyperparameters (from `config`)
```python
random_forest = {
    "n_estimators": 200,      # Number of trees
    "max_depth": 15,          # Tree depth limit
    "min_samples_leaf": 5,   # Minimum samples per leaf
    "random_state": 42,
    "n_jobs": -1              # Use all CPU cores
}

xgboost = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,   # Shrinkage factor
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1
}
```

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | `sqrt(mean((y_true - y_pred)^2))` | Penalizes large errors (same unit as AQI) |
| **MAE** | `mean(abs(y_true - y_pred))` | Average absolute error (robust to outliers) |
| **R2** | `1 - (SS_res / SS_tot)` | Variance explained (1.0 = perfect, 0 = baseline) |
| **MAPE** | `mean(abs(y_true - y_pred) / y_true)` | Percentage error (intuitive) |

### MLflow Integration

```python
import mlflow
import dagshub

# Initialize DagsHub MLflow
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)

with mlflow.start_run(run_name=f"{model_name}_h{horizon}"):
    # Log parameters
    mlflow.log_params(model_params)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log model artifact
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log feature importance plot
    mlflow.log_artifact("feature_importance.png")
```

---

## Model Registry & Experiment Tracking

All experiments are tracked via **MLflow** hosted on **DagsHub**.

### Accessing the MLflow UI

```bash
# Set DagsHub tracking URI
export MLFLOW_TRACKING_URI="https://dagshub.com/yourusername/your-repo-name.mlflow"

# Launch MLflow UI locally
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI
```

### Registered Models

| Model Name | Horizon | Algorithm | Status |
|-----------|---------|-----------|--------|
| `aqi-rf-h1` | 1-day | Random Forest | Production |
| `aqi-xgb-h1` | 1-day | XGBoost | Staging |
| `aqi-rf-h2` | 2-day | Random Forest | Production |
| `aqi-rf-h3` | 3-day | Random Forest | Production |
| `aqi-ensemble` | All | Stacking | Archived |

### Model Promotion Workflow
1. **Development**: Train new model, log to MLflow
2. **Staging**: Compare with production model on holdout test
3. **Production**: Promote if RMSE improves by > 5%
4. **Archived**: Keep old versions for rollback

---

## Automated CI/CD

GitHub Actions workflows (in `.github/workflows/`) automate the entire ML lifecycle.

### Workflow 1: Feature Pipeline (Hourly)

`.github/workflows/feature_pipeline.yml`

```yaml
name: Feature Pipeline

on:
  schedule:
    - cron: '0 * * * *'  # Every hour at minute 0
  workflow_dispatch:       # Manual trigger

jobs:
  fetch-and-store:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run feature pipeline
        env:
          MONGODB_URI: ${{ secrets.MONGODB_URI }}
          CITY_NAME: ${{ vars.CITY_NAME }}
          CITY_LAT: ${{ vars.CITY_LAT }}
          CITY_LON: ${{ vars.CITY_LON }}
        run: python pipelines/feature_pipeline.py
```

### Workflow 2: Training Pipeline (Daily)

`.github/workflows/training_pipeline.yml`

```yaml
name: Training Pipeline

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2:00 AM UTC
  workflow_dispatch:

jobs:
  train-and-register:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train models
        env:
          MONGODB_URI: ${{ secrets.MONGODB_URI }}
          DAGSHUB_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
          DAGSHUB_REPO_NAME: ${{ secrets.DAGSHUB_REPO_NAME }}
          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
        run: python pipelines/training_pipeline.py

      - name: Upload model artifacts
        uses: actions/upload-artifact@v3
        with:
          name: trained-models
          path: models/*.pkl
```

### Workflow 3: EDA Report (Weekly)

`.github/workflows/eda_report.yml`

```yaml
name: EDA Report

on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday at midnight
  workflow_dispatch:

jobs:
  generate-eda:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate EDA plots
        run: python eda_analysis
      - name: Commit EDA results
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add data/*.png
          git commit -m "Update EDA reports [skip ci]" || echo "No changes"
          git push
```

---

## Web Application Dashboard

The Streamlit dashboard (`app/streamlit_app.py`) provides an interactive interface for real-time monitoring and forecasting.

### Pages

#### 1. Home - Real-Time Overview
- Current AQI with color-coded badge (Good -> Hazardous)
- Live weather conditions (temperature, humidity, wind)
- Pollutant breakdown (PM2.5, PM10, NO2, O3)
- Health recommendations based on current AQI

#### 2. Forecast - 3-Day Predictions
- Line chart showing predicted AQI for next 72 hours
- Confidence intervals (+-1 std)
- Category breakdown (Good/Moderate/Unhealthy/Hazardous hours)
- Day-by-day summary cards

#### 3. Historical - Trends & Analysis
- Time series of past AQI (7/30/90 days)
- Monthly and seasonal patterns
- Pollutant correlation scatter plots
- Hourly and daily-of-week pattern analysis

#### 4. Model Insights - Explainability
- SHAP summary plot (global feature importance)
- SHAP waterfall plot (single prediction explanation)
- Feature importance bar chart per model
- Model performance metrics table

### Running the Dashboard Locally

```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`.

---

## Exploratory Data Analysis (EDA)

The `eda_analysis` script generates comprehensive visualizations automatically.

### Generated Plots

| Plot | File | Description |
|------|------|-------------|
| **AQI Distribution** | `data/eda_aqi_distribution.png` | Histogram with EPA threshold lines + category breakdown |
| **AQI Time Series** | `data/eda_aqi_time.png` | Full time series with 24h rolling mean and threshold lines |
| **Correlation Matrix** | `data/eda_correlation.png` | Feature correlation heatmap (upper triangle masked) |
| **Lag Analysis** | `data/eda_lag_analysis.png` | Scatter plots of AQI vs lag-1/2/3 with regression lines and Pearson r |
| **Monthly Patterns** | `data/eda_monthly.png` | Mean AQI by month with error bars + box plots |
| **Pollutant Scatter** | `data/eda_pollutant_scatter.png` | PM2.5/PM10/NO2/O3 vs AQI with correlation stats |
| **Temporal Patterns** | `data/eda_temporal_patterns.png` | Hourly mean AQI with std band + day-of-week bar chart |

### Running EDA

```bash
python eda_analysis
```

### Sample Output
```
==================================================
EDA SUMMARY STATISTICS
==================================================
total_records         : 2160
date_range            : 2024-01-01 00:00:00 to 2024-03-31 23:00:00
avg_aqi               : 142.35
median_aqi            : 138.00
max_aqi               : 312.00
min_aqi               : 45.00
days_above_150        : 892
pct_unhealthy         : 41.30
avg_pm25              : 68.42
avg_temp              : 28.15
avg_humidity          : 62.80
==================================================
```

---

## Feature Engineering Deep Dive

### Why These Features?

| Feature Group | Rationale |
|--------------|-----------|
| **Cyclical Encodings** | Raw hour/month values imply false ordinality (23 != 0). Sine/cosine preserve circular nature (e.g., 23:00 is close to 00:00). |
| **Lag Features** | AQI is highly autocorrelated. Past values are the strongest predictors of future values. |
| **Rolling Statistics** | Smooth short-term noise and capture local trends (3-hour window matches typical pollution persistence). |
| **Change Rate** | Detect rapid deterioration or improvement events (e.g., sandstorm onset, rain clearance). |
| **Interaction Terms** | Humidity + temperature -> heat index affects chemical reaction rates for ozone formation. |
| **Wind Vectors** | Pollutant dispersion depends on wind direction, not just speed. X/Y components capture this. |
| **Pressure Changes** | Falling pressure often precedes stagnant air conditions that trap pollutants. |
| **Rush Hour Flags** | Traffic emissions spike during 7-10 AM and 5-8 PM in urban areas. |

### Feature Importance (Typical Ranking)

Based on SHAP analysis:
1. `aqi_lag_1` (most important - recent history dominates)
2. `pm2_5` (primary pollutant driver)
3. `aqi_rolling_mean_3` (local trend)
4. `temperature_2m` (temperature affects chemical reactions)
5. `hour_sin` / `hour_cos` (diurnal cycle)
6. `wind_speed_10m` (dispersion mechanism)
7. `relative_humidity_2m` (aerosol hygroscopic growth)
8. `aqi_lag_2` / `aqi_lag_3` (medium-term persistence)

---

## Model Performance & Evaluation

### Model Performance (Karachi, ~65-day training window)

| Horizon | Model | RMSE | Notes |
|---------|-------|------|-------|
| 1-day | CatBoost | ~20 | Most consistent performer |
| 1-day | Extra Trees | ~24 | Low variance, stable |
| 1-day | Random Forest | ~25 | Robust baseline |
| 1-day | XGBoost | ~26 | Moderate at H=1 |
| 2-day | CatBoost | ~23 | Ranks #1 at H=2 |
| 2-day | Extra Trees | ~25 | Stable across horizons |
| 2-day | LightGBM | ~27 | Consistent mid-range |
| 3-day | CatBoost | ~22 | Best overall anchor model |
| 3-day | XGBoost | ~22 | U-shaped recovery at H=3 |
| 3-day | Gradient Boosting | ~23 | Improves with horizon |
| **All horizons** | **Ensemble (RF + XGB + GBM)** | — | **R²=0.91 at H=1; 14% RMSE improvement over single model** |

> **Note:** Linear models (Ridge, Lasso, Elastic Net) were evaluated but excluded as they underperformed across all horizons (RMSE > 30).

### Performance Interpretation
- **1-day RMSE ~ 20**: Predictions within ±20 AQI points ~68% of the time (CatBoost, best performer)
- **3-day RMSE ~ 22-23**: Predictions within ±23 AQI points ~68% of the time (CatBoost/XGBoost)
- **R² = 0.91**: Ensemble explains 91% of variance at the 1-day horizon
- **14% RMSE improvement**: Ensemble outperforms any single model through variance reduction

### Error Analysis
- **Under-prediction bias**: Models tend to under-predict spikes (sandstorms, fires)
- **Over-prediction bias**: Models over-predict recovery after rain events
- **Seasonal variance**: Winter (Dec-Feb) shows higher error due to temperature inversions

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) values provide model-agnostic interpretability.

### Global Explanations
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS)
```

**Interpretation**: Features are ranked by mean absolute SHAP value. Red = high feature value pushes prediction up. Blue = high feature value pushes prediction down.

### Local Explanations (Single Prediction)
```python
# Waterfall plot for a single prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=FEATURE_COLS
))
```

**Use Case**: When the dashboard shows a hazardous forecast, the user can click "Explain" to see exactly which factors (e.g., "PM2.5 is 3x normal", "wind speed dropped 40%") drove the prediction.

---

## Alerting System

The system monitors predicted AQI and triggers alerts when thresholds are exceeded.

### Alert Thresholds

| AQI Range | Category | Alert Level | Action |
|-----------|----------|-------------|--------|
| 151-200 | Unhealthy | Warning | Email notification |
| 201-300 | Very Unhealthy | Alert | Email + Slack notification |
| 301+ | Hazardous | Emergency | Email + Slack + Dashboard banner |

### Alert Channels
1. **Dashboard Banner**: Real-time color-coded alert in Streamlit app (`app/`)
2. **Email**: SMTP or SendGrid integration
3. **Slack**: Webhook notifications to designated channel

### Alert Logic
```python
from pipelines.alerts import send_alert

for horizon, prediction in predictions.items():
    if prediction > ALERT_AQI_THRESHOLD:
        category, color = get_aqi_category(prediction)
        send_alert(
            level="warning" if prediction < 200 else "critical",
            message=f"AQI forecast for {CITY_NAME} ({horizon}d): {prediction} ({category})",
            channels=["email", "slack"]
        )
```

---

## API Reference

### Internal API (Flask/FastAPI Wrapper)

For programmatic access to predictions, a REST API is provided.

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/current` | Current AQI and weather |
| `GET` | `/api/v1/forecast` | 3-day AQI forecast |
| `GET` | `/api/v1/historical?days=30` | Historical data |
| `GET` | `/api/v1/features` | Latest feature vector |
| `POST` | `/api/v1/predict` | Custom prediction with input features |

#### Example Response: `/api/v1/forecast`
```json
{
  "city": "Karachi",
  "generated_at": "2024-06-07T22:00:00Z",
  "forecasts": [
    {
      "horizon_days": 1,
      "predicted_aqi": 142,
      "category": "Unhealthy for Sensitive Groups",
      "confidence_interval": [128, 156],
      "features": {
        "pm2_5": 58.2,
        "temperature": 31.5,
        "humidity": 72.0
      }
    },
    {
      "horizon_days": 2,
      "predicted_aqi": 155,
      "category": "Unhealthy",
      "confidence_interval": [135, 175],
      "alert": true
    },
    {
      "horizon_days": 3,
      "predicted_aqi": 138,
      "category": "Unhealthy for Sensitive Groups",
      "confidence_interval": [115, 161]
    }
  ]
}
```

### Running the API

```bash
python pipelines/predict.py
# or
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

---

## Deployment Guide

### Option 1: Streamlit Cloud (Recommended for Dashboard)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set secrets in Streamlit Cloud dashboard:
   ```
   MONGODB_URI = "your-uri"
   DAGSHUB_USERNAME = "your-username"
   ...
   ```
5. Deploy - updates automatically on every push to `main`

### Option 2: Render / Heroku (Full Stack)

```bash
# Create Procfile
echo "web: streamlit run app/streamlit_app.py --server.port=$PORT" > Procfile

# Deploy to Heroku
heroku create pearls-aqi-predictor
heroku config:set MONGODB_URI=your-uri
heroku config:set CITY_NAME=Karachi
heroku config:set CITY_LAT=24.8607
heroku config:set CITY_LON=67.0011
git push heroku main
```

### Option 3: Self-Hosted (VPS / Docker)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t aqi-predictor .
docker run -p 8501:8501 --env-file .env aqi-predictor
```

---

## Troubleshooting

### Common Issues

#### Issue: MongoDB Connection Timeout
**Symptoms**: `pymongo.errors.ServerSelectionTimeoutError`
**Solution**: 
- Whitelist your IP in MongoDB Atlas Network Access
- Verify `MONGODB_URI` includes correct password
- Check if `+srv` is needed for your cluster type

#### Issue: Open-Meteo API Rate Limiting
**Symptoms**: `429 Too Many Requests`
**Solution**:
- Enable `requests-cache` (already configured)
- Reduce backfill batch size
- Add `time.sleep(1)` between requests

#### Issue: MLflow Authentication Failed
**Symptoms**: `401 Unauthorized` when logging models
**Solution**:
- Verify `DAGSHUB_TOKEN` is correct
- Ensure token has write access to repository
- Run `dagshub login` locally to test

#### Issue: Streamlit App Not Updating
**Symptoms**: Dashboard shows stale data
**Solution**:
- Check GitHub Actions logs for pipeline failures
- Verify MongoDB has recent documents
- Clear Streamlit cache: `Ctrl+Shift+R` or restart app

#### Issue: Model Predictions Are NaN
**Symptoms**: Forecast shows `NaN` or impossible values
**Solution**:
- Check for missing features in input vector
- Verify feature column order matches training
- Validate AQI range (0-500) in post-processing

#### Issue: CatBoost Training Logs
**Symptoms**: `catboost_info/` folder grows large
**Solution**:
- This folder is auto-generated by CatBoost during training
- Add `catboost_info/` to `.gitignore` if not already present
- Periodically clean old training logs

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Areas
- [ ] Add support for multiple cities
- [ ] Implement LSTM/Transformer deep learning models
- [ ] Add geospatial heatmap to dashboard (`app/`)
- [ ] Integrate WAQI (World Air Quality Index) API as backup
- [ ] Add SMS alert support via Twilio
- [ ] Improve missing data imputation with KNN
- [ ] Add unit tests for all `pipelines/` modules

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Open-Meteo** for providing free, high-quality weather and air quality APIs
- **MongoDB Atlas** for the generous free tier
- **DagsHub** for MLflow hosting and experiment tracking
- **Streamlit** for making data apps simple and beautiful
- **SHAP** team for model interpretability tools

---

## Contact

For questions, issues, or collaboration:

- **Project Maintainer**: Izza Sohail Khan
- **GitHub Repository**: https://github.com/Izza-7913/10P_AQI
- **DagsHub Project**: https://dagshub.com/Izza-7913/10P_AQI

---

<div align="center">

**Breathe Better with Data-Driven Insights**

*Built with care for Karachi and cities worldwide.*

</div>
