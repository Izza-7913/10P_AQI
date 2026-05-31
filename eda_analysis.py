"""
EDA (Exploratory Data Analysis) for AQI Predictor
Merged approach: Dual API data, rich visualizations
Usage: python eda_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
CITY      = os.getenv("CITY_NAME", "Karachi")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)


def load_data():
    """Load data from MongoDB city-specific collection."""
    client = MongoClient(MONGO_URI)
    col    = client["aqi_db"][f"features_{CITY.lower()}"]
    docs   = list(col.find({}, {"_id": 0}))
    client.close()
    if not docs:
        logger.error("No data found in MongoDB")
        return None
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_aqi_distribution(df):
    """AQI distribution histogram + category breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with thresholds
    axes[0].hist(df["aqi"].dropna(), bins=50, color="#4f8ef7", alpha=0.7, edgecolor="white")
    axes[0].axvline(50, color="green", linestyle="--", label="Good")
    axes[0].axvline(100, color="yellow", linestyle="--", label="Moderate")
    axes[0].axvline(150, color="orange", linestyle="--", label="Unhealthy")
    axes[0].axvline(200, color="red", linestyle="--", label="Very Unhealthy")
    axes[0].set_title("AQI Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("AQI")
    axes[0].set_ylabel("Hours")
    axes[0].legend()

    # Category bar chart
    def cat_label(aqi):
        if aqi <= 50: return "Good"
        if aqi <= 100: return "Moderate"
        if aqi <= 150: return "Unhealthy\n(Sensitive)"
        if aqi <= 200: return "Unhealthy"
        if aqi <= 300: return "Very\nUnhealthy"
        return "Hazardous"

    cats = df["aqi"].apply(cat_label).value_counts()
    cat_order = ["Good", "Moderate", "Unhealthy\n(Sensitive)", "Unhealthy", "Very\nUnhealthy", "Hazardous"]
    cats = cats.reindex([c for c in cat_order if c in cats.index], fill_value=0)
    colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
    bars = axes[1].bar(cats.index, cats.values, color=colors[:len(cats)])
    axes[1].set_title("Hours per AQI Category", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Hours")
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f"{int(height)}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("eda_aqi_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eda_aqi_distribution.png")


def plot_aqi_over_time(df):
    """AQI time series with rolling mean and thresholds."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df["timestamp"], df["aqi"], color="#4f8ef7", alpha=0.6, linewidth=0.8, label="AQI")

    # 24h rolling mean
    df["aqi_rolling_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
    ax.plot(df["timestamp"], df["aqi_rolling_24h"], color="#ef4444", linewidth=2, label="24h rolling mean")

    ax.axhline(100, color="#cccc00", linestyle="--", alpha=0.7, label="Moderate")
    ax.axhline(150, color="#ff7e00", linestyle="--", alpha=0.7, label="Unhealthy")

    ax.set_title("AQI Over Time — " + CITY, fontsize=14, fontweight="bold")
    ax.set_ylabel("AQI")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("eda_aqi_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eda_aqi_time.png")


def plot_correlation_matrix(df):
    """Feature correlation matrix."""
    cols = ["aqi", "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
            "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
            "aqi_rolling_mean_3", "aqi_change_rate", "hour", "month"]
    cols = [c for c in cols if c in df.columns]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eda_correlation.png")


def plot_lag_analysis(df):
    """Lag features vs current AQI."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, lag in enumerate([1, 2, 3]):
        col = f"aqi_lag_{lag}"
        if col in df.columns:
            sample = df[["aqi", col]].dropna().sample(min(500, len(df)), random_state=42)
            axes[i].scatter(sample[col], sample["aqi"], alpha=0.3, color="#4f8ef7", s=10)
            # Regression line
            z = np.polyfit(sample[col], sample["aqi"], 1)
            p = np.poly1d(z)
            axes[i].plot(sample[col].sort_values(), p(sample[col].sort_values()),
                        color="#ef4444", linewidth=2)
            # Perfect prediction line
            mx = max(sample[col].max(), sample["aqi"].max())
            axes[i].plot([0, mx], [0, mx], "k--", alpha=0.3, linewidth=1)
            r = sample[col].corr(sample["aqi"])
            axes[i].set_title(f"r = {r:.3f}", fontsize=12)
            axes[i].set_xlabel(f"aqi_lag_{lag}")
            axes[i].set_ylabel("Current AQI")

    fig.suptitle("Lag Features vs Current AQI  (dashed = perfect prediction)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_lag_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eda_lag_analysis.png")


def plot_monthly_patterns(df):
    """Monthly AQI patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    monthly = df.groupby(df["timestamp"].dt.month)["aqi"].agg(["mean", "std"])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x_labels = [month_names[i-1] for i in monthly.index]

    axes[0].bar(range(len(monthly)), monthly["mean"], yerr=monthly["std"],
                color="#4f8ef7", capsize=5, edgecolor="white")
    axes[0].set_xticks(range(len(monthly)))
    axes[0].set_xticklabels(x_labels)
    axes[0].set_title("Mean AQI by Month", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Mean AQI")

    # Box plot
    df["month_name"] = df["timestamp"].dt.month.map(lambda x: month_names[x-1])
    month_order = [m for m in month_names if m in df["month_name"].unique()]
    sns.boxplot(data=df, x="month_name", y="aqi", order=month_order,
                ax=axes[1], color="#4f8ef7")
    axes[1].set_title("AQI Distribution by Month", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig("eda_monthly.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eda_monthly.png")


def plot_pollutant_scatter(df):
    """Pollutant vs AQI scatter plots."""
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]
    available = [p for p in pollutants if p in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for i, pol in enumerate(available):
        sample = df[["aqi", pol]].dropna().sample(min(500, len(df)), random_state=42)
        axes[i].scatter(sample[pol], sample["aqi"], alpha=0.3, s=10)
        z = np.polyfit(sample[pol], sample["aqi"], 1)
        p = np.poly1d(z)
        axes[i].plot(sample[pol].sort_values(), p(sample[pol].sort_values()),
                    color="black", linewidth=2)
        r = sample[pol].corr(sample["aqi"])
        from scipy import stats
        _, pval = stats.pearsonr(sample[pol], sample["aqi"])
        axes[i].set_title(f"{pol}\nr={r:.2f}, p={pval:.3f}", fontsize=11)
        axes[i].set_xlabel(pol)
        axes[i].set_ylabel("AQI")

    fig.suptitle("Pollutant vs AQI Scatter (sample of 500)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_pollutant_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eda_pollutant_scatter.png")


def plot_temporal_patterns(df):
    """Hourly and daily patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hourly
    hourly = df.groupby(df["timestamp"].dt.hour)["aqi"].agg(["mean", "std"])
    axes[0].plot(hourly.index, hourly["mean"], marker="o", color="#4f8ef7", linewidth=2)
    axes[0].fill_between(hourly.index,
                         hourly["mean"] - hourly["std"],
                         hourly["mean"] + hourly["std"],
                         alpha=0.2, color="#4f8ef7")
    axes[0].set_title("Mean AQI by Hour of Day", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Mean AQI")
    axes[0].set_xticks(range(0, 24, 2))

    # Day of week
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df.groupby(df["timestamp"].dt.dayofweek)["aqi"].agg(["mean", "std"])
    colors = ["#4f8ef7" if i < 5 else "#ef4444" for i in dow.index]
    axes[1].bar(dow.index, dow["mean"], yerr=dow["std"],
                color=colors, capsize=5, edgecolor="white")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_names)
    axes[1].set_title("Mean AQI by Day of Week", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Mean AQI")

    plt.tight_layout()
    plt.savefig("eda_temporal_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eda_temporal_patterns.png")


def generate_summary_stats(df):
    """Generate summary statistics."""
    stats = {
        "total_records": len(df),
        "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        "avg_aqi": df["aqi"].mean(),
        "median_aqi": df["aqi"].median(),
        "max_aqi": df["aqi"].max(),
        "min_aqi": df["aqi"].min(),
        "days_above_150": (df["aqi"] > 150).sum(),
        "pct_unhealthy": (df["aqi"] > 150).mean() * 100,
        "avg_pm25": df["pm2_5"].mean(),
        "avg_temp": df["temperature_2m"].mean(),
        "avg_humidity": df["relative_humidity_2m"].mean(),
    }

    print("\n" + "="*50)
    print("EDA SUMMARY STATISTICS")
    print("="*50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.2f}")
        else:
            print(f"{key:20s}: {value}")
    print("="*50)
    return stats


def main():
    logger.info("Starting EDA analysis...")
    df = load_data()
    if df is None:
        return

    logger.info(f"Loaded {len(df)} records")

    plot_aqi_distribution(df)
    plot_aqi_over_time(df)
    plot_correlation_matrix(df)
    plot_lag_analysis(df)
    plot_monthly_patterns(df)
    plot_pollutant_scatter(df)
    plot_temporal_patterns(df)
    stats = generate_summary_stats(df)

    logger.info("EDA complete! Check the generated PNG files.")


if __name__ == "__main__":
    main()
