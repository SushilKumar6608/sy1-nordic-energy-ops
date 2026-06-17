import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def load_and_prepare(target: str = "heating_demand_mw") -> dict:
    """
    Load energy_demand.csv and prepare features for all 3 models.

    target: "heating_demand_mw" or "cooling_demand_mw"

    Returns a dict keyed by city, each containing:
        train_df, val_df, test_df  — time-split DataFrames with features
    """
    assert target in ("heating_demand_mw", "cooling_demand_mw"), \
        "target must be heating_demand_mw or cooling_demand_mw"

    path = os.path.join(PROCESSED_DIR, "energy_demand.csv")
    df   = pd.read_csv(path, parse_dates=["timestamp"])
    df   = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    city_splits = {}

    for city in df["city"].unique():
        cdf = df[df["city"] == city].copy().reset_index(drop=True)

        # --- Cooling activity flag (for hurdle model classifier) ---
        # 1 = cooling system is active, 0 = off
        # Threshold 0.5 MW filters out noise in near-zero demand cities
        cdf["cooling_active"] = (cdf["cooling_demand_mw"] > 0.5).astype(int)

        # --- Time features ---
        cdf["hour"]      = cdf["timestamp"].dt.hour
        cdf["dayofweek"] = cdf["timestamp"].dt.dayofweek
        cdf["month"]     = cdf["timestamp"].dt.month
        cdf["dayofyear"] = cdf["timestamp"].dt.dayofyear

        # Cyclical encoding — avoids discontinuity at hour 23->0, month 12->1
        cdf["hour_sin"]  = np.sin(2 * np.pi * cdf["hour"] / 24)
        cdf["hour_cos"]  = np.cos(2 * np.pi * cdf["hour"] / 24)
        cdf["dow_sin"]   = np.sin(2 * np.pi * cdf["dayofweek"] / 7)
        cdf["dow_cos"]   = np.cos(2 * np.pi * cdf["dayofweek"] / 7)
        cdf["month_sin"] = np.sin(2 * np.pi * cdf["month"] / 12)
        cdf["month_cos"] = np.cos(2 * np.pi * cdf["month"] / 12)

        # --- Lag features (demand history) ---
        cdf["lag_1h"]   = cdf[target].shift(1)
        cdf["lag_6h"]   = cdf[target].shift(6)
        cdf["lag_24h"]  = cdf[target].shift(24)
        cdf["lag_168h"] = cdf[target].shift(168)  # same hour last week

        # --- Rolling statistics ---
        cdf["roll_mean_24h"] = cdf[target].shift(1).rolling(24).mean()
        cdf["roll_std_24h"]  = cdf[target].shift(1).rolling(24).std()
        cdf["roll_mean_7d"]  = cdf[target].shift(1).rolling(168).mean()

        # --- Drop rows with NaN from lag/rolling (first 168 rows per city) ---
        cdf = cdf.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        # --- Time-based train/val/test split ---
        # Train: 2018-2022, Val: 2023, Test: 2024
        train = cdf[cdf["timestamp"].dt.year <= 2022].copy()
        val   = cdf[cdf["timestamp"].dt.year == 2023].copy()
        test  = cdf[cdf["timestamp"].dt.year == 2024].copy()

        city_splits[city] = {
            "train": train,
            "val":   val,
            "test":  test,
        }

        print(f"{city}: train={len(train)} | val={len(val)} | test={len(test)}")

    return city_splits


# Core feature columns used by all three models (XGBoost, LSTM, TFT)
FEATURE_COLS = [
    "temperature", "wind_speed", "humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "lag_1h", "lag_6h", "lag_24h", "lag_168h",
    "roll_mean_24h", "roll_std_24h", "roll_mean_7d",
]

# Extended feature set for cooling hurdle model classifier
# Includes cooling_active binary flag as an additional input signal
COOLING_CLASSIFIER_FEATURES = FEATURE_COLS + ["cooling_active"]


if __name__ == "__main__":
    print("=== Feature Preparation ===")
    print("\n--- Heating demand ---")
    splits = load_and_prepare("heating_demand_mw")
    print("\n--- Cooling demand ---")
    splits = load_and_prepare("cooling_demand_mw")
    print("\nCore feature columns:", FEATURE_COLS)
    print("Total core features:", len(FEATURE_COLS))
    print("\nCooling classifier features:", COOLING_CLASSIFIER_FEATURES)
    print("Total cooling classifier features:", len(COOLING_CLASSIFIER_FEATURES))

    # Verify cooling_active distribution on test set
    print("\n--- Cooling active flag distribution (test 2024) ---")
    for city, split in splits.items():
        test      = split["test"]
        n_active  = int(test["cooling_active"].sum())
        n_total   = len(test)
        print(f"{city}: active={n_active} ({n_active/n_total*100:.1f}%) | "
              f"inactive={n_total - n_active} ({(n_total-n_active)/n_total*100:.1f}%)")