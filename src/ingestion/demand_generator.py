#demand_generator.py
import os
import numpy as np
import pandas as pd

RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Heating degree day base temperature (Swedish standard: 17 degrees C)
# Below this, heating demand increases linearly with temperature drop
HEATING_BASE_TEMP = 17.0

# Cooling degree day base temperature
# Above this, cooling demand increases linearly with temperature rise
COOLING_BASE_TEMP = 22.0

# Peak heating demand per city in MW (grounded in Swedish district heating statistics)
# Source basis: Energimarknadsinspektionen annual district heating volumes
CITY_PEAK_HEATING_MW = {
    "Stockholm":     800.0,
    "Gothenburg":    450.0,
    "Malmö":         300.0,
    "Uppsala":       150.0,
    "Umeå":          120.0,
    "Kiruna":        100.0,
    "Åre-Östersund":  80.0,
}

CITY_PEAK_COOLING_MW = {
    "Stockholm":     120.0,
    "Gothenburg":     80.0,
    "Malmö":          70.0,
    "Uppsala":        40.0,
    "Umeå":           25.0,
    "Kiruna":         10.0,
    "Åre-Östersund":  15.0,
}


def compute_heating_demand(temp: np.ndarray, wind: np.ndarray,
                           peak_mw: float, rng) -> np.ndarray:
    """
    Compute hourly heating demand in MW.

    Physics:
    - Base demand from heating degree hours (temp below 17 degrees C)
    - Wind chill amplifier: higher wind increases apparent heat loss
    - Diurnal pattern: demand peaks early morning (06:00) and evening (18:00)
    - Small Gaussian noise for sensor/measurement variability (2% std)
    """
    n = len(temp)

    # Heating degree hours — linear below base temperature
    hdh = np.maximum(HEATING_BASE_TEMP - temp, 0.0)

    # Normalise to 0-1 range based on theoretical max (base - coldest possible)
    max_hdh  = HEATING_BASE_TEMP - (-40.0)  # Kiruna can reach -40 degrees C
    hdh_norm = hdh / max_hdh

    # Wind chill amplifier: each m/s adds ~1.5% additional heat loss
    wind_factor = 1.0 + (wind * 0.015)

    # Diurnal pattern: buildings cool overnight, peak demand at morning warm-up
    hours   = np.arange(n) % 24
    diurnal = 1.0 + 0.08 * np.cos(2 * np.pi * (hours - 6) / 24)

    # Raw demand
    demand = peak_mw * hdh_norm * wind_factor * diurnal

    # Add small realistic noise (2% std — sensor and load variability)
    noise  = rng.normal(0, 0.02 * peak_mw, n)
    demand = np.maximum(demand + noise, 0.0)

    return demand


def compute_cooling_demand(temp: np.ndarray, humidity: np.ndarray,
                           peak_mw: float, rng) -> np.ndarray:
    """
    Compute hourly cooling demand in MW.

    Physics:
    - Base demand from cooling degree hours (temp above 22 degrees C)
    - Humidity amplifier: high humidity increases cooling load
    - Diurnal pattern: peaks early afternoon (14:00-16:00)
    - Small Gaussian noise (2% std)
    """
    n = len(temp)

    # Cooling degree hours
    cdh      = np.maximum(temp - COOLING_BASE_TEMP, 0.0)
    max_cdh  = 40.0 - COOLING_BASE_TEMP
    cdh_norm = cdh / max_cdh

    # Humidity amplifier: above 60% RH, each % adds 0.3% cooling load
    humidity_factor = 1.0 + np.maximum(humidity - 60.0, 0.0) * 0.003

    # Diurnal pattern: peaks at 15:00
    hours   = np.arange(n) % 24
    diurnal = 1.0 + 0.15 * np.cos(2 * np.pi * (hours - 15) / 24)

    demand = peak_mw * cdh_norm * humidity_factor * diurnal
    noise  = rng.normal(0, 0.02 * peak_mw, n)
    demand = np.maximum(demand + noise, 0.0)

    return demand


def inject_anomalies(demand: np.ndarray, temp: np.ndarray,
                     rng, anomaly_rate: float = 0.001,
                     is_cooling: bool = False) -> tuple:
    """
    Inject physically grounded anomalies into the demand signal.

    Heating anomaly types:
    1. Spike: pipe burst or sensor fault — demand jumps 40-80%
    2. Drop: supply failure — demand drops to near zero for 2-6 hours
    3. Drift: slow leak or degrading insulation — demand creeps up 10-20%

    Cooling fault signatures (physically grounded):
    A. Stuck valve: constant low-level draw during zero-demand periods
    B. Sensor drift: slow linear accumulation of background noise in winter
    C. Anti-cyclical pulse: brief spike during freezing conditions when
       cooling is physically impossible under normal operations
    """
    n      = len(demand)
    labels = np.zeros(n, dtype=int)
    demand = demand.copy()

    if is_cooling:
        # --- Cooling fault signatures ---

        # A. Stuck valve: constant draw during periods demand should be zero
        n_stuck = max(1, int(n * anomaly_rate * 0.5))
        for _ in range(n_stuck):
            start = int(rng.integers(0, n - 72))
            end   = min(start + int(rng.integers(12, 25)), n)
            # Only inject during periods where demand is near zero
            if demand[start:end].mean() < 0.5:
                stuck_level       = float(rng.uniform(0.05, 0.15)) * max(demand.max(), 1.0)
                demand[start:end] = stuck_level
                labels[start:end] = 1

        # B. Sensor drift: slow linear ramp during cold periods
        n_drift = max(1, int(n * anomaly_rate * 0.3))
        for _ in range(n_drift):
            start    = int(rng.integers(0, n - 336))
            end      = min(start + int(rng.integers(48, 97)), n)
            duration = end - start
            # Only inject during cold periods (temp below 5 degrees C)
            if temp[start:end].mean() < 5.0:
                alpha             = float(rng.uniform(0.001, 0.003)) * max(demand.max(), 1.0)
                ramp              = np.linspace(0, alpha * duration, duration)
                noise             = rng.normal(0, alpha * 0.1, duration)
                demand[start:end] += ramp + noise
                demand[start:end]  = np.maximum(demand[start:end], 0.0)
                labels[start:end]  = 1

        # C. Anti-cyclical pulse: brief spike during freezing conditions
        n_pulse = max(1, int(n * anomaly_rate * 0.5))
        for _ in range(n_pulse):
            start = int(rng.integers(0, n - 4))
            end   = min(start + int(rng.integers(1, 4)), n)
            # Only inject during freezing temperatures
            if temp[start:end].mean() < -5.0:
                pulse             = float(rng.uniform(0.3, 0.8)) * max(demand.max(), 1.0)
                demand[start:end] += pulse
                labels[start:end]  = 1

    else:
        # --- Heating anomaly types ---
        n_anomalies = max(1, int(n * anomaly_rate))
        for _ in range(n_anomalies):
            anomaly_type = rng.choice(["spike", "drop", "drift"])
            start        = int(rng.integers(0, n - 24))

            if anomaly_type == "spike":
                duration          = int(rng.integers(1, 4))
                end               = min(start + duration, n)
                factor            = float(rng.uniform(1.4, 1.8))
                demand[start:end] *= factor
                labels[start:end]  = 1

            elif anomaly_type == "drop":
                duration          = int(rng.integers(2, 7))
                end               = min(start + duration, n)
                demand[start:end] *= float(rng.uniform(0.05, 0.2))
                labels[start:end]  = 1

            elif anomaly_type == "drift":
                duration          = int(rng.integers(12, 48))
                end               = min(start + duration, n)
                drift             = np.linspace(1.0, float(rng.uniform(1.1, 1.2)), end - start)
                demand[start:end] *= drift
                labels[start:end]  = 1

    return demand, labels


def generate_demand() -> pd.DataFrame:
    """
    Generate hourly heating and cooling demand for all cities.
    Correlated with real SMHI weather data.
    Saves to data/processed/energy_demand.csv
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    weather_path = os.path.join(RAW_DIR, "smhi_weather_raw.csv")
    weather      = pd.read_csv(weather_path, parse_dates=["timestamp"])
    print(f"Loaded weather data: {len(weather)} records")

    all_frames = []

    for city in weather["city"].unique():
        print(f"\nGenerating demand for {city}...")
        city_weather = weather[weather["city"] == city].copy()
        city_weather = city_weather.sort_values("timestamp").reset_index(drop=True)

        # Fill small gaps in wind/humidity with forward fill
        city_weather["wind_speed"] = city_weather["wind_speed"].ffill().fillna(2.0)
        city_weather["humidity"]   = city_weather["humidity"].ffill().fillna(70.0)

        temp     = city_weather["temperature"].values
        wind     = city_weather["wind_speed"].values
        humidity = city_weather["humidity"].values

        rng = np.random.default_rng(seed=hash(city) % (2**32))

        peak_heat = CITY_PEAK_HEATING_MW.get(city, 100.0)
        peak_cool = CITY_PEAK_COOLING_MW.get(city, 20.0)

        heating = compute_heating_demand(temp, wind, peak_heat, rng)
        cooling = compute_cooling_demand(temp, humidity, peak_cool, rng)

        heating, heat_labels = inject_anomalies(
            heating, temp, rng, anomaly_rate=0.001, is_cooling=False)
        cooling, cool_labels = inject_anomalies(
            cooling, temp, rng, anomaly_rate=0.001, is_cooling=True)

        city_df = city_weather[["timestamp", "city", "temperature",
                                 "wind_speed", "humidity"]].copy()
        city_df["heating_demand_mw"] = heating
        city_df["cooling_demand_mw"] = cooling
        city_df["anomaly_heating"]   = heat_labels
        city_df["anomaly_cooling"]   = cool_labels

        n_heat_anom = heat_labels.sum()
        n_cool_anom = cool_labels.sum()
        print(f"  Records: {len(city_df)} | "
              f"Heating anomalies: {n_heat_anom} ({n_heat_anom/len(city_df)*100:.2f}%) | "
              f"Cooling anomalies: {n_cool_anom} ({n_cool_anom/len(city_df)*100:.2f}%)")

        all_frames.append(city_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined  = combined.sort_values(["city", "timestamp"]).reset_index(drop=True)
    # --- Physical realism mask ---
    # Force cooling demand to absolute zero when ambient temperature cannot
    # physically support cooling operations (below 12 degrees C), unless an
    # explicit synthetic anomaly flag is set for that hour.
    # This mirrors how a real Nordic grid operator like atNorth would record data.
    combined.loc[
        (combined["temperature"] < 12.0) & (combined["anomaly_cooling"] == 0),
        "cooling_demand_mw"
    ] = 0.0

    # Clip both signals to remove any negative regression noise
    combined["heating_demand_mw"] = combined["heating_demand_mw"].clip(lower=0.0)
    combined["cooling_demand_mw"] = combined["cooling_demand_mw"].clip(lower=0.0)

    output_path = os.path.join(PROCESSED_DIR, "energy_demand.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Total records: {len(combined)}")
    print(f"\nSample:")
    print(combined.head(5).to_string())

    return combined


if __name__ == "__main__":
    print("=== Energy Demand Generator ===")
    generate_demand()