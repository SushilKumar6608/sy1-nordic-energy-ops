import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forecasting"))
from prepare_features import load_and_prepare, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Adaptive thresholds — expressed as fraction of each city's peak demand
# Heating: 10% of peak (tighter — heating signal is strong and forecast is accurate)
# Cooling: 20% of peak (looser — cooling signal is sparse and noisy)
# Minimum floor: 5% of city peak to avoid fixed MW floors breaking small cities
DEVIATION_THRESHOLD_FRAC_HEATING = 0.10
DEVIATION_THRESHOLD_FRAC_COOLING = 0.20
MIN_ABSOLUTE_DEVIATION_FRAC = 0.05


def load_xgboost_model(city: str, target: str):
    """Load saved XGBoost model and scaler for a city and target."""
    city_key = (
        city.replace(" ", "_")
        .replace("-", "_")
        .replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
    )
    target_key = "heat" if "heating" in target else "cool"

    model_path = os.path.join(MODELS_DIR, f"xgb_{city_key}_{target_key}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"xgb_scaler_{city_key}_{target_key}.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def detect_anomalies_for_target(target: str) -> pd.DataFrame:
    """
    Run deviation detection for all cities on the test set (2024).

    Logic:
    1. Load XGBoost model per city
    2. Generate forecasts on test set
    3. Compute deviation = actual - forecast
    4. Flag as anomaly if abs(deviation) exceeds adaptive threshold
       Threshold = max(MIN_ABSOLUTE_DEVIATION_FRAC, target_frac) * peak_demand
       This avoids fixed MW floors that break small cities (e.g. Kiruna cooling)
    5. Evaluate detected anomalies against ground truth labels from demand generator

    Returns a DataFrame with per-city metrics: precision, recall, F1, TP, FP, FN, TN.
    """
    print(f"\n{'='*50}")
    print(f"Anomaly Detection -- target: {target}")
    print(f"{'='*50}")

    city_splits = load_and_prepare(target)
    all_results = []

    for city, splits in city_splits.items():
        print(f"\nProcessing {city}...")

        test = splits["test"]

        X_test = test[FEATURE_COLS].values
        y_true = test[target].values

        # Ground truth anomaly labels from demand generator
        anomaly_col = "anomaly_heating" if "heating" in target else "anomaly_cooling"
        y_true_labels = (
            test[anomaly_col].values if anomaly_col in test.columns else None
        )

        model, scaler = load_xgboost_model(city, target)
        X_test_scaled = scaler.transform(X_test)
        y_forecast = np.maximum(model.predict(X_test_scaled), 0.0)

        # Compute deviation
        deviation = y_true - y_forecast
        abs_deviation = np.abs(deviation)

        # Peak demand — use 99th percentile of actual demand
        peak_demand = float(np.percentile(y_true, 99))
        if peak_demand < 0.1:
            peak_demand = 0.1  # safety floor for near-zero cities

        # Adaptive threshold: fraction of city peak demand
        # Minimum is 5% of peak to avoid fixed floor breaking small cities
        if "heating" in target:
            frac = DEVIATION_THRESHOLD_FRAC_HEATING
        else:
            frac = DEVIATION_THRESHOLD_FRAC_COOLING

        threshold_mw = max(
            MIN_ABSOLUTE_DEVIATION_FRAC * peak_demand, frac * peak_demand
        )

        detected = (abs_deviation > threshold_mw).astype(int)

        # Evaluation against ground truth
        if y_true_labels is not None:
            tp = int(((detected == 1) & (y_true_labels == 1)).sum())
            fp = int(((detected == 1) & (y_true_labels == 0)).sum())
            fn = int(((detected == 0) & (y_true_labels == 1)).sum())
            tn = int(((detected == 0) & (y_true_labels == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        else:
            tp = fp = fn = tn = 0
            precision = recall = f1 = 0.0

        n_detected = int(detected.sum())
        n_true_anom = int(y_true_labels.sum()) if y_true_labels is not None else 0

        print(f"  Peak demand (99th pct): {peak_demand:.1f} MW")
        print(f"  Threshold: {threshold_mw:.2f} MW ({frac*100:.0f}% of peak)")
        print(f"  Detected: {n_detected} | True anomalies: {n_true_anom}")
        print(f"  Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}")
        print(f"  TP={tp} | FP={fp} | FN={fn} | TN={tn}")

        all_results.append(
            {
                "city": city,
                "target": target,
                "peak_demand_mw": round(peak_demand, 2),
                "threshold_mw": round(threshold_mw, 2),
                "threshold_frac": frac,
                "n_detected": n_detected,
                "n_true_anom": n_true_anom,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    return pd.DataFrame(all_results)


def run_detection() -> pd.DataFrame:
    """Run anomaly detection for both heating and cooling, save results."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    results_heat = detect_anomalies_for_target("heating_demand_mw")
    results_cool = detect_anomalies_for_target("cooling_demand_mw")

    combined = pd.concat([results_heat, results_cool], ignore_index=True)

    output_path = os.path.join(PROCESSED_DIR, "anomaly_detection_results.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print("\nFull results:")
    print(combined.to_string(index=False))

    return combined


if __name__ == "__main__":
    print("=== Deviation-Based Anomaly Detector ===")
    print(
        f"Heating threshold: {DEVIATION_THRESHOLD_FRAC_HEATING*100:.0f}% of peak demand"
    )
    print(
        f"Cooling threshold: {DEVIATION_THRESHOLD_FRAC_COOLING*100:.0f}% of peak demand"
    )
    print(f"Minimum floor: {MIN_ABSOLUTE_DEVIATION_FRAC*100:.0f}% of city peak demand")
    run_detection()
