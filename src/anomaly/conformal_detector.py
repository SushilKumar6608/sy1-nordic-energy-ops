import os
import pickle
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forecasting"))
from prepare_features import load_and_prepare, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# =============================================================================
# STRATEGY SUMMARY
# =============================================================================
# Heating: Original conformal prediction strategy
#   - Raw absolute residual non-conformity scores (no sigma normalization)
#   - Per-month quantile thresholds at alpha=0.05
#   - No persistence filter (heating anomalies last only 1-3h)
#
# Cooling: Latest hybrid adaptive strategy
#   - Active months (mean demand > 0.5 MW): normalized scores S_t = |e_t|/sigma_t
#   - Zero-demand months: raw absolute residual thresholds
#   - Per-month quantile thresholds at alpha=0.01
#   - 3h temporal persistence filter (fault signatures persist >= 3h)
# =============================================================================

ALPHA_HEATING = 0.05
MIN_CONSECUTIVE_HEATING = 1

ALPHA_COOLING = 0.01
MIN_CONSECUTIVE_COOLING = 3
SIGMA_FLOOR_COOLING = 0.1
SIGMA_WINDOW_COOLING = 24
ZERO_DEMAND_THRESHOLD_MW = 0.5


def load_xgboost_model(city: str, target: str):
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


# =============================================================================
# HEATING — original strategy functions
# =============================================================================


def fit_heating_thresholds(
    cal_residuals: np.ndarray, cal_months: np.ndarray, alpha: float
) -> dict:
    """
    Heating: per-month quantile thresholds on raw absolute residuals.
    No sigma normalization — keeps recall high for short-duration anomalies.
    """
    cal_scores = np.abs(cal_residuals)
    thresholds = {}
    global_level = min(1.0, (1 - alpha) * (1 + 1 / len(cal_scores)))
    global_thresh = float(np.quantile(cal_scores, global_level))

    for month in range(1, 13):
        mask = cal_months == month
        if mask.sum() < 30:
            thresholds[month] = global_thresh
        else:
            n = mask.sum()
            level = min(1.0, (1 - alpha) * (1 + 1 / n))
            thresholds[month] = float(np.quantile(cal_scores[mask], level))

    return thresholds


def detect_heating(
    city: str, cal: pd.DataFrame, test: pd.DataFrame, model, scaler
) -> tuple:
    """
    Heating anomaly detection using original conformal strategy.
    Returns (detected array, metrics dict).
    """
    target = "heating_demand_mw"
    anomaly_col = "anomaly_heating"

    y_cal = cal[target].values
    y_test = test[target].values
    y_labels = test[anomaly_col].values

    X_cal = scaler.transform(cal[FEATURE_COLS].values)
    y_cal_pred = np.maximum(model.predict(X_cal), 0.0)
    cal_residuals = y_cal - y_cal_pred
    cal_months = cal["timestamp"].dt.month.values

    thresholds = fit_heating_thresholds(cal_residuals, cal_months, ALPHA_HEATING)

    X_test = scaler.transform(test[FEATURE_COLS].values)
    y_test_pred = np.maximum(model.predict(X_test), 0.0)
    test_residuals = y_test - y_test_pred
    test_scores = np.abs(test_residuals)
    test_months = test["timestamp"].dt.month.values

    detected = np.zeros(len(test), dtype=int)
    for month, thresh in thresholds.items():
        mask = test_months == month
        detected[mask] = (test_scores[mask] > thresh).astype(int)

    metrics = compute_metrics(detected, y_labels)
    thresh_min = min(thresholds.values())
    thresh_max = max(thresholds.values())

    print(f"  Strategy: original | alpha={ALPHA_HEATING} | no persistence filter")
    print(f"  Thresholds: {thresh_min:.2f}-{thresh_max:.2f} MW")
    print(f"  Detected: {detected.sum()} | True anomalies: {y_labels.sum()}")
    print(
        f"  Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f} | "
        f"F1={metrics['f1']:.3f} | FPR={metrics['fpr']:.4f}"
    )
    print(
        f"  TP={metrics['tp']} | FP={metrics['fp']} | "
        f"FN={metrics['fn']} | TN={metrics['tn']}"
    )

    return detected, metrics


# =============================================================================
# COOLING — hybrid adaptive strategy functions
# =============================================================================


def compute_difficulty_estimator(
    residuals: np.ndarray,
    window: int = SIGMA_WINDOW_COOLING,
    floor: float = SIGMA_FLOOR_COOLING,
) -> np.ndarray:
    """Rolling std of absolute residuals as difficulty estimator sigma_t."""
    abs_res = np.abs(residuals)
    sigma = (
        pd.Series(abs_res)
        .rolling(window=window, min_periods=1)
        .std()
        .fillna(floor)
        .values
    )
    return np.maximum(sigma, floor)


def fit_cooling_thresholds(
    cal_residuals: np.ndarray,
    cal_months: np.ndarray,
    cal_sigma: np.ndarray,
    cal_demand: np.ndarray,
    alpha: float,
) -> dict:
    """
    Cooling: hybrid per-month thresholds.
    Active months (mean demand > 0.5 MW): normalized scores S_t = |e_t|/sigma_t
    Zero-demand months: raw absolute residual thresholds
    """
    cal_scores_norm = np.abs(cal_residuals) / cal_sigma
    cal_scores_raw = np.abs(cal_residuals)

    global_level = min(1.0, (1 - alpha) * (1 + 1 / len(cal_residuals)))
    global_norm = float(np.quantile(cal_scores_norm, global_level))
    global_raw = float(np.quantile(cal_scores_raw, global_level))

    thresholds = {}
    for month in range(1, 13):
        mask = cal_months == month
        mean_demand = cal_demand[mask].mean() if mask.sum() > 0 else 0.0
        is_zero = mean_demand <= ZERO_DEMAND_THRESHOLD_MW

        if mask.sum() < 30:
            thresholds[month] = (
                ("raw", global_raw) if is_zero else ("normalized", global_norm)
            )
        else:
            n = mask.sum()
            level = min(1.0, (1 - alpha) * (1 + 1 / n))
            if is_zero:
                thresholds[month] = (
                    "raw",
                    float(np.quantile(cal_scores_raw[mask], level)),
                )
            else:
                thresholds[month] = (
                    "normalized",
                    float(np.quantile(cal_scores_norm[mask], level)),
                )

    return thresholds


def apply_persistence_filter(
    raw_detected: np.ndarray, min_consecutive: int
) -> np.ndarray:
    """Flag anomaly only if breached for >= min_consecutive consecutive hours."""
    if min_consecutive <= 1:
        return raw_detected.copy()

    filtered = np.zeros_like(raw_detected)
    n = len(raw_detected)
    i = 0
    while i < n:
        if raw_detected[i] == 1:
            j = i
            while j < n and raw_detected[j] == 1:
                j += 1
            if (j - i) >= min_consecutive:
                filtered[i:j] = 1
            i = j
        else:
            i += 1
    return filtered


def detect_cooling(
    city: str, cal: pd.DataFrame, test: pd.DataFrame, model, scaler
) -> tuple:
    """
    Cooling anomaly detection using hybrid adaptive conformal strategy.
    Returns (detected array, metrics dict).
    """
    target = "cooling_demand_mw"
    anomaly_col = "anomaly_cooling"

    y_cal = cal[target].values
    y_test = test[target].values
    y_labels = test[anomaly_col].values

    X_cal = scaler.transform(cal[FEATURE_COLS].values)
    y_cal_pred = np.maximum(model.predict(X_cal), 0.0)
    cal_residuals = y_cal - y_cal_pred
    cal_sigma = compute_difficulty_estimator(cal_residuals)
    cal_months = cal["timestamp"].dt.month.values

    thresholds = fit_cooling_thresholds(
        cal_residuals, cal_months, cal_sigma, y_cal, ALPHA_COOLING
    )

    X_test = scaler.transform(test[FEATURE_COLS].values)
    y_test_pred = np.maximum(model.predict(X_test), 0.0)
    test_residuals = y_test - y_test_pred
    test_sigma = compute_difficulty_estimator(test_residuals)
    test_scores_norm = np.abs(test_residuals) / test_sigma
    test_scores_raw = np.abs(test_residuals)
    test_months = test["timestamp"].dt.month.values

    raw_detected = np.zeros(len(test), dtype=int)
    for month, (mode, thresh) in thresholds.items():
        mask = test_months == month
        if mode == "normalized":
            raw_detected[mask] = (test_scores_norm[mask] > thresh).astype(int)
        else:
            raw_detected[mask] = (test_scores_raw[mask] > thresh).astype(int)

    detected = apply_persistence_filter(raw_detected, MIN_CONSECUTIVE_COOLING)

    n_active = sum(1 for _, (mode, _) in thresholds.items() if mode == "normalized")
    metrics = compute_metrics(detected, y_labels)

    print(
        f"  Strategy: hybrid | alpha={ALPHA_COOLING} | "
        f"persistence>={MIN_CONSECUTIVE_COOLING}h"
    )
    print(f"  Active months: {n_active} | Zero-demand months: {12 - n_active}")
    print(
        f"  Raw detections: {raw_detected.sum()} | "
        f"After filter: {detected.sum()} | True anomalies: {y_labels.sum()}"
    )
    print(
        f"  Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f} | "
        f"F1={metrics['f1']:.3f} | FPR={metrics['fpr']:.4f}"
    )
    print(
        f"  TP={metrics['tp']} | FP={metrics['fp']} | "
        f"FN={metrics['fn']} | TN={metrics['tn']}"
    )

    return detected, metrics


# =============================================================================
# SHARED UTILITIES
# =============================================================================


def compute_metrics(detected: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(((detected == 1) & (labels == 1)).sum())
    fp = int(((detected == 1) & (labels == 0)).sum())
    fn = int(((detected == 0) & (labels == 1)).sum())
    tn = int(((detected == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================


def run_conformal_detection() -> pd.DataFrame:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    city_splits_heat = load_and_prepare("heating_demand_mw")
    city_splits_cool = load_and_prepare("cooling_demand_mw")

    all_results = []
    cities = list(city_splits_heat.keys())
    print(f"\nRunning detection for {len(cities)} cities...")

    for city in cities:
        print(f"\n{'='*50}")
        print(f"City: {city}")
        print(f"{'='*50}")

        # Heating
        print("\n--- Heating ---")
        splits_h = city_splits_heat[city]
        model_h, scaler_h = load_xgboost_model(city, "heating_demand_mw")
        detected_h, metrics_h = detect_heating(
            city, splits_h["val"], splits_h["test"], model_h, scaler_h
        )

        all_results.append(
            {
                "city": city,
                "target": "heating_demand_mw",
                "strategy": "original_conformal",
                "alpha": ALPHA_HEATING,
                "n_detected": int(detected_h.sum()),
                "n_true_anom": int(splits_h["test"]["anomaly_heating"].sum()),
                **metrics_h,
            }
        )

        # Cooling
        print("\n--- Cooling ---")
        splits_c = city_splits_cool[city]
        model_c, scaler_c = load_xgboost_model(city, "cooling_demand_mw")
        detected_c, metrics_c = detect_cooling(
            city, splits_c["val"], splits_c["test"], model_c, scaler_c
        )

        all_results.append(
            {
                "city": city,
                "target": "cooling_demand_mw",
                "strategy": "hybrid_adaptive_conformal",
                "alpha": ALPHA_COOLING,
                "n_detected": int(detected_c.sum()),
                "n_true_anom": int(splits_c["test"]["anomaly_cooling"].sum()),
                **metrics_c,
            }
        )

    combined = pd.DataFrame(all_results)
    output_path = os.path.join(PROCESSED_DIR, "conformal_detection_results.csv")
    combined.to_csv(output_path, index=False)

    print(f"\n\nSaved to: {output_path}")
    print("\nFull results:")
    print(combined.to_string(index=False))

    return combined


if __name__ == "__main__":
    print("=== Final Conformal Prediction Anomaly Detector ===")
    print("Heating : original conformal (raw residuals, alpha=0.05, no persistence)")
    print(
        "Cooling : hybrid adaptive (normalized active months, raw zero-demand months,"
    )
    print(f"          alpha=0.01, persistence>={MIN_CONSECUTIVE_COOLING}h)")
    run_conformal_detection()
