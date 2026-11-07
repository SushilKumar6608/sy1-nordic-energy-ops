import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, os.path.dirname(__file__))
from prepare_features import load_and_prepare, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def mape(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> float:
    """
    Compute MAPE with a target-aware threshold.

    Heating: mask values below 0.1 MW (near-zero during mild weather)
    Cooling: mask values below 1.0 MW (Sweden has near-zero cooling
             demand for 8-9 months — MAPE is only meaningful when
             cooling demand is non-trivial)
    """
    threshold = 1.0 if "cooling" in target else 0.1
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_val = mape(y_true, y_pred, target)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape_val, 4)}


def train_xgboost_for_target(target: str) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"XGBoost — target: {target}")
    print(f"{'='*50}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    city_splits = load_and_prepare(target)
    results = []

    for city, splits in city_splits.items():
        print(f"\nTraining {city}...")

        train = splits["train"]
        val = splits["val"]
        test = splits["test"]

        X_train = train[FEATURE_COLS].values
        y_train = train[target].values
        X_val = val[FEATURE_COLS].values
        y_val = val[target].values
        X_test = test[FEATURE_COLS].values
        y_test = test[target].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = np.maximum(model.predict(X_test), 0.0)
        metrics = evaluate(y_test, y_pred, target)

        threshold_note = "MAPE@>1MW" if "cooling" in target else "MAPE@>0.1MW"
        print(
            f"  MAE={metrics['mae']:.3f} MW | RMSE={metrics['rmse']:.3f} MW | "
            f"{threshold_note}={metrics['mape']:.2f}%"
        )

        city_key = (
            city.replace(" ", "_")
            .replace("-", "_")
            .replace("å", "a")
            .replace("ä", "a")
            .replace("ö", "o")
        )
        target_key = "heat" if "heating" in target else "cool"

        model_path = os.path.join(MODELS_DIR, f"xgb_{city_key}_{target_key}.pkl")
        scaler_path = os.path.join(
            MODELS_DIR, f"xgb_scaler_{city_key}_{target_key}.pkl"
        )

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        results.append(
            {
                "model": "XGBoost",
                "city": city,
                "target": target,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "mape_threshold": "1.0 MW" if "cooling" in target else "0.1 MW",
                "best_iteration": model.best_iteration,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== XGBoost Forecaster ===")
    results_heat = train_xgboost_for_target("heating_demand_mw")
    results_cool = train_xgboost_for_target("cooling_demand_mw")

    all_results = pd.concat([results_heat, results_cool], ignore_index=True)

    output_path = os.path.join(PROCESSED_DIR, "xgboost_results.csv")
    all_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print("\nFull results:")
    print(all_results.to_string(index=False))
