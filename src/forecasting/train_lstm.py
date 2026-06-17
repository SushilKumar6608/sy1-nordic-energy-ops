import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.insert(0, os.path.dirname(__file__))
from prepare_features import load_and_prepare, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Option C: 3-hour pooled sequences
# 7 days of history at 3-hour resolution = 56 timesteps
# Captures weekly seasonality with better gradient flow than 168 raw steps
POOL_HOURS = 3
LOOKBACK_DAYS = 7
SEQUENCE_LENGTH = (LOOKBACK_DAYS * 24) // POOL_HOURS  # = 56

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


def mape(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> float:
    """
    Compute MAPE with a target-aware threshold.

    Heating: mask values below 0.1 MW (near-zero during mild weather)
    Cooling: mask values below 1.0 MW (Sweden has near-zero cooling
             demand for 8-9 months -- MAPE is only meaningful when
             cooling demand is non-trivial)
    """
    threshold = 1.0 if "cooling" in target else 0.1
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    mae      = mean_absolute_error(y_true, y_pred)
    rmse     = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_val = mape(y_true, y_pred, target)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape_val, 4)}


def build_pooled_sequences(X_hourly: np.ndarray,
                           y_hourly: np.ndarray,
                           pool_hours: int,
                           seq_len: int):
    """
    Build (X_seq, y_target) pairs using 3-hour pooled lookback windows.

    For each hourly target timestep t:
    - Look back seq_len * pool_hours raw hours
    - Average every pool_hours consecutive rows into one pooled step
    - Input sequence shape: (seq_len, n_features)
    - Target: y at hourly resolution (NOT averaged -- we predict actual MW)

    This preserves weekly context (7 days) while reducing sequence length
    from 168 to 56 steps, improving gradient flow and cutting compute ~3x.
    """
    n = len(X_hourly)
    lookback = seq_len * pool_hours  # 56 * 3 = 168 raw hours

    X_seqs = []
    y_targets = []

    for t in range(lookback, n):
        raw_window = X_hourly[t - lookback: t]
        pooled = raw_window.reshape(seq_len, pool_hours, raw_window.shape[1]).mean(axis=1)
        X_seqs.append(pooled)
        y_targets.append(y_hourly[t])

    return np.array(X_seqs, dtype=np.float32), np.array(y_targets, dtype=np.float32)


class PooledSequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y,     dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def train_lstm_for_target(target: str) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"LSTM (Option C: 3h pooled, seq_len={SEQUENCE_LENGTH}) -- target: {target}")
    print(f"{'='*50}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    city_splits = load_and_prepare(target)
    results = []

    for city, splits in city_splits.items():
        print(f"\nTraining {city}...")

        train = splits["train"]
        val   = splits["val"]
        test  = splits["test"]

        X_train_raw = train[FEATURE_COLS].values
        y_train_raw = train[target].values
        X_val_raw   = val[FEATURE_COLS].values
        y_val_raw   = val[target].values
        X_test_raw  = test[FEATURE_COLS].values
        y_test_raw  = test[target].values

        scaler = StandardScaler()
        X_train_raw = scaler.fit_transform(X_train_raw)
        X_val_raw   = scaler.transform(X_val_raw)
        X_test_raw  = scaler.transform(X_test_raw)

        print(f"  Building pooled sequences (pool={POOL_HOURS}h, seq_len={SEQUENCE_LENGTH})...")
        X_train_seq, y_train_seq = build_pooled_sequences(X_train_raw, y_train_raw, POOL_HOURS, SEQUENCE_LENGTH)
        X_val_seq,   y_val_seq   = build_pooled_sequences(X_val_raw,   y_val_raw,   POOL_HOURS, SEQUENCE_LENGTH)
        X_test_seq,  y_test_seq  = build_pooled_sequences(X_test_raw,  y_test_raw,  POOL_HOURS, SEQUENCE_LENGTH)

        print(f"  Train sequences: {len(X_train_seq)} | Val: {len(X_val_seq)} | Test: {len(X_test_seq)}")

        train_ds = PooledSequenceDataset(X_train_seq, y_train_seq)
        val_ds   = PooledSequenceDataset(X_val_seq,   y_val_seq)
        test_ds  = PooledSequenceDataset(X_test_seq,  y_test_seq)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

        model = LSTMForecaster(
            input_size=len(FEATURE_COLS),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss       = float("inf")
        best_state          = None
        patience_counter    = 0
        early_stop_patience = 7

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item() * len(X_batch)

            train_loss /= len(train_ds)
            val_loss   /= len(val_ds)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        model.eval()

        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                pred = model(X_batch.to(device)).cpu().numpy()
                all_preds.append(pred)
                all_true.append(y_batch.numpy())

        y_pred = np.maximum(np.concatenate(all_preds), 0.0)
        y_true = np.concatenate(all_true)

        metrics = evaluate(y_true, y_pred, target)

        threshold_note = "MAPE@>1MW" if "cooling" in target else "MAPE@>0.1MW"
        print(f"  MAE={metrics['mae']:.3f} MW | RMSE={metrics['rmse']:.3f} MW | "
              f"{threshold_note}={metrics['mape']:.2f}%")

        city_key   = (city.replace(" ", "_").replace("-", "_")
                         .replace("å", "a").replace("ä", "a").replace("ö", "o"))
        target_key = "heat" if "heating" in target else "cool"

        model_path  = os.path.join(MODELS_DIR, f"lstm_{city_key}_{target_key}.pt")
        scaler_path = os.path.join(MODELS_DIR, f"lstm_scaler_{city_key}_{target_key}.pkl")

        torch.save({
            "model_state":     best_state,
            "input_size":      len(FEATURE_COLS),
            "hidden_size":     HIDDEN_SIZE,
            "num_layers":      NUM_LAYERS,
            "dropout":         DROPOUT,
            "pool_hours":      POOL_HOURS,
            "sequence_length": SEQUENCE_LENGTH,
        }, model_path)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        results.append({
            "model":          "LSTM_3hPool",
            "city":           city,
            "target":         target,
            "mae":            metrics["mae"],
            "rmse":           metrics["rmse"],
            "mape":           metrics["mape"],
            "mape_threshold": "1.0 MW" if "cooling" in target else "0.1 MW",
            "seq_len":        SEQUENCE_LENGTH,
            "pool_hours":     POOL_HOURS,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== LSTM Forecaster -- Option C: 3-Hour Pooled Sequences ===")
    print(f"Strategy: {LOOKBACK_DAYS}-day lookback, {POOL_HOURS}h pooling -> {SEQUENCE_LENGTH} timesteps")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Hidden: {HIDDEN_SIZE} | Layers: {NUM_LAYERS}")

    results_heat = train_lstm_for_target("heating_demand_mw")
    results_cool = train_lstm_for_target("cooling_demand_mw")

    all_results = pd.concat([results_heat, results_cool], ignore_index=True)
    output_path = os.path.join(PROCESSED_DIR, "lstm_results.csv")
    all_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print("\nFull results:")
    print(all_results.to_string(index=False))