"""LSTM forecasting of grid carbon intensity (Week 6, Pillar 2).
Trains an LSTM on ~111 days of REAL UK National Grid carbon data to predict future carbon intensity,
then reports accuracy on a held-out (unseen) test period. Output feeds the carbon-aware scheduler so it
can shift jobs toward PREDICTED low-carbon windows (reactive -> predictive)."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

ROOT = "/home/durga/carbon-aware-scheduler"
np.random.seed(42); tf.random.set_seed(42)
LOOK_BACK = 48          # use the past 24 h (48 half-hour slots) to predict the next slot

# ---- real carbon history ----
series = pd.read_csv(ROOT + "/data/carbon/carbon_history.csv")["intensity"].astype(float).values
print("History: %d half-hourly points (%.0f days)" % (len(series), len(series)/48))

# ---- train/test split (last 20% held out, never seen in training) ----
n_test = int(len(series) * 0.2)
train_raw, test_raw = series[:-n_test], series[-n_test:]

# ---- scale using TRAIN stats only (no leakage) ----
lo, hi = train_raw.min(), train_raw.max()
scale = lambda a: (a - lo) / (hi - lo)
inv   = lambda a: a * (hi - lo) + lo

def windows(arr):
    X, y = [], []
    for i in range(len(arr) - LOOK_BACK):
        X.append(arr[i:i+LOOK_BACK]); y.append(arr[i+LOOK_BACK])
    return np.array(X)[..., None], np.array(y)

Xtr, ytr = windows(scale(train_raw))
# build test windows using the tail of train as warm-up context
ctx = scale(np.concatenate([train_raw[-LOOK_BACK:], test_raw]))
Xte, yte = windows(ctx)

# ---- model ----
model = Sequential([Input((LOOK_BACK, 1)), LSTM(32), Dense(1)])
model.compile(optimizer="adam", loss="mse")
print("Training LSTM...")
model.fit(Xtr, ytr, epochs=25, batch_size=32, validation_split=0.1, verbose=0)

# ---- evaluate on held-out test (1-step-ahead), back in real units ----
pred = inv(model.predict(Xte, verbose=0).ravel())
true = inv(yte)
mae  = float(np.mean(np.abs(pred - true)))
rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
# naive baseline: persistence (predict next = current)
persist = inv(ctx[LOOK_BACK-1:-1])
mae_p   = float(np.mean(np.abs(persist - true)))
print("\n=== Forecast accuracy on HELD-OUT test (%d points, gCO2/kWh) ===" % len(true))
print("LSTM       : MAE %.2f | RMSE %.2f" % (mae, rmse))
print("Persistence: MAE %.2f   (naive baseline)" % mae_p)
print("LSTM improves MAE by %.0f%% over naive" % (100*(mae_p-mae)/mae_p))

# ---- plot forecast vs actual (first 3 test days) ----
m = min(144, len(true))
plt.figure(figsize=(11, 4))
plt.plot(true[:m], label="Actual carbon intensity", color="tab:red")
plt.plot(pred[:m], label="LSTM forecast", color="tab:blue", ls="--")
plt.title("LSTM 1-step forecast vs actual (held-out test) — MAE %.1f gCO2/kWh" % mae)
plt.xlabel("half-hour slot"); plt.ylabel("gCO2/kWh"); plt.legend(); plt.tight_layout()
plt.savefig(ROOT + "/results/lstm_forecast_vs_actual.png", dpi=120)
model.save(ROOT + "/results/lstm_carbon.keras")
print("\nSaved -> results/lstm_forecast_vs_actual.png and results/lstm_carbon.keras")
