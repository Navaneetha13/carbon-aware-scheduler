"""Compare forecasting models for grid carbon intensity — find which fits (overlaps) best.
Same real data (111 days UK carbon), same look-back, same held-out test. Models: LSTM, GRU, Bi-LSTM,
CNN-LSTM, and Gradient Boosting (on lag features), vs a naive persistence baseline. Lower MAE = better fit."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Conv1D, Dense
from sklearn.ensemble import GradientBoostingRegressor

ROOT = "/home/durga/carbon-aware-scheduler"
np.random.seed(42); tf.random.set_seed(42)
LOOK_BACK = 48

series = pd.read_csv(ROOT + "/data/carbon/carbon_history.csv")["intensity"].astype(float).values
nte = int(len(series)*0.2); tr, te = series[:-nte], series[-nte:]
lo, hi = tr.min(), tr.max(); sc = lambda a:(a-lo)/(hi-lo); inv = lambda a:a*(hi-lo)+lo
def win(a):
    return (np.array([a[i:i+LOOK_BACK] for i in range(len(a)-LOOK_BACK)]),
            np.array([a[i+LOOK_BACK] for i in range(len(a)-LOOK_BACK)]))
Xtr, ytr = win(sc(tr)); ctx = sc(np.concatenate([tr[-LOOK_BACK:], te])); Xte, yte = win(ctx)
true = inv(yte)
Xtr3, Xte3 = Xtr[..., None], Xte[..., None]     # 3D for keras

def keras_mae(name, layers, epochs=25):
    m = Sequential([Input((LOOK_BACK,1))] + layers + [Dense(1)]); m.compile("adam","mse")
    m.fit(Xtr3, ytr, epochs=epochs, batch_size=32, verbose=0)
    p = inv(m.predict(Xte3, verbose=0).ravel())
    return p, float(np.mean(np.abs(p-true)))

results = {}
print("Training models...")
results["LSTM"]     = keras_mae("LSTM",     [LSTM(32)])
results["GRU"]      = keras_mae("GRU",      [GRU(32)])
results["Bi-LSTM"]  = keras_mae("Bi-LSTM",  [Bidirectional(LSTM(32))])
results["CNN-LSTM"] = keras_mae("CNN-LSTM", [Conv1D(32, 3, activation="relu"), LSTM(32)])
# Gradient Boosting on lag features (2D)
gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42).fit(Xtr, ytr)
gp = inv(gbr.predict(Xte)); results["GradBoost"] = (gp, float(np.mean(np.abs(gp-true))))
# naive persistence
pers = inv(ctx[LOOK_BACK-1:-1]); results["Persistence (naive)"] = (pers, float(np.mean(np.abs(pers-true))))

rank = sorted(results.items(), key=lambda kv: kv[1][1])
print("\n=== Forecast MAE on held-out test (gCO2/kWh) — lower fits better ===")
for name, (_, mae) in rank:
    print("  %-22s MAE %.3f" % (name, mae))
best = rank[0][0]
print("\nBest-fitting model: %s (MAE %.3f)" % (best, rank[0][1][1]))
pd.DataFrame({k: [v[1]] for k, v in results.items()}, index=["MAE"]).T.to_csv(ROOT + "/results/forecast_model_comparison.csv")

# plot best vs LSTM vs actual (first 3 days)
m = min(144, len(true))
plt.figure(figsize=(12, 4))
plt.plot(true[:m], color="black", lw=2, label="Actual")
plt.plot(results["LSTM"][0][:m], "--", label="LSTM (MAE %.2f)" % results["LSTM"][1])
plt.plot(results[best][0][:m], ":", lw=2, label="%s BEST (MAE %.2f)" % (best, results[best][1]))
plt.title("Carbon forecast — model comparison (held-out)"); plt.xlabel("half-hour slot"); plt.ylabel("gCO2/kWh")
plt.legend(); plt.tight_layout(); plt.savefig(ROOT + "/results/forecast_model_comparison.png", dpi=120)
print("Saved -> results/forecast_model_comparison.csv and .png")
