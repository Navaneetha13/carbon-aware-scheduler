"""Research-backed carbon-intensity forecasting.
Literature (CarbonCast, BuildSys 2022; multiple 2023-24 studies) shows CNN-LSTM hybrids are the leading
models for grid carbon-intensity forecasting, and ensembles (EnsembleCI 2025) are state-of-the-art.
So we build a multivariate CNN-LSTM + an ensemble and benchmark them against LSTM / GRU / Gradient Boosting
on the same real UK carbon data + held-out test. Lower MAE = fits the actual curve better."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Conv1D, MaxPooling1D
from sklearn.ensemble import GradientBoostingRegressor

ROOT = "/home/durga/carbon-aware-scheduler"
np.random.seed(42); tf.random.set_seed(42)
LOOK_BACK = 48

# ---- real carbon + time features (multivariate) ----
df = pd.read_csv(ROOT + "/data/carbon/carbon_history.csv")
carbon = df["intensity"].astype(float).values
ts = pd.to_datetime(df["from"], utc=True)
slot = ts.dt.hour.values*2 + ts.dt.minute.values//30; dow = ts.dt.dayofweek.values
nte = int(len(carbon)*0.2); lo, hi = carbon[:-nte].min(), carbon[:-nte].max()
csc = (carbon-lo)/(hi-lo); inv = lambda a: a*(hi-lo)+lo
feats = np.column_stack([csc, np.sin(2*np.pi*slot/48), np.cos(2*np.pi*slot/48),
                         np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])
def windows(F, t):
    return (np.array([F[i:i+LOOK_BACK] for i in range(len(F)-LOOK_BACK)]),
            np.array([t[i+LOOK_BACK] for i in range(len(F)-LOOK_BACK)]))
split = len(feats)-nte
Xtr, ytr = windows(feats[:split], csc[:split])
Xte, yte = windows(feats[split-LOOK_BACK:], csc[split-LOOK_BACK:])
true = inv(yte); nf = feats.shape[1]

def fit_keras(model, ep):
    model.compile("adam","mse"); model.fit(Xtr, ytr, epochs=ep, batch_size=32, verbose=0)
    return inv(model.predict(Xte, verbose=0).ravel())

print("Training models...")
preds = {}
# CNN-LSTM (CarbonCast-style): stacked Conv1D feature extractor -> LSTM
preds["CNN-LSTM (CarbonCast-style)"] = fit_keras(Sequential([
    Input((LOOK_BACK, nf)), Conv1D(64, 3, activation="relu", padding="same"),
    Conv1D(64, 3, activation="relu", padding="same"), MaxPooling1D(2), LSTM(48), Dense(1)]), 40)
preds["LSTM"] = fit_keras(Sequential([Input((LOOK_BACK, nf)), LSTM(32), Dense(1)]), 30)
preds["GRU"]  = fit_keras(Sequential([Input((LOOK_BACK, nf)), GRU(32), Dense(1)]), 30)
gbr = GradientBoostingRegressor(n_estimators=400, max_depth=3, random_state=42)
gbr.fit(Xtr.reshape(len(Xtr),-1), ytr); preds["Gradient Boosting"] = inv(gbr.predict(Xte.reshape(len(Xte),-1)))
# Ensemble (EnsembleCI-style): average of the strong models
preds["Ensemble (CNN-LSTM+GRU+GBR)"] = np.mean([preds["CNN-LSTM (CarbonCast-style)"],
                                                preds["GRU"], preds["Gradient Boosting"]], axis=0)

mae = {k: float(np.mean(np.abs(v-true))) for k, v in preds.items()}
rank = sorted(mae.items(), key=lambda kv: kv[1])
print("\n=== Forecast MAE on held-out test (gCO2/kWh) — lower fits better ===")
for k, v in rank: print("  %-30s MAE %.3f" % (k, v))
best = rank[0][0]
print("\nBest (research-backed) model: %s  (MAE %.3f)" % (best, rank[0][1]))
pd.DataFrame({k:[round(v,3)] for k,v in mae.items()}, index=["MAE"]).T.to_csv(ROOT+"/results/advanced_forecast_comparison.csv")

m = min(144, len(true))
plt.figure(figsize=(12,4)); plt.plot(true[:m], color="black", lw=2, label="Actual")
plt.plot(preds["LSTM"][:m], "--", alpha=0.8, label="LSTM (MAE %.2f)" % mae["LSTM"])
plt.plot(preds[best][:m], ":", lw=2.2, color="tab:green", label="%s (MAE %.2f)" % (best, mae[best]))
plt.title("Carbon forecast — research-backed model vs LSTM (held-out)")
plt.xlabel("half-hour slot"); plt.ylabel("gCO2/kWh"); plt.legend(); plt.tight_layout()
plt.savefig(ROOT+"/results/advanced_forecast_comparison.png", dpi=120)
print("Saved -> results/advanced_forecast_comparison.csv and .png")
