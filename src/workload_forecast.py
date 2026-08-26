"""Workload forecasting (completes the LSTM task: 'carbon intensity AND workload').
Builds an hourly workload-demand series from the REAL NASA-iPSC trace (job arrivals over ~months),
trains an LSTM to forecast incoming demand, and evaluates on held-out data. Predicting demand peaks
supports 'predictive scaling' (provisioning/consolidating servers ahead of load)."""
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

import os
# repo root = parent directory of src/ ; keeps the project runnable from any location
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42); tf.random.set_seed(42)
# Kernel-level determinism: without this, TF kernels are nondeterministic even with a
# fixed seed, which is why the same model previously scored differently between runs.
# Requires TF >= 2.8; guarded so older versions still run.
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    print("note: op-determinism unavailable in this TensorFlow build; results may vary between runs")

# ---- build hourly workload-demand series from the real NASA trace ----
cols = ["job","submit","wait","runtime","nproc","avg_cpu","used_mem","req_proc","req_time",
        "req_mem","status","uid","gid","app","queue","partition","prev_job","think"]
df = pd.read_csv(ROOT + "/data/workload/NASA.swf", sep=r"\s+", comment=";", header=None, names=cols)
df = df[(df.submit >= 0) & (df.nproc > 0)]
HOUR = 3600
bins = (df.submit // HOUR).astype(int)
demand = np.bincount(bins, weights=df.nproc.values)          # processors requested per hour (incoming load)
# trim leading/trailing empty hours
nz = np.nonzero(demand)[0]
series = demand[nz.min(): nz.max()+1].astype(float)
print("Workload series: %d hourly points (%.0f days), mean demand %.1f, peak %.0f"
      % (len(series), len(series)/24, series.mean(), series.max()))

LOOK_BACK = 24                                               # use past 24 h to predict next hour
nte = int(len(series)*0.2); tr, te = series[:-nte], series[-nte:]
lo, hi = tr.min(), tr.max()
# The scaler is fitted on the TRAIN split only; fitting on the full series would leak
# test information. Values outside the fitted range are clipped rather than extrapolated,
# and the count is reported, so an out-of-distribution demand spike cannot pass silently.
# On the current NASA trace nothing falls outside the range, so the held-out MAE is unaffected.
def sc(a):
    z = (np.asarray(a, dtype=float) - lo) / (hi - lo + 1e-9)
    oob = int(((z < 0.0) | (z > 1.0)).sum())
    if oob:
        print("WARNING: %d/%d demand values fall outside the training range [%.0f, %.0f] "
              "procs/h and were clipped to it." % (oob, z.size, lo, hi))
    return np.clip(z, 0.0, 1.0)
inv = lambda a: a*(hi-lo+1e-9)+lo
def win(a):
    return (np.array([a[i:i+LOOK_BACK] for i in range(len(a)-LOOK_BACK)])[..., None],
            np.array([a[i+LOOK_BACK] for i in range(len(a)-LOOK_BACK)]))
Xtr, ytr = win(sc(tr)); ctx = sc(np.concatenate([tr[-LOOK_BACK:], te])); Xte, yte = win(ctx)

model = Sequential([Input((LOOK_BACK,1)), LSTM(32), Dense(1)]); model.compile("adam","mse")
print("Training workload LSTM..."); model.fit(Xtr, ytr, epochs=30, batch_size=32, verbose=0)
pred = inv(model.predict(Xte, verbose=0).ravel()); true = inv(yte)
mae = float(np.mean(np.abs(pred-true))); rmse = float(np.sqrt(np.mean((pred-true)**2)))
persist = inv(ctx[LOOK_BACK-1:-1]); mae_p = float(np.mean(np.abs(persist-true)))
print("\n=== Workload forecast accuracy on HELD-OUT data (procs/hour) ===")
print("LSTM       : MAE %.1f | RMSE %.1f" % (mae, rmse))
print("Persistence: MAE %.1f  (naive)" % mae_p)
print("LSTM improves MAE by %.0f%% over naive" % (100*(mae_p-mae)/mae_p))

m = min(168, len(true))
plt.figure(figsize=(11,4))
plt.plot(true[:m], label="Actual demand", color="tab:red")
plt.plot(pred[:m], "--", label="LSTM forecast", color="tab:blue")
plt.title("Workload demand: LSTM forecast vs actual (held-out) — MAE %.0f procs/h" % mae)
plt.xlabel("hour"); plt.ylabel("processors requested"); plt.legend(); plt.tight_layout()
plt.savefig(ROOT + "/results/workload_forecast.png", dpi=120)
print("\nSaved -> results/workload_forecast.png")
