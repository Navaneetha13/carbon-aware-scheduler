"""Forecast-driven scheduling: schedule deferrable jobs using the LSTM's PREDICTED carbon, then score
the schedule on the ACTUAL carbon. Compares three schedulers on an unseen 3-day window:
  * Reactive   : no future knowledge -> run each job at its earliest slot.
  * Forecast   : shift each job to its lowest *predicted* (LSTM) carbon slot  [realistic].
  * Oracle      : shift each job to its lowest *actual* carbon slot           [perfect foresight, upper bound].
All real data. Shows forecasting captures most of the oracle benefit -> reactive becomes predictive."""
import os, math, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT = "/home/durga/carbon-aware-scheduler"
np.random.seed(42); tf.random.set_seed(42)
LOOK_BACK, H = 48, 144

# ---- rebuild the same split/scaling and get LSTM forecast on the held-out window ----
series = pd.read_csv(ROOT + "/data/carbon/carbon_history.csv")["intensity"].astype(float).values
n_test = int(len(series)*0.2); train_raw, test_raw = series[:-n_test], series[-n_test:]
lo, hi = train_raw.min(), train_raw.max(); scale = lambda a:(a-lo)/(hi-lo); inv = lambda a:a*(hi-lo)+lo
ctx = scale(np.concatenate([train_raw[-LOOK_BACK:], test_raw]))
Xte = np.array([ctx[i:i+LOOK_BACK] for i in range(len(ctx)-LOOK_BACK)])[..., None]
true = inv(np.array([ctx[i+LOOK_BACK] for i in range(len(ctx)-LOOK_BACK)]))
model = tf.keras.models.load_model(ROOT + "/results/lstm_carbon.keras")
pred = inv(model.predict(Xte, verbose=0).ravel())

CI_actual   = true[-H:]          # real carbon of the unseen 3-day scheduling window
CI_forecast = pred[-H:]          # the LSTM's prediction for that window
print("Window: %d slots | actual %.0f-%.0f, forecast MAE %.2f gCO2/kWh"
      % (H, CI_actual.min(), CI_actual.max(), np.mean(np.abs(CI_forecast-CI_actual))))

# ---- real Google tasks mapped onto this window ----
cols = ["time","missing","job_id","task_index","machine_id","event_type","user","sched_class",
        "priority","cpu_request","mem_request","disk_request","constraint"]
df = pd.read_csv(ROOT + "/data/workload/google_task_events_part0.csv.gz", header=None, names=cols)
sub = (df[df.event_type==0][["job_id","task_index","time","cpu_request"]].dropna(subset=["cpu_request"])
       .rename(columns={"time":"submit"}).groupby(["job_id","task_index"], as_index=False).first())
end = (df[df.event_type.isin([2,3,4,5])][["job_id","task_index","time"]]
       .rename(columns={"time":"end"}).groupby(["job_id","task_index"], as_index=False).first())
m = sub.merge(end, on=["job_id","task_index"]); m["dur_us"] = m["end"]-m["submit"]
m = m[(m.dur_us>0)&(m.cpu_request>0)].reset_index(drop=True).head(60)
smin, smax, SLACK, MAX_DEFER = m.submit.min(), m.submit.max(), 8, 24
tasks = []
for _, r in m.iterrows():
    dur = int(np.clip(math.ceil(r.dur_us/1.8e9),1,12)); u = float(np.clip(r.cpu_request,0.05,1.0))
    e = int((r.submit-smin)/(smax-smin+1)*(H//3)); tasks.append({"dur":dur,"u":u,"earliest":e,"deadline":e+dur+SLACK})

P_IDLE_W, P_MAX_W, SLOT_H = 100.0, 250.0, 0.5
def slot_e(u): return (P_IDLE_W+(P_MAX_W-P_IDLE_W)*u)*SLOT_H/1000.0

def real_carbon(starts):                      # always scored on ACTUAL carbon
    c = 0.0
    for t, s in zip(tasks, starts):
        c += sum(slot_e(t["u"])*CI_actual[k] for k in range(s, s+t["dur"]) if k < H)
    return c/1000.0

def greedy(CI_decide):                         # shift each job to its lowest-(CI_decide) feasible slot
    starts = []
    for t in tasks:
        hi_o = max(0, min(MAX_DEFER, H-t["dur"]-t["earliest"], t["deadline"]-t["dur"]-t["earliest"]))
        bo, bc = 0, float("inf")
        for o in range(0, hi_o+1):
            c = sum(CI_decide[k] for k in range(t["earliest"]+o, t["earliest"]+o+t["dur"]) if k < H)
            if c < bc: bc, bo = c, o
        starts.append(t["earliest"]+bo)
    return starts

reactive = real_carbon([t["earliest"] for t in tasks])
forecast = real_carbon(greedy(CI_forecast))
oracle   = real_carbon(greedy(CI_actual))
red_f = (reactive-forecast)/reactive*100
red_o = (reactive-oracle)/reactive*100
print("\n=== Carbon (kg CO2) of each scheduler, scored on ACTUAL carbon ===")
print("Reactive (no forecast)      : %.4f  (0.0%% reduction)" % reactive)
print("Forecast-driven (LSTM)      : %.4f  (%.1f%% reduction)" % (forecast, red_f))
print("Oracle (perfect foresight)  : %.4f  (%.1f%% reduction)" % (oracle, red_o))
print("\n-> Forecast-driven captures %.0f%% of the oracle's possible saving." % (100*red_f/red_o if red_o else 0))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(CI_actual, color="tab:red", label="Actual"); ax[0].plot(CI_forecast, "--", color="tab:blue", label="LSTM forecast")
ax[0].set(title="Carbon over scheduling window", xlabel="slot", ylabel="gCO2/kWh"); ax[0].legend()
ax[1].bar(["Reactive","Forecast (LSTM)","Oracle"], [reactive, forecast, oracle],
          color=["tab:gray","tab:green","tab:blue"])
ax[1].set(title="Carbon emitted (kg CO2) — lower is better", ylabel="kg CO2")
plt.tight_layout(); plt.savefig(ROOT + "/results/lstm_scheduling.png", dpi=120)
print("Saved -> results/lstm_scheduling.png")
