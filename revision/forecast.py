"""Forecasting experiments for the revision.

Addresses, with real runs:
  * forecast HORIZON. The published work validated one step (30 min) while the
    scheduler defers up to 24 slots (12 h). Here: horizons 1, 6, 12, 24, 48.
  * multi-step evaluation (direct multi-step: predict y[t+h] from a window ending at t)
  * RMSE / MAPE / MASE in addition to MAE
  * naive and seasonal baselines (persistence, seasonal-naive 24 h, seasonal-naive 1 week)
  * REPEATED runs with mean +/- sd, because the published single-run figures did
    not reproduce (ensemble 3.385 -> 3.328, LSTM 3.865 -> 3.570 on a clean re-run)
  * forecast-error degradation -> scheduling performance
  * predicted vs reactive vs perfect-foresight at IDENTICAL optimisation budgets

Usage: python forecast.py [accuracy|degrade|couple|all]
"""
import os, sys, math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)
LOOK_BACK = 48
HORIZONS = [1, 6, 12, 24, 48]        # 30 min, 3 h, 6 h, 12 h, 24 h
REPEATS = 5
TEST_FRAC = 0.2


def load_series():
    df = pd.read_csv(ROOT + "/data/carbon/carbon_history.csv")
    y = df["intensity"].astype(float).values
    ts = pd.to_datetime(df["from"], utc=True)
    return y, ts.dt.hour.values * 2 + ts.dt.minute.values // 30, ts.dt.dayofweek.values


def windows(feats, target, horizon):
    n = len(feats) - LOOK_BACK - horizon + 1
    X = np.stack([feats[i:i + LOOK_BACK] for i in range(n)])
    yv = np.array([target[i + LOOK_BACK + horizon - 1] for i in range(n)])
    return X, yv


def metrics(pred, true, naive_mae):
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    nz = true != 0
    return {"MAE": mae, "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE_%": float(np.mean(np.abs(err[nz] / true[nz])) * 100),
            "MASE": mae / naive_mae if naive_mae else float("nan")}


def accuracy():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Conv1D, MaxPooling1D
    from sklearn.ensemble import GradientBoostingRegressor
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    y, slot, dow = load_series()
    n_test = int(len(y) * TEST_FRAC); split = len(y) - n_test
    lo, hi = y[:split].min(), y[:split].max()            # train-only scaling
    ysc = np.clip((y - lo) / (hi - lo), 0, 1)
    inv = lambda a: a * (hi - lo) + lo
    feats = np.column_stack([ysc,
                             np.sin(2 * np.pi * slot / 48), np.cos(2 * np.pi * slot / 48),
                             np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)])
    nf = feats.shape[1]
    rows = []

    def build(kind):
        if kind == "LSTM":
            return Sequential([Input((LOOK_BACK, nf)), LSTM(32), Dense(1)]), 30
        if kind == "GRU":
            return Sequential([Input((LOOK_BACK, nf)), GRU(32), Dense(1)]), 30
        return Sequential([Input((LOOK_BACK, nf)),
                           Conv1D(64, 3, activation="relu", padding="same"),
                           Conv1D(64, 3, activation="relu", padding="same"),
                           MaxPooling1D(2), LSTM(48), Dense(1)]), 40

    for hz in HORIZONS:
        Xtr, ytr = windows(feats[:split], ysc[:split], hz)
        off = split - LOOK_BACK - hz + 1
        Xte, yte = windows(feats[off:], ysc[off:], hz)
        true = inv(yte)
        abs_idx = np.arange(len(true)) + split           # index of each test target

        pers = inv(Xte[:, -1, 0])                        # last observed value in window
        mae_pers = float(np.mean(np.abs(pers - true)))

        def seasonal(lag):
            src = abs_idx - lag
            out = np.full(len(true), np.nan)
            out[src >= 0] = y[src[src >= 0]]
            return out

        for nm, p in (("Persistence", pers), ("SeasonalNaive-24h", seasonal(48)),
                      ("SeasonalNaive-1week", seasonal(336))):
            m = np.isfinite(p)
            r = metrics(p[m], true[m], mae_pers)
            rows.append({"horizon_slots": hz, "horizon_h": hz * 0.5, "model": nm,
                         "repeat": 0, **{k: round(v, 4) for k, v in r.items()}})
            print("  h=%-3d %-20s MAE %7.3f RMSE %7.3f MAPE %6.2f%% MASE %5.3f"
                  % (hz, nm, r["MAE"], r["RMSE"], r["MAPE_%"], r["MASE"]), flush=True)

        for rep in range(1, REPEATS + 1):
            tf.keras.utils.set_random_seed(100 + rep)
            preds = {}
            for kind in ("LSTM", "GRU", "CNN-LSTM"):
                mdl, ep = build(kind)
                mdl.compile("adam", "mse")
                mdl.fit(Xtr, ytr, epochs=ep, batch_size=32, verbose=0)
                preds[kind] = inv(mdl.predict(Xte, verbose=0).ravel())
                tf.keras.backend.clear_session()
            g = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                          random_state=100 + rep)
            g.fit(Xtr.reshape(len(Xtr), -1), ytr)
            preds["GradBoost"] = inv(g.predict(Xte.reshape(len(Xte), -1)))
            preds["Ensemble"] = np.mean([preds["CNN-LSTM"], preds["GRU"],
                                         preds["GradBoost"]], axis=0)
            for nm, p in preds.items():
                r = metrics(p, true, mae_pers)
                rows.append({"horizon_slots": hz, "horizon_h": hz * 0.5, "model": nm,
                             "repeat": rep, **{k: round(v, 4) for k, v in r.items()}})
            print("  h=%-3d rep %d  LSTM %.3f | Ensemble %.3f | persistence %.3f"
                  % (hz, rep, metrics(preds["LSTM"], true, mae_pers)["MAE"],
                     metrics(preds["Ensemble"], true, mae_pers)["MAE"], mae_pers), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT + "/raw_F_accuracy.csv", index=False)
    agg = (df[df.repeat > 0].groupby(["horizon_slots", "model"])
           .agg(MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
                RMSE_mean=("RMSE", "mean"), MAPE_mean=("MAPE_%", "mean"),
                MASE_mean=("MASE", "mean")).round(4).reset_index())
    naive = df[df.repeat == 0][["horizon_slots", "model", "MAE", "RMSE", "MAPE_%", "MASE"]]
    agg.to_csv(OUT + "/F_accuracy_summary.csv", index=False)
    naive.to_csv(OUT + "/F_naive_baselines.csv", index=False)
    print("\nLearned models (mean +/- sd over %d repeats):" % REPEATS)
    print(agg.to_string(index=False))
    print("\nNaive baselines:")
    print(naive.to_string(index=False))
    print("\nSaved -> raw_F_accuracy.csv, F_accuracy_summary.csv, F_naive_baselines.csv")


def degrade():
    """Inject controlled forecast error, decide on the noisy signal, score on truth."""
    import core as K
    CI, PRICE, H = K.load_carbon()
    rows = []
    for N, M, pen in ((1000, 10, 0.0), (3000, 20, 0.0), (3000, 10, 10.0)):
        env = K.Env(K.build_google(N, H), CI, PRICE, H, M=M, hard=True, pen=pen)
        oracle = env.cred(env.evaluate(env.greedy_starts(), consolidate=True))
        for target in (0, 5, 10, 20, 40, 80):
            for sd in range(1, 11):
                rng = np.random.default_rng(1000 + sd)
                noise = rng.normal(0, target * 1.2533, H) if target else np.zeros(H)
                noisy = np.clip(CI + noise, 1.0, None)
                envn = K.Env({"dur": env.dur, "u": env.u, "earliest": env.earliest,
                              "deadline": env.deadline}, noisy, PRICE, H, M=M,
                             hard=True, pen=pen)
                mm = env.evaluate(envn.greedy_starts(), consolidate=True)
                rows.append({"N": N, "M": M, "pen": pen, "target_mae": target,
                             "realised_mae": round(float(np.mean(np.abs(noisy - CI))), 3),
                             "seed": sd, "carbon_red_%": round(env.cred(mm), 4),
                             "oracle_red_%": round(oracle, 4),
                             "gap_pp": round(oracle - env.cred(mm), 4),
                             "sla_%": round(mm["SLA_%"], 3)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "/raw_F_degradation.csv", index=False)
    agg = (df.groupby(["N", "M", "pen", "target_mae"])
           .agg(realised_mae=("realised_mae", "mean"),
                carbon_red_mean=("carbon_red_%", "mean"),
                carbon_red_sd=("carbon_red_%", "std"),
                gap_pp=("gap_pp", "mean"), oracle=("oracle_red_%", "first"))
           .round(3).reset_index())
    agg.to_csv(OUT + "/F_degradation_summary.csv", index=False)
    print(agg.to_string(index=False))
    print("\nSaved -> raw_F_degradation.csv, F_degradation_summary.csv")


def couple():
    """Predicted vs reactive vs perfect-foresight at IDENTICAL optimisation budgets,
    using a real 12-hour-ahead multi-step forecast rather than a 1-step trace."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, GRU, Dense, Conv1D, MaxPooling1D, LSTM
    from sklearn.ensemble import GradientBoostingRegressor
    import core as K
    from mealpy import FloatVar, WOA
    tf.config.threading.set_intra_op_parallelism_threads(3)

    y, slot, dow = load_series()
    n_test = int(len(y) * TEST_FRAC); split = len(y) - n_test
    lo, hi = y[:split].min(), y[:split].max()
    ysc = np.clip((y - lo) / (hi - lo), 0, 1); inv = lambda a: a * (hi - lo) + lo
    feats = np.column_stack([ysc, np.sin(2*np.pi*slot/48), np.cos(2*np.pi*slot/48),
                             np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])
    nf = feats.shape[1]; HZ = 24
    Xtr, ytr = windows(feats[:split], ysc[:split], HZ)
    off = split - LOOK_BACK - HZ + 1
    Xte, yte = windows(feats[off:], ysc[off:], HZ)
    tf.keras.utils.set_random_seed(42)
    ps = []
    for kind, ep in (("cnn", 40), ("gru", 30)):
        m = (Sequential([Input((LOOK_BACK, nf)),
                         Conv1D(64, 3, activation="relu", padding="same"),
                         Conv1D(64, 3, activation="relu", padding="same"),
                         MaxPooling1D(2), LSTM(48), Dense(1)]) if kind == "cnn"
             else Sequential([Input((LOOK_BACK, nf)), GRU(32), Dense(1)]))
        m.compile("adam", "mse"); m.fit(Xtr, ytr, epochs=ep, batch_size=32, verbose=0)
        ps.append(inv(m.predict(Xte, verbose=0).ravel())); tf.keras.backend.clear_session()
    g = GradientBoostingRegressor(n_estimators=400, max_depth=3, random_state=42)
    g.fit(Xtr.reshape(len(Xtr), -1), ytr)
    ps.append(inv(g.predict(Xte.reshape(len(Xte), -1))))
    fc_all, true_all = np.mean(ps, axis=0), inv(yte)
    fmae = float(np.mean(np.abs(fc_all - true_all)))
    print("12 h-ahead ensemble MAE on held-out: %.3f gCO2/kWh" % fmae)

    _, PRICE, H = K.load_carbon()
    CI_true, CI_fc = true_all[-H:], fc_all[-H:]
    rows = []
    for N, M, pen in ((1000, 10, 0.0), (3000, 20, 0.0), (3000, 10, 10.0)):
        tasks = K.build_google(N, H)
        e_true = K.Env(tasks, CI_true, PRICE, H, M=M, hard=True, pen=pen)
        e_fc = K.Env(tasks, CI_fc, PRICE, H, M=M, hard=True, pen=pen)
        for sd in range(1, 11):
            rng = np.random.default_rng(sd)
            for lab, dec in (("reactive", None), ("forecast", e_fc), ("oracle", e_true)):
                if dec is None:
                    starts = tasks["earliest"]
                else:
                    prob = {"obj_func": dec.fitness,
                            "bounds": FloatVar(lb=[0.0]*N, ub=[1.0]*N),
                            "minmax": "min", "log_to": None}
                    r = WOA.OriginalWOA(epoch=120, pop_size=40).solve(
                        prob, starting_solutions=dec.pop_carbon(40, rng), seed=sd)
                    starts = dec.decode(r.solution)
                mm = e_true.evaluate(starts, consolidate=True)   # scored on TRUTH
                rows.append({"N": N, "M": M, "pen": pen, "scheduler": lab, "seed": sd,
                             "forecast_mae": round(fmae, 3),
                             "carbon_kg": round(mm["Carbon_kgCO2"], 6),
                             "carbon_red_%": round(e_true.cred(mm), 4),
                             "sla_%": round(mm["SLA_%"], 3),
                             "nfe": 0 if lab == "reactive" else 4840})
    df = pd.DataFrame(rows); df.to_csv(OUT + "/raw_F_coupling.csv", index=False)
    agg = (df.groupby(["N", "M", "pen", "scheduler"])
           .agg(carbon_kg=("carbon_kg", "mean"), red_mean=("carbon_red_%", "mean"),
                red_sd=("carbon_red_%", "std"), sla=("sla_%", "mean"),
                nfe=("nfe", "mean")).round(4).reset_index())
    agg.to_csv(OUT + "/F_coupling_summary.csv", index=False)
    print(agg.to_string(index=False))
    for (N, M, pen), grp in agg.groupby(["N", "M", "pen"]):
        d = grp.set_index("scheduler")["red_mean"]
        if {"oracle", "reactive", "forecast"} <= set(d.index):
            # Fraction of the ADVANTAGE that perfect foresight has over reactive
            # scheduling which the forecast actually captures. Dividing the two
            # total reductions instead would be wrong: both are dominated by the
            # consolidation baseline, which inflates the ratio to ~99% and makes a
            # forecast look near-perfect regardless of its accuracy.
            gain = d["oracle"] - d["reactive"]
            got = d["forecast"] - d["reactive"]
            print("  N=%d M=%d pen=%s: forecast captures %.0f%% of the advantage "
                  "perfect foresight has over reactive scheduling"
                  % (N, M, pen, 100 * got / gain if gain > 0 else float("nan")))
    print("\nSaved -> raw_F_coupling.csv, F_coupling_summary.csv")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("accuracy", "all"): print("=== forecast accuracy ==="); accuracy()
    if what in ("degrade", "all"):  print("\n=== forecast degradation ==="); degrade()
    if what in ("couple", "all"):   print("\n=== forecast coupling ==="); couple()
