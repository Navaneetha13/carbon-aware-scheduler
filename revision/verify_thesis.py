"""Check every load-bearing number in the compiled PDF against the raw CSVs."""
import re, sys, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/Users/durga/carbon-aware-scheduler/revision")
os.chdir("/Users/durga/carbon-aware-scheduler/revision")
import core as K

PDF = open("/tmp/pdfclean.txt", encoding="latin-1").read()
def in_pdf(*frags):
    return all(f.replace(" ", "") in PDF.replace(" ", "") for f in frags)

L = lambda d: np.where((d.method=="WOA") & (d.init=="carbon"), "CA-WOA", d.method)
def load(e):
    d = pd.read_csv("raw_%s.csv" % e); d["label"] = L(d); return d

R = []
def chk(name, claim, real, tol=0.02, note=""):
    ok = abs(float(claim) - float(real)) <= tol
    R.append((ok, name, claim, round(float(real), 4), note))
def chkeq(name, claim, real, note=""):
    R.append((str(claim) == str(real), name, claim, real, note))

# --- decomposition -------------------------------------------------------
env = K.Env(K.build_published(60), *K.load_carbon(), M=None)
chk("consolidation-only reduction", 80.50, env.cred(env.cons))
g = env.cred(env.evaluate(env.greedy_starts(), consolidate=True))
chk("carbon-aware greedy", 82.56, g)
chk("timing adds", 2.06, g - env.cred(env.cons))
chk("CA-WOA adds over greedy", 0.69, 83.25 - g, tol=0.02)
chk("derived M (published sampling)", 5, np.ceil(env.peak))

# --- E20 feasibility -----------------------------------------------------
d = load("E20")
gr = d[(d.method=="Greedy(EDF+greenest)") & (d.M==10)].groupby("N")["overload_%"].mean()
chk("greedy overload N=1000 M=10", 7.14, gr[1000])
chk("greedy overload N=3000 M=10", 45.23, gr[3000])
chk("max SLA in E20", 0.0, d["sla_%"].max(), tol=1e-9)
mm = d[d.nfe>0].groupby(["N","M","label"])["carbon_red_vs_naive_%"].mean().reset_index()
w = mm.loc[mm.groupby(["N","M"])["carbon_red_vs_naive_%"].idxmax()]
chkeq("CA-WOA best in N of 12 (E20)", "9 of 12", "%d of 12" % (w.label=="CA-WOA").sum())

# --- greedy feasible rate across regions --> the 58-75% claim ------------
a = pd.concat([load("E26"), load("E28")], ignore_index=True)
fr = a[a.method=="Greedy(EDF+greenest)"].groupby("region")["feasible"].mean()
inf = (1-fr)*100
chkeq("greedy infeasible range (abstract 58-75%)", "58-75",
      "%d-%d" % (round(inf.min()), round(inf.max())))
chkeq("greedy feasible rate range", "25-42",
      "%d-%d" % (round(fr.min()*100), round(fr.max()*100)))

# --- seeding 59 of 60 ----------------------------------------------------
tot = hit = 0
for e in ("E3","E22"):
    x = load(e); x = x[(x.nfe>0)&(x.beta>0)&(x.gamma>0)]
    for _, gg in x.groupby(["N","M","method"]):
        r = gg[gg.init=="random"]["carbon_red_vs_naive_%"]; c = gg[gg.init=="carbon"]["carbon_red_vs_naive_%"]
        if len(r)>3 and len(c)>3: tot+=1; hit += c.mean()>r.mean()
chkeq("seeding helps N of 60", "59 of 60", "%d of %d" % (hit,tot))

# --- E3 ablation table ---------------------------------------------------
x = load("E3"); x = x[(x.N==3000)&(x.M==10)]
def arm(r):
    if r["method"]=="Greedy(EDF+greenest)": return "greedy"
    if r["method"]!="WOA": return None
    if r["init"]=="carbon" and r["beta"]==0: return "b0"
    if r["init"]=="carbon" and r["gamma"]==0: return "g0"
    return {"random":"woa","improved":"imp","carbon":"ca"}[r["init"]]
x = x.assign(a=x.apply(arm,axis=1)); x = x[x.a.notna()]
t = x.groupby("a")[["carbon_red_vs_naive_%","sla_%","overload_%"]].mean()
for k,(c,s_,o) in {"greedy":(89.40,0.00,45.23),"woa":(87.15,0.57,0.58),
                   "imp":(87.17,0.41,0.50),"ca":(87.58,0.13,0.03),
                   "b0":(88.39,37.53,0.00),"g0":(89.41,0.00,45.05)}.items():
    chk("ablation %s carbon"%k, c, t.loc[k,"carbon_red_vs_naive_%"])
    chk("ablation %s SLA"%k, s_, t.loc[k,"sla_%"])
    chk("ablation %s overload"%k, o, t.loc[k,"overload_%"])

# --- weights -------------------------------------------------------------
x = load("E5"); x["abg"]=x.apply(lambda r:"(%.2f,%.2f,%.2f)"%(r.alpha,r.beta,r.gamma),axis=1)
w5 = x[(x.init=="carbon")&(~x.cap)&(x.seed_frac.round(4)==0.3333)]
a10 = w5[(w5.abg=="(1.00,0.00,0.00)") & (w5.N==3000) & (w5.M==10)]
chk("alpha=1 carbon (N=3000,M=10)", 89.92, a10["carbon_red_vs_naive_%"].mean(), tol=0.05)
chk("alpha=1 SLA (N=3000,M=10)", 71.1, a10["sla_%"].mean(), tol=0.06)
chk("alpha=1 overload (N=3000,M=10)", 33.5, a10["overload_%"].mean(), tol=0.06)
pub = w5[(w5.abg=="(0.40,0.30,0.30)")&(w5.N==3000)&(w5.M==10)]
chk("published weights carbon (N=3000,M=10)", 87.58, pub["carbon_red_vs_naive_%"].mean(), tol=0.05)
chk("published weights SLA", 0.13, pub["sla_%"].mean(), tol=0.05)

# --- power models 18/18 --------------------------------------------------
x = load("E2"); x = x[x.nfe>0]
b = x.groupby(["power","N","M","label"])["carbon_red_vs_naive_%"].mean().reset_index()
wp = b.loc[b.groupby(["power","N","M"])["carbon_red_vs_naive_%"].idxmax()]
chkeq("power models CA-WOA best", "18 of 18", "%d of %d" % ((wp.label=="CA-WOA").sum(), len(wp)))

# --- VCC 12/12 -----------------------------------------------------------
x = load("E24"); x = x[x.hard]
p = x.pivot_table(index=["N","M"], columns="label", values="carbon_red_vs_naive_%")
diff = p["CA-WOA"]-p["VCC(Google)"]
chkeq("CA-WOA beats VCC", "12 of 12", "%d of %d" % ((diff>0).sum(), len(diff)))
chk("VCC margin max", 4.48, diff.max()); chk("VCC margin min", 0.35, diff.min())

# --- NFE -----------------------------------------------------------------
al = pd.concat([load(e) for e in ("E1","E2","E3","E4","E20","E21","E22")], ignore_index=True)
al = al[al.nfe>0]
chk("HHO mean NFE", 8908, al[al.method=="HHO"]["nfe"].mean(), tol=1)
chk("HHO min NFE", 8698, al[al.method=="HHO"]["nfe"].min(), tol=0)
chk("HHO max NFE", 9136, al[al.method=="HHO"]["nfe"].max(), tol=0)
chk("others NFE", 4840, al[al.method=="WOA"]["nfe"].mean(), tol=0)
chk("HHO budget ratio", 1.84, al[al.method=="HHO"]["nfe"].mean()/4840, tol=0.01)
gb = load("GABUG")
chk("OriginalGA NFE", 40, gb[gb.method=="GA_OriginalGA"]["nfe"].mean(), tol=0)

# --- het fleets 52/54 ----------------------------------------------------
x = load("E29"); tot=hit=0
for _, gg in x.groupby(["fleet","startup","N","M"]):
    ca=gg[gg.label=="CA-WOA"]["carbon_red_vs_naive_%"]; wo=gg[gg.label=="WOA"]["carbon_red_vs_naive_%"]
    if len(wo)>3: tot+=1; hit += ca.mean()>wo.mean()
chkeq("het fleets seeding helps", "52 of 54", "%d of %d" % (hit,tot))

# --- min active host -----------------------------------------------------
x = load("E23")
for mmv, cg, cw in (("0","3 of 12",1.30), ("auto","12 of 12",0.04)):
    s_ = x[x.mmin.astype(str)==mmv]; vg=[]; vw=[]
    for (N,M), gg in s_.groupby(["N","M"]):
        ca=gg[gg.label=="CA-WOA"]["carbon_red_vs_naive_%"]
        g2=gg[gg.label=="Greedy(EDF+greenest)"]["carbon_red_vs_naive_%"]
        wo=gg[gg.label=="WOA"]["carbon_red_vs_naive_%"]
        if len(g2): vg.append(ca.mean()-g2.mean())
        if len(wo)>3: vw.append(ca.mean()-wo.mean())
    chkeq("mmin=%s beats greedy"%mmv, cg, "%d of %d"%(sum(v>0 for v in vg),len(vg)))
    chk("mmin=%s vs plain WOA pp"%mmv, cw, np.mean(vw), tol=0.02)

# --- regions -------------------------------------------------------------
for r, mean, slots in (("UK",111.9,5329),("IE",168.6,5374),("NI",197.9,4990),
                       ("US-CAL",173.0,5376),("US-MIDA",333.5,5376)):
    f = "../data/carbon/carbon_history%s.csv" % ("" if r=="UK" else "_"+r)
    dd = pd.read_csv(f)
    chk("%s mean CI"%r, mean, dd.intensity.mean(), tol=0.06)
    chkeq("%s slots"%r, slots, len(dd))
    ws = K.carbon_windows(r); av = np.concatenate(ws)
chk("US-MIDA variability", 1.78, pd.read_csv("../data/carbon/carbon_history_US-MIDA.csv").pipe(
    lambda x: x.intensity.max()/x.intensity.min()), tol=0.01)
chk("UK variability", 12.30, pd.read_csv("../data/carbon/carbon_history.csv").pipe(
    lambda x: x.intensity.max()/x.intensity.min()), tol=0.01)

# --- forecast ------------------------------------------------------------
fa = pd.read_csv("raw_F_accuracy.csv"); h12 = fa[fa.horizon_h==12.0]
e12 = h12[h12.model=="Ensemble"]["MAE"]
chk("ensemble MAE @12h", 26.09, e12.mean(), tol=0.02)
chk("ensemble sd @12h", 0.48, e12.std(), tol=0.02)
chk("seasonal-naive-24h @12h", 37.03, h12[h12.model=="SeasonalNaive-24h"]["MAE"].mean(), tol=0.02)
chk("ensemble MASE @12h", 0.59, h12[h12.model=="Ensemble"]["MASE"].mean(), tol=0.01)
ml = fa[~fa.model.isin(["Persistence","SeasonalNaive-24h","SeasonalNaive-1week"])]
worst = ml.groupby(["horizon_h","model"])["MAE"].mean().unstack().idxmax(axis=1)
chkeq("CNN-LSTM worst at every horizon", True, bool((worst=="CNN-LSTM").all()))
fc = pd.read_csv("F_coupling_summary.csv")
cap = []
for (N,M,pen), gg in fc.groupby(["N","M","pen"]):
    dd = gg.set_index("scheduler")["red_mean"]
    cap.append(100*(dd["forecast"]-dd["reactive"])/(dd["oracle"]-dd["reactive"]))
chkeq("forecast captures % of oracle ADVANTAGE", "73-89",
      "%d-%d" % (round(min(cap)), round(max(cap))))

# --- convergence ---------------------------------------------------------
cv = load("CONV"); cv = cv[cv.curve.notna() & (cv.N==3000) & (cv.M==10)]
res = {}
for l, gg in cv.groupby("label"):
    m = np.array([[float(v) for v in c.split(";")] for c in gg.curve]).mean(0)
    res[l] = (m[0], m[119], int(np.argmax(m <= m[-1]*1.01))+1)
chk("CA-WOA epoch1 fitness", 0.0626, res["CA-WOA"][0], tol=0.0005)
chk("WOA epoch1 fitness", 0.1118, res["WOA"][0], tol=0.0005)
chk("CA-WOA final fitness", 0.0502, res["CA-WOA"][1], tol=0.0005)
chk("WOA final fitness", 0.0544, res["WOA"][1], tol=0.0005)
chkeq("CA-WOA epochs to 1%", 8, res["CA-WOA"][2])
chkeq("WOA epochs to 1%", 16, res["WOA"][2])
chkeq("GA epochs to 1%", 102, res["GA"][2])
chkeq("GWO epochs to 1%", 112, res["GWO"][2])

# --- E1 SLA --------------------------------------------------------------
e1 = load("E1"); e1 = e1[e1.nfe>0]
for m_, v in (("PSO",58.03),("DE",50.73),("GA",35.42),("CA-WOA",0.03)):
    chk("E1 mean SLA %s"%m_, v, e1[e1.label==m_]["sla_%"].mean(), tol=0.02)

# --- cost inert ----------------------------------------------------------
pc = pd.read_csv("../results/proposed_ca_woa_comparison.csv")
chk("temporal-model cost, all methods (0.755)", 0.755, pc["Cost_GBP"].max(), tol=0.001)
chkeq("temporal-model cost identical across methods", 1, pc["Cost_GBP"].nunique())
ac = pd.read_csv("../results/algorithm_comparison.csv")
chk("capacity-model cost min (0.796)", 0.796, ac["Cost_GBP"].min(), tol=0.001)
chk("capacity-model cost max (0.935)", 0.935, ac["Cost_GBP"].max(), tol=0.001)

# ---------------------------------------------------------------- report
bad = [r for r in R if not r[0]]
print("%-52s %-12s %-12s %s" % ("CLAIM", "IN THESIS", "IN DATA", ""))
print("-"*92)
for ok,n,c,r,note in R:
    print("%-52s %-12s %-12s %s" % (n, c, r, "OK" if ok else "*** MISMATCH ***"))
print("-"*92)
print("%d checks, %d passed, %d MISMATCHED" % (len(R), len(R)-len(bad), len(bad)))
if bad:
    print("\nMISMATCHES:")
    for ok,n,c,r,note in bad: print("  %-50s thesis=%s  data=%s" % (n,c,r))
