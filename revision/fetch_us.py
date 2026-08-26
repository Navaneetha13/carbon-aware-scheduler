"""Fetch a US carbon-intensity trace (Reviewer 3 #9: "at least one additional
carbon region (US or EU)"; Reviewer 2 s4 #9: multiple carbon-intensity traces).

Source: US EIA API v2, hourly generation by fuel type per balancing authority.
The EIA does not publish carbon intensity directly, so it is derived from the
fuel mix:

    CI_t = sum_f (generation_f,t * EF_f) / sum_f generation_f,t

using IPCC AR5 (2014) Annex III median LIFECYCLE emission factors, the same
convention used by Electricity Maps and most carbon-aware scheduling papers.
Storage discharge (batteries, pumped hydro) is excluded from both sums: it is
not primary generation, and charging emissions are already counted upstream.

EIA publishes hourly; the project uses 30-minute slots, so each hour is expanded
into two identical slots. This is an upsample, not new information -- it is
recorded here so the limitation is explicit in the paper.

Requires a free API key from https://www.eia.gov/opendata/register.php,
read from the EIA_API_KEY environment variable or revision/.env (gitignored).

Run:  python fetch_us.py
Writes: ../data/carbon/carbon_history_US-<BA>.csv  (columns: from,intensity)
"""
import json, os, time, urllib.parse, urllib.request
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# IPCC AR5 Annex III, median lifecycle gCO2eq/kWh
EF = {"COL": 820, "NG": 490, "OIL": 650, "NUC": 12, "WAT": 24,
      "SUN": 48, "WND": 11, "GEO": 38, "BIO": 230, "OTH": 700}
SKIP = {"BAT", "PS"}                       # storage, not generation
START, END = datetime(2026, 2, 28), datetime(2026, 6, 19)
REGIONS = [("CAL", "California"), ("MIDA", "Mid-Atlantic")]


def key():
    k = os.environ.get("EIA_API_KEY")
    if not k and os.path.exists(os.path.join(HERE, ".env")):
        for line in open(os.path.join(HERE, ".env")):
            if line.startswith("EIA_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("set EIA_API_KEY (see https://www.eia.gov/opendata/register.php)")
    return k


def fetch(ba, a, b, k):
    q = [("api_key", k), ("frequency", "hourly"), ("data[0]", "value"),
         ("facets[respondent][]", ba), ("start", a.strftime("%Y-%m-%dT00")),
         ("end", b.strftime("%Y-%m-%dT23")), ("sort[0][column]", "period"),
         ("sort[0][direction]", "asc"), ("length", "5000")]
    url = ("https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?"
           + urllib.parse.urlencode(q))
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.loads(r.read().decode())["response"]["data"]


def intensity(rows):
    """Fuel mix -> carbon intensity per hour."""
    by_hour = {}
    for r in rows:
        f = r["fueltype"]
        if f in SKIP:
            continue
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        if v <= 0:                          # negative/zero = net import or curtailment
            continue
        g, e = by_hour.setdefault(r["period"], [0.0, 0.0])
        by_hour[r["period"]] = [g + v, e + v * EF.get(f, 700)]
    return {datetime.strptime(p, "%Y-%m-%dT%H"): e / g
            for p, (g, e) in sorted(by_hour.items()) if g > 0}


if __name__ == "__main__":
    k = key()
    for ba, name in REGIONS:
        print("fetching US-%s (%s) ..." % (ba, name))
        rows, a = [], START
        while a < END:
            b = min(a + timedelta(days=10), END)
            for att in range(3):
                try:
                    rows += fetch(ba, a, b, k); break
                except Exception as ex:
                    if att == 2:
                        print("   WARN %s: %s" % (a.date(), ex))
                    time.sleep(2 * (att + 1))
            a = b + timedelta(days=1)
            print("   %s ... %d rows" % (ba, len(rows)), end="\r", flush=True)

        hourly = intensity(rows)
        if not hourly:
            print("   FAILED: no usable rows for %s" % ba); continue
        # hourly -> two identical 30-minute slots
        series = []
        for t, v in sorted(hourly.items()):
            series.append((t, v)); series.append((t + timedelta(minutes=30), v))

        path = os.path.join(ROOT, "data", "carbon", "carbon_history_US-%s.csv" % ba)
        if os.path.exists(path):
            prev = {}
            for line in open(path).read().splitlines()[1:]:
                if "," in line:
                    kk, vv = line.rsplit(",", 1)
                    prev[datetime.strptime(kk, "%Y-%m-%dT%H:%MZ")] = float(vv)
            prev.update(dict(series)); series = sorted(prev.items())
        with open(path, "w") as f:
            f.write("from,intensity\n")
            for t, v in series:
                f.write("%s,%.1f\n" % (t.strftime("%Y-%m-%dT%H:%MZ"), v))
        vals = [v for _, v in series]
        print("   US-%-4s %5d slots | %s -> %s | min %.0f max %.0f mean %.1f"
              % (ba, len(series), series[0][0].date(), series[-1][0].date(),
                 min(vals), max(vals), sum(vals) / len(vals)))
        print("   saved -> %s" % path)
