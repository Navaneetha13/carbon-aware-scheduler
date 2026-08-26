"""Fetch real carbon-intensity traces for additional regions (Reviewer 2 s4 #9,
Reviewer 3 #1 and #9: "at least one additional carbon region (US or EU)").

Source: EirGrid Smart Grid Dashboard public API -- no key required.
  ROI = Republic of Ireland, NI = Northern Ireland.
Published at 15-minute resolution; resampled to the 30-minute slots used
throughout this project so the scheduling horizon is directly comparable to the
UK National Grid signal already in data/carbon/.

Run:  python fetch_regions.py
Writes: ../data/carbon/carbon_history_<REGION>.csv  (columns: from,intensity)
"""
import io, json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://www.smartgriddashboard.com/DashboardService.svc/data"
# match the span of the existing UK history so comparisons are like-for-like
START, END = datetime(2026, 2, 28), datetime(2026, 6, 19)
CHUNK = timedelta(days=7)


def fetch(region, a, b):
    q = urllib.parse.urlencode({"area": "co2intensity", "region": region,
                                "datefrom": a.strftime("%d-%b-%Y 00:00"),
                                "dateto": b.strftime("%d-%b-%Y 23:59")})
    with urllib.request.urlopen("%s?%s" % (API, q), timeout=60) as r:
        return json.loads(r.read().decode())["Rows"]


def collect(region):
    rows, a = [], START
    while a < END:
        b = min(a + CHUNK, END)
        for attempt in range(3):
            try:
                rows += fetch(region, a, b)
                break
            except Exception as e:                       # transient network error
                if attempt == 2:
                    print("   WARN %s %s: %s" % (region, a.date(), e))
                time.sleep(2 * (attempt + 1))
        a = b + timedelta(days=1)
        print("   %s ... %d rows" % (region, len(rows)), end="\r", flush=True)
    return rows


def to_halfhourly(rows):
    """15-min -> 30-min slots by averaging, dropping nulls. Returns sorted pairs."""
    buck = {}
    for r in rows:
        v = r.get("Value")
        if v is None:
            continue
        t = datetime.strptime(r["EffectiveTime"], "%d-%b-%Y %H:%M:%S")
        t = t.replace(minute=0 if t.minute < 30 else 30, second=0)
        buck.setdefault(t, []).append(float(v))
    return [(t, sum(v) / len(v)) for t, v in sorted(buck.items())]


if __name__ == "__main__":
    for region, name in (("ROI", "IE"), ("NI", "NI")):
        print("fetching %s (%s) ..." % (name, region))
        series = to_halfhourly(collect(region))
        if not series:
            print("   FAILED: no rows for %s" % name)
            continue
        path = os.path.join(ROOT, "data", "carbon", "carbon_history_%s.csv" % name)
        # merge with anything already fetched so a re-run backfills chunks that
        # failed with a transient 503 instead of discarding good data
        if os.path.exists(path):
            prev = {}
            for line in open(path).read().splitlines()[1:]:
                if "," in line:
                    k, v = line.rsplit(",", 1)
                    prev[datetime.strptime(k, "%Y-%m-%dT%H:%MZ")] = float(v)
            prev.update(dict(series))
            series = sorted(prev.items())
        with open(path, "w") as f:
            f.write("from,intensity\n")
            for t, v in series:
                f.write("%s,%.1f\n" % (t.strftime("%Y-%m-%dT%H:%MZ"), v))
        vals = [v for _, v in series]
        print("   %-3s %5d half-hourly slots | %s -> %s | min %.0f max %.0f mean %.1f"
              % (name, len(series), series[0][0].date(), series[-1][0].date(),
                 min(vals), max(vals), sum(vals) / len(vals)))
        print("   saved -> %s" % path)
