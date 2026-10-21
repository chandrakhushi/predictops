"""Live scorer: every few seconds, pull the last hour (W ticks) of telemetry per host from
ClickHouse, score failure risk, and write it back for the dashboard."""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import joblib
import numpy as np

from predictops.features import NAMES, W, window_features
from predictops.sim import COLS

CH_URL = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123")
CH_AUTH = {"X-ClickHouse-User": "default", "X-ClickHouse-Key": os.environ.get("CLICKHOUSE_PASSWORD", "predictops")}
LAG = 3  # seconds; Kafka engine flushes every second, so give the newest tick time to land


def ch(sql, rows=None):
    body = "\n".join(map(json.dumps, rows)).encode() if rows is not None else None
    req = urllib.request.Request(f"{CH_URL}/?{urllib.parse.urlencode({'query': sql})}", data=body, headers=CH_AUTH)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return [json.loads(line) for line in r.read().splitlines()]
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()[:300]) from None


def fetch(end):
    start = end - W + 1
    sql = f"""
    SELECT host, toUnixTimestamp(ts) AS t, cpu, mem, disk, p99_ms, warns, errors
    FROM (SELECT host, ts, avg(cpu) cpu, avg(mem) mem, avg(disk) disk, avg(p99_ms) p99_ms
          FROM metrics WHERE ts BETWEEN toDateTime({start}) AND toDateTime({end}) GROUP BY host, ts) AS m
    LEFT JOIN (SELECT host, ts, countIf(level = 'WARN') warns, countIf(level = 'ERROR') errors
               FROM logs WHERE ts BETWEEN toDateTime({start}) AND toDateTime({end}) GROUP BY host, ts) AS l
    USING (host, ts) FORMAT JSONEachRow"""
    series = {}
    for r in ch(sql):
        arr = series.setdefault(r["host"], np.full((W, len(COLS)), np.nan, np.float32))
        arr[r["t"] - start] = [r[c] for c in COLS]
    for arr in series.values():  # one dropped tick should not blank the host; a real outage (longer gap) should
        for i in range(1, W - 1):
            if np.isnan(arr[i, 0]) and not np.isnan(arr[i + 1, 0]):
                arr[i] = arr[i - 1]
    return series


def explain(f):
    """Plain-English reason an on-call engineer can act on."""
    v = dict(zip(NAMES, f))
    tte = min(("memory", v["mem_tte"]), ("disk", v["disk_tte"]), key=lambda x: x[1])
    if tte[1] < 240:
        return f"{tte[0]} on track to run out in ~{tte[1]:.0f} min"
    return f"p99 {v['p99_ms_last']:.0f}ms ({v['p99_ms_slope']:+.1f}ms/min), {v['warns_mean']:.1f} warns/min"


def run(model_path, every):
    bundle = joblib.load(model_path)
    model, threshold = bundle["model"], bundle["threshold"]
    print(f"loaded {model_path}, alert threshold {threshold:.2f}")
    while True:
        end = int(time.time()) - LAG
        try:
            series = fetch(end)
        except (OSError, RuntimeError) as e:  # ClickHouse restarting etc.: keep the scorer alive
            print(f"{time.strftime('%X')} fetch failed: {e}", flush=True)
            time.sleep(every)
            continue
        hosts, feats = [], []
        for host, arr in series.items():
            f = window_features(arr)[0]
            if not np.isnan(f).any():
                hosts.append(host)
                feats.append(f)
        if feats:
            risk = model.predict_proba(np.array(feats))[:, 1]
            ch("INSERT INTO predictions FORMAT JSONEachRow",
               [{"ts": end, "host": h, "risk": round(float(r), 4), "alert": int(r >= threshold), "reason": explain(f)} for h, r, f in zip(hosts, risk, feats)])
            hot = [f"{h}={r:.2f}" for h, r in sorted(zip(hosts, risk), key=lambda x: -x[1]) if r >= threshold]
            print(f"{time.strftime('%X')} scored {len(hosts)} hosts, at risk: {', '.join(hot) or 'none'}", flush=True)
        else:
            print(f"{time.strftime('%X')} waiting for {W}s of telemetry ({len(series)} hosts seen)", flush=True)
        time.sleep(every)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/predictor.joblib")
    p.add_argument("--every", type=float, default=5)
    a = p.parse_args()
    run(a.model, a.every)
