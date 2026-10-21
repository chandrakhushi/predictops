"""Train the failure predictor on a simulated fleet and benchmark it against static-threshold alerting.

Three independent fleets (different seeds, different hosts): train, validation
(picks the alert threshold), test (reported numbers). The model's threshold is
set so it raises no more false alarms than the static rules do on validation,
so the comparison is lead time at equal alert noise.
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from predictops.features import NAMES, W, window_features
from predictops.sim import COLS, simulate

H = 120  # label: host fails within the next 120 ticks (2 simulated hours)
GAP = 5  # alert ticks this close together count as one alert episode
DAY = 1440


def dataset(data, failures):
    n_hosts, ticks, _ = data.shape
    X = np.concatenate([window_features(data[h]) for h in range(n_hosts)])
    y = np.zeros((n_hosts, ticks - W + 1), bool)
    t = np.arange(W - 1, ticks)
    for h, ft, _ in failures:
        y[h] |= (t < ft) & (t >= ft - H)
    return X, y.ravel()


def score(model, data):
    """Risk per (host, tick); 0 where there is no full window."""
    n_hosts, ticks, _ = data.shape
    out = np.zeros((n_hosts, ticks), np.float32)
    for h in range(n_hosts):
        f = window_features(data[h])
        ok = ~np.isnan(f).any(1)
        out[h, W - 1:][ok] = model.predict_proba(f[ok])[:, 1]
    return out


def static_rules(data):
    """What most teams page on today."""
    c = {name: np.nan_to_num(data[..., i]) for i, name in enumerate(COLS)}
    return (c["mem"] > 90) | (c["disk"] > 90) | (c["p99_ms"] > 500) | (c["errors"] >= 5)


def episodes(a):
    idx = np.flatnonzero(a)
    if not len(idx):
        return []
    br = np.flatnonzero(np.diff(idx) > GAP + 1)
    return list(zip(idx[np.r_[0, br + 1]], idx[np.r_[br, len(idx) - 1]]))


def evaluate(alert, failures):
    """alert: bool (hosts, ticks). Lead time = how long the alert has been firing (continuously) when the host dies."""
    n_hosts, ticks = alert.shape
    leads = []
    for h, ft, _ in failures:
        t, gap, start = ft - 1, 0, None
        while t >= 0 and gap <= GAP:
            start, gap = (t, 0) if alert[h, t] else (start, gap + 1)
            t -= 1
        leads.append(ft - start if start is not None else 0)
    fail_at = {}
    for h, ft, _ in failures:
        fail_at.setdefault(h, []).append(ft)
    eps = [(h, s, e) for h in range(n_hosts) for s, e in episodes(alert[h])]
    true = sum(any(s <= ft <= e + GAP + 1 for ft in fail_at.get(h, [])) for h, s, e in eps)
    leads = np.array(leads)
    caught = leads[leads > 0]
    return {
        "failures": len(failures),
        "caught": len(caught) / max(len(failures), 1),
        "caught_30m_early": (leads >= 30).mean() if len(leads) else 0.0,
        "median_lead_min": float(np.median(caught)) if len(caught) else 0.0,
        "precision": true / max(len(eps), 1),
        "false_alarms_per_host_day": (len(eps) - true) / (n_hosts * ticks / DAY),
    }


def report(rows):
    keys = list(next(iter(rows.values())))
    lines = ["| | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in m.values()) + " |")
    return "\n".join(lines)


def main(hosts, days, out):
    ticks = days * DAY
    train, val, test = (simulate(n, ticks, seed=s) for n, s in ((hosts, 1), (hosts // 2, 2), (hosts // 2, 3)))

    X, y = dataset(*train)
    ok = ~np.isnan(X).any(1)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=0)
    model.fit(X[ok], y[ok])

    # Pick the lowest threshold that is no noisier than the static rules on validation.
    budget = evaluate(static_rules(val[0]), val[1])["false_alarms_per_host_day"]
    val_risk = score(model, val[0])
    threshold = next((th for th in np.arange(0.5, 1.0, 0.02)
                      if evaluate(val_risk >= th, val[1])["false_alarms_per_host_day"] <= budget), 0.98)

    results = {
        "static thresholds": evaluate(static_rules(test[0]), test[1]),
        f"model (risk >= {threshold:.2f})": evaluate(score(model, test[0]) >= threshold, test[1]),
    }
    print(f"test fleet: {hosts // 2} hosts x {days} days, {len(test[1])} failures, 1 tick = 1 simulated minute\n")
    print(report(results))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    joblib.dump({"model": model, "threshold": float(threshold), "features": NAMES}, out)
    print(f"\nsaved {out}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hosts", type=int, default=40)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--out", default="models/predictor.joblib")
    a = p.parse_args()
    main(a.hosts, a.days, a.out)
