"""Fleet telemetry simulator with fault injection.

One tick = one simulated minute. `simulate()` returns numpy arrays for
training; `python -m predictops.sim` streams JSON to Kafka at one tick per
wall-clock second, so a 4-hour memory leak plays out in 4 minutes.

Faults ramp toward a hard limit and then kill the host. Benign noise
(diurnal CPU, traffic spikes, disk growth reset by log rotation) is there so
the model has to tell real degradation from everyday wobble.
"""
import argparse
import json
import math
import time

import numpy as np

COLS = ("cpu", "mem", "disk", "p99_ms", "warns", "errors")
FAULTS = ("memory_leak", "disk_fill", "conn_pool")
FAILURE = {"memory_leak": "oom", "disk_fill": "disk_full", "conn_pool": "pool_exhausted"}
LIMIT = {"memory_leak": 98.0, "disk_fill": 99.0, "conn_pool": 1.0}

WARN_MSG = {
    "memory_leak": "GC pause {p99:.0f}ms, heap {mem:.0f}% used",
    "disk_fill": "slow fsync {p99:.0f}ms on /var/data ({disk:.0f}% used)",
    "conn_pool": "waited {p99:.0f}ms for db connection",
    None: "request queue depth high, cpu {cpu:.0f}%",
}
ERROR_MSG = {
    "memory_leak": "allocation failed, retrying request",
    "disk_fill": "write failed: no space left on device, retrying",
    "conn_pool": "timeout acquiring db connection after 5000ms",
    None: "upstream 503 from payments-svc",
}


class Host:
    def __init__(self, name, rng, fault_rate):
        self.name, self.rng, self.fault_rate = name, rng, fault_rate
        self.phase = rng.uniform(0, 2 * math.pi)
        self.down = 0
        self._restart()

    def _restart(self):
        r = self.rng
        self.cpu0, self.p990 = r.uniform(20, 45), r.uniform(60, 120)
        self.mem, self.disk0 = r.uniform(35, 60), r.uniform(30, 60)
        self.disk, self.pool = self.disk0, r.uniform(0.3, 0.6)
        self.fault, self.rate, self.spike = None, 0.0, 0

    def inject(self, kind, ticks_to_fail=None):
        # 20% of faults are fast (20-60 min): hard to catch early, kept on purpose.
        r = self.rng
        T = ticks_to_fail or (r.uniform(20, 60) if r.random() < 0.2 else r.uniform(90, 480))
        level = {"memory_leak": self.mem, "disk_fill": self.disk, "conn_pool": self.pool}[kind]
        self.fault, self.rate = kind, (LIMIT[kind] - level) / T

    def step(self, t):
        """Advance one tick. Returns (row, cause, failure); row is None while the host is down."""
        r = self.rng
        if self.down:
            self.down -= 1
            if not self.down:
                self._restart()
            return None, None, None
        if self.fault is None and r.random() < self.fault_rate:
            self.inject(FAULTS[r.integers(len(FAULTS))])
        if not self.spike and r.random() < 1 / 300:
            self.spike = int(r.integers(5, 20))
        spiking = self.spike > 0
        self.spike = max(0, self.spike - 1)

        kind = self.fault
        if kind == "memory_leak":
            self.mem += self.rate
        if kind == "conn_pool":
            self.pool += self.rate
        if kind == "disk_fill":
            self.disk += self.rate
        else:  # slow benign growth, reset by log rotation
            self.disk += 0.02
            if self.disk > self.disk0 + 15:
                self.disk = self.disk0

        pm = max(0.0, (self.mem - 80) / 18)
        pd = max(0.0, (self.disk - 85) / 14)
        pp = max(0.0, (self.pool - 0.75) / 0.25)
        cpu = self.cpu0 + 10 * math.sin(2 * math.pi * t / 1440 + self.phase) + r.normal(0, 4) + 40 * spiking
        p99 = self.p990 * (1 + cpu / 200) * (1 + 2 * pm + 1.5 * pd + 3 * pp**2 + spiking) * r.lognormal(0, 0.1)
        warns = r.poisson(0.2 + 4 * (pm + pd + pp) + 3 * spiking)
        errors = r.poisson(0.05 + 10 * max(pm, pd, pp) ** 3)
        row = (min(max(cpu, 0), 100), min(self.mem + r.normal(0, 1.5), 100), min(self.disk, 100), p99, warns, errors)

        failure = None
        if kind and {"memory_leak": self.mem, "disk_fill": self.disk, "conn_pool": self.pool}[kind] >= LIMIT[kind]:
            failure, self.down = FAILURE[kind], int(r.integers(5, 15))
        cause = kind if kind and max(pm, pd, pp) > 0 else None
        return row, cause, failure


def simulate(n_hosts, ticks, seed=0, fault_rate=1 / 2880):
    """Offline fleet run. Returns data (hosts, ticks, len(COLS)) with NaN while down, and [(host, tick, kind)] failures."""
    rng = np.random.default_rng(seed)
    data = np.full((n_hosts, ticks, len(COLS)), np.nan, dtype=np.float32)
    failures = []
    for h in range(n_hosts):
        host = Host(f"h{h}", rng, fault_rate)
        for t in range(ticks):
            row, _, failure = host.step(t)
            if row is not None:
                data[h, t] = row
            if failure:
                failures.append((h, t, failure))
    return data, failures


def stream(n_hosts, fault_rate, inject, brokers, info_per_tick):
    from confluent_kafka import Producer

    producer = Producer({
        "bootstrap.servers": brokers,
        "linger.ms": 20,
        "compression.type": "lz4",
        "queue.buffering.max.messages": 1_000_000,
    })
    rng = np.random.default_rng()
    hosts = {f"web-{i:02d}": Host(f"web-{i:02d}", rng, fault_rate) for i in range(n_hosts)}
    for spec in inject:
        name, kind = spec.split(":")
        hosts[name].inject(kind)

    t, sent = 0, 0
    while True:
        now = int(time.time())
        for host in hosts.values():
            row, cause, failure = host.step(t)
            if row is None:
                continue
            cpu, mem, disk, p99, warns, errors = row
            vals = dict(cpu=cpu, mem=mem, disk=disk, p99=p99)
            producer.produce("metrics", json.dumps(
                {"ts": now, "host": host.name, "cpu": round(cpu, 2), "mem": round(mem, 2), "disk": round(disk, 2), "p99_ms": round(p99, 1)}))
            logs = [("WARN", WARN_MSG[cause])] * int(warns) + [("ERROR", ERROR_MSG[cause])] * int(errors)
            logs += [("INFO", "GET /api/orders 200 {p99:.0f}ms")] * info_per_tick
            for level, msg in logs:
                producer.produce("logs", json.dumps({"ts": now, "host": host.name, "level": level, "msg": msg.format(**vals)}))
            sent += 1 + len(logs)
            if failure:
                producer.produce("incidents", json.dumps({"ts": now, "host": host.name, "kind": failure}))
                print(f"{time.strftime('%X')} {host.name} FAILED: {failure}", flush=True)
        producer.poll(0)
        t += 1
        if t % 30 == 0:
            print(f"{time.strftime('%X')} tick {t}, {sent} events sent", flush=True)
        time.sleep(max(0.0, now + 1 - time.time()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hosts", type=int, default=20)
    p.add_argument("--fault-rate", type=float, default=1 / 1440, help="per host per tick")
    p.add_argument("--inject", action="append", default=[], metavar="HOST:FAULT",
                   help=f"start a fault at t=0, e.g. web-03:memory_leak ({', '.join(FAULTS)})")
    p.add_argument("--info-per-tick", type=int, default=20, help="INFO log lines per host per tick (raise to load-test)")
    p.add_argument("--brokers", default="localhost:19092")
    a = p.parse_args()
    stream(a.hosts, a.fault_rate, a.inject, a.brokers, a.info_per_tick)
