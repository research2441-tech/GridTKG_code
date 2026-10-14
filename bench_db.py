"""
System-level benchmark: GridTKG versus PostgreSQL, DuckDB and Kuzu+DuckDB on
the operational (O), analytical (A), maintenance (M) and data-storage (S)
workloads.  Every workload is executed for one discarded warm-up run followed
by 30 measured runs; each run issues a fixed number of calls with parameters
drawn from a run-specific seed, so all systems receive identical requests.

usage: python bench_db.py --tag KG-M --systems gridtkg,postgres,duckdb,polyglot
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

from baselines import PgSystem, DuckSystem, PolyglotSystem
from gridtkg_store import GridTKG, load_dataset, materialise
from ontology import CLASS_ID

RUNS = int(os.environ.get("BENCH_RUNS", 30))
CALLS = dict(O1=50, O2=50, O3=50, O4=20, A1=10, A2=3, A3=10, A4=10, A5=5,
             M1=20, M2=5, M3=5, S1=5, S2=10)
READ_ORDER = ["O2", "O3", "A1", "A2", "A3", "A4", "A5", "M1", "M2", "M3"]
WRITE_ORDER = ["O1", "S1", "O4", "S2"]


class TkgSystem:
    def __init__(self, workdir):
        self.dir = os.path.join(workdir, "gridtkg")

    def start(self, fresh=True):
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        self.s = GridTKG(os.path.join(self.dir, "wal.log"))
        self.version = f"GridTKG 1.0 (Python, NumPy {np.__version__})"

    def stop(self):
        self.s.sync()
        self.s.wal.close()

    def load(self, d, derived):
        self.s.bulk_load(d, derived)
        self.s.finalize_indexes(d["meta_blocks"]["Branch"])

    def __getattr__(self, k):
        return getattr(self.s, k)

    def ingest_slot(self, slot, values, sensors):
        self.s.ingest_slot(slot, values)

    def history(self, sensors, t0, t1):
        return self.s.history(sensors, t0, t1)


def make_system(name, workdir):
    os.makedirs(workdir, exist_ok=True)
    return dict(gridtkg=TkgSystem, postgres=PgSystem, duckdb=DuckSystem, polyglot=PolyglotSystem)[name](workdir)


# ---------------------------------------------------------------------------
class Workload:
    def __init__(self, d):
        m = d["meta"]
        b = d["meta_blocks"]
        self.T0 = m["T"]
        self.spd = m["steps_per_day"]
        self.dt_h = m["step_min"] / 60.0
        self.sensors = np.arange(*b["Sensor"])
        self.subs = np.arange(*b["Substation"])
        self.buses = np.arange(*b["Bus"])
        self.br0 = b["Branch"][0]
        et = d["etype"]
        st = d["sens_target"]
        sk = d["sens_kind"]
        self.load_subs = np.unique(d["e_sub"][st[sk == 1]] + b["Substation"][0])
        trs = np.where(d["is_trafo"])[0] + self.br0
        tr_sens = np.where((sk == 0) & np.isin(st, trs))[0] + b["Sensor"][0]
        self.tr_sens = tr_sens
        self.nonbridge = np.where(~d["bridge"])[0] + self.br0
        # loading threshold for M2/M3: 80th percentile of 24-h mean transformer loading
        F = d["F_true"][d["is_trafo"]]
        R = d["rating"][d["is_trafo"]][:, None]
        ld = np.abs(F[:, -self.spd:]).mean(1) / R[:, 0]
        self.thr = float(np.quantile(ld, 0.8))
        self.kv = d["e_kv"]
        self.n_reg = int(d["e_reg"][b["Region"][0]:b["Region"][1]].size)
        self.next_slot = self.T0
        self.last_vals = np.nan_to_num(d["V"][:, -1]).astype(np.float32)

    def params(self, q, rng):
        T0, spd = self.T0, self.spd
        if q == "O1":
            self.next_slot += 1
            return (int(rng.choice(self.sensors)), int(self.next_slot), float(rng.normal(50, 10)))
        if q == "O2":
            return (int(rng.choice(self.subs)),)
        if q == "O3":
            return (int(rng.choice(self.buses)), int(rng.integers(0, T0)))
        if q == "O4":
            t0 = int(rng.integers(0, T0))
            return (int(rng.choice(self.nonbridge)), t0, t0 + int(rng.integers(4, 33)))
        if q == "A1":
            t0 = int(rng.integers(0, T0 - 7 * spd + 1))
            return (int(rng.choice(self.load_subs)), t0, t0 + 7 * spd, spd)
        if q == "A2":
            return (0, T0, spd, self.dt_h)
        if q == "A3":
            return (int(rng.choice(self.buses)), 3, int(rng.integers(0, T0)))
        if q == "A4":
            return (20,)
        if q == "A5":
            a, b = rng.choice(self.buses, 2, replace=False)
            return (int(a), int(b), int(rng.integers(0, T0)))
        if q == "M1":
            return ([int(rng.choice(self.tr_sens))], T0 - 7 * spd, T0)
        if q in ("M2", "M3"):
            return (int(rng.integers(spd, T0 + 1)), spd, self.thr)
        if q == "S1":
            self.next_slot += 1
            return (int(self.next_slot), (self.last_vals * (1 + rng.normal(0, 0.01, len(self.last_vals)))).astype(np.float32),
                    self.sensors)
        if q == "S2":
            bus = int(rng.choice(self.buses))
            kv = float(self.kv[bus])
            return (int(rng.integers(0, self.n_reg)), bus, kv, kv / 2.0, 100.0, int(T0))
        raise KeyError(q)


METHOD = dict(O1="ingest", O2="substation_status", O3="bus_neighbours_now", O4="switch_out",
              A1="load_profile", A2="region_daily_energy", A3="khop", A4="top_substations",
              A5="shortest_hops", M1="history", M2="overloaded_transformers", M3="overload_context",
              S1="ingest_slot", S2="add_substation_bay")


def canon(q, r):
    """Canonical, comparable form of a query result."""
    if q == "O2":
        s, v = r
        return [np.asarray(s, np.int64), np.round(np.nan_to_num(np.asarray(v, np.float64), nan=-9e9), 2)]
    if q == "O3":
        return sorted((a, b, None if c is None or (isinstance(c, float) and np.isnan(c)) else round(c, 2)) for a, b, c in r)
    if q in ("A1", "A2", "M1"):
        return np.round(np.nan_to_num(np.asarray(r, np.float64), nan=-9e9), 1)
    if q == "A3":
        return (r[0], round(r[1], 1))
    if q == "A4" or q == "A5":
        return r
    if q == "M2":
        return [list(map(int, r[0])), np.round(np.asarray(r[1], np.float64), 4)]
    if q == "M3":
        return [(int(a), int(b), round(c, 4)) for a, b, c in r]
    return r


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and np.allclose(a, b, rtol=2e-3, atol=0.15)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= max(0.15, 2e-3 * abs(a))
    return a == b


def run_query(sysobj, wl, q, seed, n_calls):
    rng = np.random.default_rng(seed)
    fn = getattr(sysobj, METHOD[q])
    lat = []
    for _ in range(n_calls):
        p = wl.params(q, rng)
        t = time.perf_counter()
        fn(*p)
        lat.append((time.perf_counter() - t) * 1e3)
    return lat


def bench_system(name, d, derived, workdir, out):
    wl = Workload(d)
    s = make_system(name, os.path.join(workdir, name))
    s.start(True)
    t = time.perf_counter()
    s.load(d, derived)
    load_s = time.perf_counter() - t
    foot, wal = s.footprint_bytes()
    res = dict(system=name, version=s.version, load_seconds=load_s, footprint_bytes=foot, wal_bytes=wal, queries={})
    # validation results (canonical) on fixed parameters, read-only queries
    val = {}
    for q in READ_ORDER:
        rng = np.random.default_rng(10_000)
        val[q] = []
        for _ in range(5):
            p = wl.params(q, rng)
            val[q].append(canon(q, getattr(s, METHOD[q])(*p)))
    out["_validation"][name] = val
    for q in READ_ORDER + WRITE_ORDER:
        n = CALLS[q]
        run_lat = []
        t_q = time.perf_counter()
        run_query(s, wl, q, 999, n)          # warm-up (discarded)
        for r in range(RUNS):
            run_lat.append(run_query(s, wl, q, 1000 + r, n))
        res["queries"][q] = dict(calls_per_run=n, lat_ms=run_lat, wall_s=time.perf_counter() - t_q)
        print(f"  {name:9s} {q}: mean {np.mean(run_lat):9.3f} ms   ({time.perf_counter() - t_q:6.1f} s)", flush=True)
    # mixed HTAP workload: 90 % operational, 10 % analytical
    mix = ["O1"] * 40 + ["O2"] * 25 + ["O3"] * 25 + ["A1"] * 4 + ["A3"] * 3 + ["M2"] * 3
    mres = []
    for r in range(RUNS + 1):
        rng = np.random.default_rng(5000 + r)
        seq = rng.choice(mix, 200)
        lat = {"oltp": [], "olap": []}
        t0 = time.perf_counter()
        for q in seq:
            p = wl.params(q, rng)
            t = time.perf_counter()
            getattr(s, METHOD[q])(*p)
            lat["oltp" if q[0] == "O" else "olap"].append((time.perf_counter() - t) * 1e3)
        el = time.perf_counter() - t0
        if r > 0:
            mres.append(dict(ops_per_s=len(seq) / el, oltp_p95=float(np.percentile(lat["oltp"], 95)),
                             oltp_mean=float(np.mean(lat["oltp"])), olap_mean=float(np.mean(lat["olap"]))))
    res["mixed"] = mres
    print(f"  {name:9s} MIX: {np.mean([m['ops_per_s'] for m in mres]):8.1f} ops/s", flush=True)
    foot2, wal2 = s.footprint_bytes()
    res["footprint_after_bytes"] = foot2
    res["wal_after_bytes"] = wal2
    s.stop()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="KG-S")
    ap.add_argument("--data", default="data")
    ap.add_argument("--systems", default="gridtkg,postgres,duckdb,polyglot")
    ap.add_argument("--work", default="/tmp/gridtkg_bench")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    d = load_dataset(os.path.join(a.data, f"{a.tag}.npz"))
    t = time.perf_counter()
    derived = materialise(d)
    mat_s = time.perf_counter() - t
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"db_{a.tag}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out.setdefault("_validation", {})
    out["_meta"] = dict(d["meta"], materialise_seconds=mat_s,
                        n_derived={k: int(len(v)) for k, v in derived.items()})
    for name in a.systems.split(","):
        print(f"[{a.tag}] {name}", flush=True)
        out[name] = bench_system(name, d, derived, a.work, out)
        # compare validation answers with GridTKG
        with open(path, "w") as fh:
            json.dump(out, fh, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    ref = out["_validation"].get("gridtkg")
    if ref is not None:
        agree = {}
        for name, val in out["_validation"].items():
            agree[name] = {q: [bool(same(x, y)) for x, y in zip(ref[q], val[q])] for q in ref}
        out["_agreement"] = agree
    with open(path, "w") as fh:
        json.dump(out, fh, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


if __name__ == "__main__":
    main()
