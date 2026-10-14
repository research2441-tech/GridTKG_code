"""
Design ablation of GridTKG on one dataset:
  full        - materialised busNeighbor relation + slot-aligned dense chunks
  no-derived  - neighbours reconstructed at query time from the explicit
                endpoint facts (fromBus/toBus/hvBus/lvBus) and the in-service
                intervals implied by 'affects' facts (vectorised CSR joins)
  row-log     - measurements kept as a (sensor, ts, value) row log sorted by
                sensor and timestamp, accessed by binary search per sensor
Same parameters, warm-up and 30 runs as bench_db.py.
"""
import argparse
import json
import os
import time

import numpy as np

from bench_db import Workload, METHOD, run_query, RUNS, CALLS, canon, same
from gridtkg_store import GridTKG, load_dataset, materialise, _PredPartition
from ontology import PRED_ID, ENDPOINT_PREDS, INF_T

QUERIES_TOPO = ["O3", "A3", "A5"]
QUERIES_TS = ["A1", "A2", "M1", "M2"]


class _EndpointNeighbours:
    """Replaces the busNeighbor partition by a join over endpoint and affects facts."""

    def __init__(self, store):
        self.s = store
        self.ep = [store.parts[PRED_ID[p]] for p in ENDPOINT_PREDS]
        self.aff = store.parts[PRED_ID["affects"]]
        b0, b1 = store.branch_block
        self.b0 = b0
        other = np.zeros(store.cap, np.int64)
        # endpoint lookup table built on the fly per call would be slower; we keep the
        # explicit facts in CSR form and resolve the opposite end with forward lookups
        self.d_out = {}

    def _rev(self, part, objs):
        a, b = part.r_ptr[objs], part.r_ptr[objs + 1]
        cnt = b - a
        tot = int(cnt.sum())
        if tot == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        idx = np.repeat(a - np.cumsum(np.r_[0, cnt[:-1]]), cnt) + np.arange(tot)
        rows = part.r_perm[idx]
        return part.src[rows], np.repeat(objs, cnt)

    def _fwd_one(self, part, subj):
        a, b = part.f_ptr[subj], part.f_ptr[subj + 1]
        cnt = b - a
        out = np.full(len(subj), -1, np.int64)
        has = cnt > 0
        out[has] = part.f_dst[a[has]]
        return out

    def neighbours_with_via(self, frontier, t):
        brs, at = [], []
        for part in self.ep:
            b, o = self._rev(part, frontier)
            brs.append(b); at.append(o)
        br = np.concatenate(brs)
        at = np.concatenate(at)
        if len(br) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        # in service at t: no affects fact with vf <= t < vt
        a, b = self.aff.r_ptr[br], self.aff.r_ptr[br + 1]
        cnt = b - a
        out = np.zeros(len(br), bool)
        tot = int(cnt.sum())
        if tot:
            idx = np.repeat(a - np.cumsum(np.r_[0, cnt[:-1]]), cnt) + np.arange(tot)
            rows = self.aff.r_perm[idx]
            hit = (self.aff.vf[rows] <= t) & (t < self.aff.vt[rows])
            owner = np.repeat(np.arange(len(br)), cnt)
            out[owner[hit]] = True
        for k, (s, vf, vt) in [(k, self.aff.in_subjects(int(x))) for k, x in enumerate(br) if int(x) in self.aff.d_in]:
            if np.any((vf <= t) & (t < vt)):
                out[k] = True
        keep = ~out
        br, at = br[keep], at[keep]
        ends = np.stack([self._fwd_one(p, br) for p in self.ep])   # (4, n)
        e1 = np.where(ends[0] >= 0, ends[0], ends[2])
        e2 = np.where(ends[1] >= 0, ends[1], ends[3])
        nb = np.where(e1 == at, e2, e1)
        return nb, br

    def neighbours(self, frontier, t):
        return self.neighbours_with_via(frontier, t)[0]


class NoDerivedStore(GridTKG):
    def enable(self):
        self._ep = _EndpointNeighbours(self)
        self.parts_bn_backup = self.parts[PRED_ID["busNeighbor"]]

    def bus_neighbours_now(self, bus, t):
        nb, via = self._ep.neighbours_with_via(np.array([bus], np.int64), t)
        mon = self.parts[PRED_ID["monitors"]]
        out = []
        for o, b in zip(nb.tolist(), via.tolist()):
            s = mon.in_subjects(b)[0]
            out.append((o, b, float(self.cur_val[s[0] - self.sensor0]) if len(s) else None))
        return out

    def khop(self, bus, k, t):
        real = self.parts[PRED_ID["busNeighbor"]]
        self.parts[PRED_ID["busNeighbor"]] = self._ep
        try:
            return GridTKG.khop(self, bus, k, t)
        finally:
            self.parts[PRED_ID["busNeighbor"]] = real

    def shortest_hops(self, a, b, t, max_depth=30):
        real = self.parts[PRED_ID["busNeighbor"]]
        self.parts[PRED_ID["busNeighbor"]] = self._ep
        try:
            return GridTKG.shortest_hops(self, a, b, t, max_depth)
        finally:
            self.parts[PRED_ID["busNeighbor"]] = real


class RowLogStore(GridTKG):
    def enable(self, V):
        ns, T = V.shape
        self.r_sensor = np.repeat(np.arange(ns), T)
        self.r_ts = np.tile(np.arange(T, dtype=np.int64), ns)
        self.r_val = V.reshape(-1).copy()
        self.r_ptr = np.arange(ns + 1, dtype=np.int64) * T
        self.chunks = None

    def history(self, sensors, t0, t1):
        k = np.asarray(sensors, np.int64) - self.sensor0
        out = np.full((len(k), t1 - t0), np.nan, np.float32)
        for i, s in enumerate(k.tolist()):
            a, b = self.r_ptr[s], self.r_ptr[s + 1]
            ts = self.r_ts[a:b]
            lo = a + np.searchsorted(ts, t0)
            hi = a + np.searchsorted(ts, t1)
            out[i, self.r_ts[lo:hi] - t0] = self.r_val[lo:hi]
        return out


def build(kind, d, derived, path):
    cls = dict(full=GridTKG, **{"no-derived": NoDerivedStore, "row-log": RowLogStore})[kind]
    s = cls(path)
    s.bulk_load(d, derived)
    s.finalize_indexes(d["meta_blocks"]["Branch"])
    if kind == "no-derived":
        s.enable()
    if kind == "row-log":
        s.enable(d["V"])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="KG-L")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    d = load_dataset(os.path.join(a.data, f"{a.tag}.npz"))
    derived = materialise(d)
    res = {}
    agree = {}
    for kind, qs in (("full", QUERIES_TOPO + QUERIES_TS), ("no-derived", QUERIES_TOPO), ("row-log", QUERIES_TS)):
        wal = f"/tmp/abl_{kind}.wal"
        if os.path.exists(wal):
            os.remove(wal)
        s = build(kind, d, derived, wal)
        res[kind] = {}
        for q in qs:
            wl = Workload(d)
            rng = np.random.default_rng(10_000)
            agree.setdefault(q, {})[kind] = [canon(q, getattr(s, METHOD[q])(*wl.params(q, rng))) for _ in range(5)]
            run_query(s, wl, q, 999, CALLS[q])
            res[kind][q] = [run_query(s, wl, q, 1000 + r, CALLS[q]) for r in range(RUNS)]
            print(f"{kind:11s} {q}: {np.mean(res[kind][q]):9.3f} ms", flush=True)
    res["_agreement"] = {q: {k: all(same(x, y) for x, y in zip(v, agree[q]["full"])) for k, v in kinds.items()}
                         for q, kinds in agree.items()}
    json.dump(res, open(os.path.join(a.out, f"ablation_{a.tag}.json"), "w"))


if __name__ == "__main__":
    main()
