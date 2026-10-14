"""
GridTKG: an in-process temporal knowledge-graph store for unified OLTP/OLAP
on power-grid data, written from scratch in Python/NumPy (no external DBMS).

Physical design
---------------
* Dictionary-encoded entities in typed columns (etype, substation, region, kV,
  rating, age).
* Facts are temporal quadruples (s, p, o, [vf, vt)) with an optional `via`
  qualifier.  Each predicate has a read-optimised base partition held in CSR
  form (forward and reverse) plus a small write-optimised delta partition;
  the delta is merged into the base when it exceeds `merge_threshold`.
* Measurements are slot-aligned: chunk c holds a dense float32 matrix
  (n_sensors x L) for slots [cL, (c+1)L).  The latest value of every sensor is
  kept in a current-state vector for O(1) operational reads.
* Durability: every committed transaction is appended to a write-ahead log;
  the log is flushed and fsync-ed by group commit (default every 10 ms).
* Consistency: semantic constraints from ontology.py are checked before a
  transaction is written to the log; a transaction is applied in memory only
  after its log record is written (atomic publication, single writer).
* Derived relations (busNeighbor, adjacentTo, substationLink) are materialised
  by the rules in ontology.RULES, with temporal validity inherited from the
  in-service intervals of the underlying branch.
"""
import os
import struct
import time

import numpy as np

from ontology import (CLASS_ID, ID_CLASS, PRED, PRED_ID, INF_T, ENDPOINT_PREDS,
                      ConstraintViolation, check_typing, check_interval, is_a)

P_BN = PRED_ID["busNeighbor"]
P_ADJ = PRED_ID["adjacentTo"]
P_SL = PRED_ID["substationLink"]


class _PredPartition:
    """Base (CSR) + delta partition of one predicate."""

    def __init__(self, n_nodes):
        self.src = np.zeros(0, np.int64)
        self.dst = np.zeros(0, np.int64)
        self.vf = np.zeros(0, np.int64)
        self.vt = np.zeros(0, np.int64)
        self.via = np.zeros(0, np.int64)
        self.d_src, self.d_dst, self.d_vf, self.d_vt, self.d_via = [], [], [], [], []
        self.d_out, self.d_in = {}, {}
        self.n_nodes = n_nodes
        self._build()

    def _build(self):
        n = self.n_nodes
        order = np.argsort(self.src, kind="stable")
        self.f_perm = order
        self.f_ptr = np.zeros(n + 1, np.int64)
        np.add.at(self.f_ptr, self.src + 1, 1)
        self.f_ptr = np.cumsum(self.f_ptr)
        order_r = np.argsort(self.dst, kind="stable")
        self.r_perm = order_r
        self.r_ptr = np.zeros(n + 1, np.int64)
        np.add.at(self.r_ptr, self.dst + 1, 1)
        self.r_ptr = np.cumsum(self.r_ptr)
        # CSR-ordered copies for vectorised traversal
        self.f_dst = self.dst[order]
        self.f_vf = self.vf[order]
        self.f_vt = self.vt[order]
        self.f_via = self.via[order]

    def bulk_load(self, src, dst, vf, vt, via):
        self.src, self.dst = src.astype(np.int64), dst.astype(np.int64)
        self.vf, self.vt, self.via = vf.astype(np.int64), vt.astype(np.int64), via.astype(np.int64)
        self._build()

    def grow(self, n_nodes):
        if n_nodes > self.n_nodes:
            self.n_nodes = n_nodes
            self.f_ptr = np.r_[self.f_ptr, np.full(n_nodes - len(self.f_ptr) + 1, self.f_ptr[-1])]
            self.r_ptr = np.r_[self.r_ptr, np.full(n_nodes - len(self.r_ptr) + 1, self.r_ptr[-1])]

    def add(self, s, o, vf, vt, via):
        k = len(self.d_src)
        self.d_src.append(s); self.d_dst.append(o); self.d_vf.append(vf); self.d_vt.append(vt); self.d_via.append(via)
        self.d_out.setdefault(s, []).append(k)
        self.d_in.setdefault(o, []).append(k)
        return ("d", k)

    def merge(self):
        if not self.d_src:
            return
        self.src = np.r_[self.src, np.array(self.d_src, np.int64)]
        self.dst = np.r_[self.dst, np.array(self.d_dst, np.int64)]
        self.vf = np.r_[self.vf, np.array(self.d_vf, np.int64)]
        self.vt = np.r_[self.vt, np.array(self.d_vt, np.int64)]
        self.via = np.r_[self.via, np.array(self.d_via, np.int64)]
        self.d_src, self.d_dst, self.d_vf, self.d_vt, self.d_via = [], [], [], [], []
        self.d_out, self.d_in = {}, {}
        self._build()

    def n_delta(self):
        return len(self.d_src)

    # -- point lookups ------------------------------------------------------
    def out_rows(self, s):
        """(objects, vf, vt, via, handles) of facts with subject s."""
        a, b = self.f_ptr[s], self.f_ptr[s + 1]
        rows = self.f_perm[a:b]
        o, vf, vt, via = self.dst[rows], self.vf[rows], self.vt[rows], self.via[rows]
        dk = self.d_out.get(s)
        handles = [("b", int(r)) for r in rows]
        if dk:
            o = np.r_[o, [self.d_dst[k] for k in dk]]
            vf = np.r_[vf, [self.d_vf[k] for k in dk]]
            vt = np.r_[vt, [self.d_vt[k] for k in dk]]
            via = np.r_[via, [self.d_via[k] for k in dk]]
            handles += [("d", k) for k in dk]
        return o, vf, vt, via, handles

    def in_subjects(self, o):
        a, b = self.r_ptr[o], self.r_ptr[o + 1]
        rows = self.r_perm[a:b]
        s, vf, vt = self.src[rows], self.vf[rows], self.vt[rows]
        dk = self.d_in.get(o)
        if dk:
            s = np.r_[s, [self.d_src[k] for k in dk]]
            vf = np.r_[vf, [self.d_vf[k] for k in dk]]
            vt = np.r_[vt, [self.d_vt[k] for k in dk]]
        return s, vf, vt

    def in_subjects_many(self, objs):
        """Vectorised reverse lookup for an array of objects (valid now)."""
        objs = np.asarray(objs, np.int64)
        a, b = self.r_ptr[objs], self.r_ptr[objs + 1]
        cnt = b - a
        if cnt.sum() == 0:
            out = np.zeros(0, np.int64)
        else:
            idx = np.repeat(a - np.cumsum(np.r_[0, cnt[:-1]]), cnt) + np.arange(cnt.sum())
            rows = self.r_perm[idx]
            out = self.src[rows[self.vt[rows] >= INF_T]]
        if self.d_in:
            extra = [self.d_src[k] for o in objs.tolist() for k in self.d_in.get(o, ()) if self.d_vt[k] >= INF_T]
            if extra:
                out = np.r_[out, extra]
        return out

    def close(self, handle, vt):
        kind, k = handle
        if kind == "b":
            self.vt[k] = vt
            self._close_csr(k, vt)
        else:
            self.d_vt[k] = vt

    def _close_csr(self, row, vt):
        if not hasattr(self, "_f_inv") or len(self._f_inv) != len(self.f_perm):
            self._f_inv = np.empty_like(self.f_perm)
            self._f_inv[self.f_perm] = np.arange(len(self.f_perm))
        self.f_vt[self._f_inv[row]] = vt

    def neighbours(self, frontier, t):
        """Objects reachable in one step from `frontier` through facts valid at t."""
        a, b = self.f_ptr[frontier], self.f_ptr[frontier + 1]
        cnt = b - a
        tot = int(cnt.sum())
        if tot:
            idx = np.repeat(a - np.cumsum(np.r_[0, cnt[:-1]]), cnt) + np.arange(tot)
            ok = (self.f_vf[idx] <= t) & (t < self.f_vt[idx])
            nxt = self.f_dst[idx[ok]]
        else:
            nxt = np.zeros(0, np.int64)
        if self.d_out:
            extra = [self.d_dst[k] for s in frontier.tolist() for k in self.d_out.get(s, ())
                     if self.d_vf[k] <= t < self.d_vt[k]]
            if extra:
                nxt = np.r_[nxt, np.array(extra, np.int64)]
        return nxt


class GridTKG:
    def __init__(self, wal_path, slots_per_chunk=96, group_commit_ms=10.0, merge_threshold=4096):
        self.wal_path = wal_path
        self.wal = open(wal_path, "ab", buffering=1 << 20)
        self.gc = group_commit_ms / 1000.0
        self.last_sync = time.perf_counter()
        self.L = slots_per_chunk
        self.merge_threshold = merge_threshold
        self.n_entities = 0
        self.cap = 0
        self.cols = {}
        self.parts = {}
        self.chunks = []
        self.lsn = 0

    # ------------------------------------------------------------------ WAL
    def _log(self, payload: bytes):
        self.wal.write(struct.pack("<QI", self.lsn, len(payload)))
        self.wal.write(payload)
        self.lsn += 1
        now = time.perf_counter()
        if now - self.last_sync >= self.gc:
            self.wal.flush()
            os.fsync(self.wal.fileno())
            self.last_sync = now

    def sync(self):
        self.wal.flush()
        os.fsync(self.wal.fileno())

    # --------------------------------------------------------------- loading
    def bulk_load(self, d, derived):
        """d: npz arrays from grid_generator; derived: dict pred -> (n,5) array."""
        n = len(d["etype"])
        self._alloc(n + 4096)
        self.n_entities = n
        for c in ("etype", "e_sub", "e_reg", "e_kv", "e_rating", "e_age"):
            self.cols[c][:n] = d[c]
        for p in PRED_ID:
            self.parts[PRED_ID[p]] = _PredPartition(self.cap)
            arr = d[f"fact_{p}"] if f"fact_{p}" in d else None
            if p in derived:
                arr = derived[p]
            if arr is None or len(arr) == 0:
                continue
            via = arr[:, 4] if arr.shape[1] > 4 else np.full(len(arr), -1)
            self.parts[PRED_ID[p]].bulk_load(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], via)
        # sensors
        s0, s1 = d["meta_blocks"]["Sensor"]
        self.sensor0 = s0
        self.n_sensor = s1 - s0
        self.sens_target = np.asarray(d["sens_target"], np.int64)
        V = d["V"]
        T = V.shape[1]
        nchunk = int(np.ceil(T / self.L))
        self.chunks = []
        for c in range(nchunk):
            m = np.full((self.n_sensor, self.L), np.nan, np.float32)
            seg = V[:, c * self.L:(c + 1) * self.L]
            m[:, :seg.shape[1]] = seg
            self.chunks.append(m)
        self.T = T
        self.cur_val = V[:, -1].copy()
        self.cur_ts = np.full(self.n_sensor, T - 1, np.int64)
        self.sync()

    def _alloc(self, cap):
        specs = dict(etype=np.int16, e_sub=np.int32, e_reg=np.int32, e_kv=np.float32,
                     e_rating=np.float32, e_age=np.float32)
        for c, dt in specs.items():
            new = np.zeros(cap, dt)
            if c in self.cols:
                new[:self.cap] = self.cols[c][:self.cap]
            self.cols[c] = new
        self.cap = cap
        for p in self.parts.values():
            p.grow(cap)

    # ----------------------------------------------------------- write path
    def _new_entity(self, cls, sub=-1, reg=-1, kv=0.0, rating=0.0, age=0.0):
        if self.n_entities >= self.cap:
            self._alloc(self.cap * 2)
        e = self.n_entities
        self.n_entities += 1
        c = self.cols
        c["etype"][e], c["e_sub"][e], c["e_reg"][e] = CLASS_ID[cls], sub, reg
        c["e_kv"][e], c["e_rating"][e], c["e_age"][e] = kv, rating, age
        return e

    def cls_of(self, e):
        return ID_CLASS[int(self.cols["etype"][e])]

    def _validate_fact(self, pred, s, o, vf, vt, pending=()):
        check_typing(pred, self.cls_of(s), self.cls_of(o))
        check_interval(vf, vt)
        if PRED[pred].functional:
            part = self.parts[PRED_ID[pred]]
            _, f_vf, f_vt, _, _ = part.out_rows(s)
            if len(f_vf) and np.any((f_vf < vt) & (vf < f_vt)):
                raise ConstraintViolation(f"cardinality violation: {pred}({s}, .) already valid")
            for (p2, s2, _, vf2, vt2) in pending:
                if p2 == pred and s2 == s and vf2 < vt and vf < vt2:
                    raise ConstraintViolation(f"cardinality violation: {pred}({s}, .) twice")

    def _add_fact(self, pred, s, o, vf, vt, via=-1):
        part = self.parts[PRED_ID[pred]]
        part.add(s, o, vf, vt, via)
        if part.n_delta() >= self.merge_threshold:
            part.merge()

    def ingest(self, sensor, slot, value):
        """O1: append one measurement and update the current state (one transaction)."""
        k = sensor - self.sensor0
        self._log(struct.pack("<Bqqf", 1, sensor, slot, value))
        c = slot // self.L
        while c >= len(self.chunks):
            self.chunks.append(np.full((self.n_sensor, self.L), np.nan, np.float32))
        self.chunks[c][k, slot % self.L] = value
        if slot >= self.cur_ts[k]:
            self.cur_val[k] = value
            self.cur_ts[k] = slot
        if slot >= self.T:
            self.T = slot + 1

    def ingest_slot(self, slot, values):
        """S1: one slot for all sensors (bulk, one transaction)."""
        self._log(struct.pack("<Bq", 2, slot) + values.astype(np.float32).tobytes())
        c = slot // self.L
        while c >= len(self.chunks):
            self.chunks.append(np.full((self.n_sensor, self.L), np.nan, np.float32))
        self.chunks[c][:, slot % self.L] = values
        self.cur_val[:] = values
        self.cur_ts[:] = slot
        self.T = max(self.T, slot + 1)

    def switch_out(self, branch, t0, t1):
        """O4: record a forced outage of `branch` in [t0, t1) as one transaction:
        new FaultEvent entity, affects fact, and temporal split of the derived
        busNeighbor facts carried by the branch."""
        if not is_a(self.cls_of(branch), "Branch"):
            raise ConstraintViolation("affects range violation")
        check_interval(t0, t1)
        self._log(struct.pack("<Bqqq", 3, branch, t0, t1))
        fe = self._new_entity("FaultEvent", sub=int(self.cols["e_sub"][branch]), reg=int(self.cols["e_reg"][branch]))
        self._add_fact("affects", fe, branch, t0, t1)
        bn = self.parts[P_BN]
        ends = self.branch_ends(branch)
        for x, y in ((ends[0], ends[1]), (ends[1], ends[0])):
            o, vf, vt, via, h = bn.out_rows(x)
            for j in np.where((via == branch) & (o == y) & (vf <= t0) & (t0 < vt))[0]:
                old_vt = int(vt[j])
                bn.close(h[j], t0)
                if t1 < old_vt:
                    self._add_fact("busNeighbor", x, y, t1, old_vt, branch)
        return fe

    def add_substation_bay(self, region, bus_attach, kv_hv, kv_lv, rating, t0):
        """S2: grow the topology (new substation, two buses, transformer, line to an
        existing bus, two sensors) in one validated transaction."""
        if abs(self.cols["e_kv"][bus_attach] - kv_hv) > 0.05 * kv_hv:
            raise ConstraintViolation("voltage mismatch on new line")
        if not kv_hv >= kv_lv:
            raise ConstraintViolation("transformer hv below lv")
        self._log(struct.pack("<Bqqfff", 4, region, bus_attach, kv_hv, kv_lv, rating))
        sub = self._new_entity("Substation", reg=region)
        self.cols["e_sub"][sub] = sub
        hv = self._new_entity("Bus", sub=sub, reg=region, kv=kv_hv)
        lv = self._new_entity("Bus", sub=sub, reg=region, kv=kv_lv)
        tr = self._new_entity("Transformer", sub=sub, reg=region, kv=kv_hv, rating=rating)
        ln = self._new_entity("Line", sub=sub, reg=region, kv=kv_hv, rating=rating)
        s1 = self._new_entity("SCADA", sub=sub, reg=region, rating=rating)
        s2 = self._new_entity("SCADA", sub=sub, reg=region, rating=rating)
        facts = [("locatedIn", sub, region), ("containedIn", hv, sub), ("containedIn", lv, sub),
                 ("containedIn", tr, sub), ("hvBus", tr, hv), ("lvBus", tr, lv),
                 ("fromBus", ln, hv), ("toBus", ln, bus_attach), ("monitors", s1, tr), ("monitors", s2, ln)]
        pending = []
        for p, s, o in facts:
            self._validate_fact(p, s, o, t0, INF_T, pending)
            pending.append((p, s, o, t0, INF_T))
        for p, s, o in facts:
            self._add_fact(p, s, o, t0, INF_T)
        for a, b, via in ((hv, lv, tr), (hv, bus_attach, ln)):
            self._add_fact("busNeighbor", a, b, t0, INF_T, via)
            self._add_fact("busNeighbor", b, a, t0, INF_T, via)
        return sub

    # ------------------------------------------------------------ read path
    def branch_ends(self, br):
        cls = self.cls_of(br)
        p1, p2 = ("fromBus", "toBus") if cls == "Line" else ("hvBus", "lvBus")
        a = self.parts[PRED_ID[p1]].out_rows(br)[0]
        b = self.parts[PRED_ID[p2]].out_rows(br)[0]
        return int(a[0]), int(b[0])

    def sensors_of(self, entities):
        s = self.parts[PRED_ID["monitors"]].in_subjects_many(entities)
        return s

    def substation_status(self, sub):
        """O2: current value of every sensor attached to equipment of substation `sub`."""
        cont = self.parts[PRED_ID["containedIn"]].in_subjects_many([sub])
        et = self.cols["etype"][cont]
        buses = cont[et == CLASS_ID["Bus"]]
        attached = self.parts[PRED_ID["connectedTo"]].in_subjects_many(buses)
        lines = self.parts[PRED_ID["fromBus"]].in_subjects_many(buses)
        equip = np.unique(np.r_[cont, attached, lines])
        sens = np.sort(self.sensors_of(equip))
        k = sens - self.sensor0
        return sens, self.cur_val[k]

    def bus_neighbours_now(self, bus, t):
        """O3: energised neighbours of `bus` at t with the current flow of the carrying branch."""
        o, vf, vt, via, _ = self.parts[P_BN].out_rows(bus)
        m = (vf <= t) & (t < vt)
        o, via = o[m], via[m]
        mon = self.parts[PRED_ID["monitors"]]
        vals = []
        for b in via.tolist():
            s = mon.in_subjects(b)[0]
            vals.append(float(self.cur_val[s[0] - self.sensor0]) if len(s) else None)
        return list(zip(o.tolist(), via.tolist(), vals))

    def history(self, sensors, t0, t1):
        k = np.asarray(sensors, np.int64) - self.sensor0
        c0, c1 = t0 // self.L, (t1 - 1) // self.L
        parts = [self.chunks[c][k] for c in range(c0, c1 + 1)]
        m = np.concatenate(parts, axis=1)
        off = t0 - c0 * self.L
        return m[:, off:off + (t1 - t0)]

    def load_sensors_of_substation(self, sub):
        cont = self.parts[PRED_ID["containedIn"]].in_subjects_many([sub])
        buses = cont[self.cols["etype"][cont] == CLASS_ID["Bus"]]
        att = self.parts[PRED_ID["connectedTo"]].in_subjects_many(buses)
        loads = att[self.cols["etype"][att] == CLASS_ID["Load"]]
        return np.sort(self.sensors_of(loads))

    def load_profile(self, sub, t0, t1, spd):
        """A1: mean hour-of-day profile (MW) of the total load of substation `sub`."""
        sens = self.load_sensors_of_substation(sub)
        if len(sens) == 0:
            return np.full(24, np.nan)
        H = self.history(sens, t0, t1)
        h = np.nansum(H, axis=0)
        ok = ~np.all(np.isnan(H), axis=0)
        slot_hour = ((np.arange(t0, t1) % spd) * 24) // spd
        num = np.bincount(slot_hour[ok], weights=h[ok], minlength=24)
        den = np.bincount(slot_hour[ok], minlength=24)
        prof = np.full(24, np.nan)
        prof[den > 0] = num[den > 0] / den[den > 0]
        return prof

    def region_daily_energy(self, t0, t1, spd, dt_h):
        """A2: daily energy (MWh) of all loads, grouped by region."""
        sm = np.where(self.cols["etype"][self.sensor0:self.sensor0 + self.n_sensor] == CLASS_ID["SmartMeter"])[0]
        sens = sm + self.sensor0
        reg = self.cols["e_reg"][sens]
        h = np.nan_to_num(self.history(sens, t0, t1))
        nd = (t1 - t0) // spd
        daily = h[:, :nd * spd].reshape(len(sens), nd, spd).sum(2) * dt_h
        nreg = int(reg.max()) + 1
        out = np.zeros((nreg, nd))
        np.add.at(out, reg, daily)
        return out

    def khop(self, bus, k, t):
        """A3: buses within k energised hops of `bus` at t and their connected current load (MW)."""
        bn = self.parts[P_BN]
        seen = np.zeros(self.cap, bool)
        seen[bus] = True
        frontier = np.array([bus], np.int64)
        for _ in range(k):
            nxt = bn.neighbours(frontier, t)
            nxt = np.unique(nxt)
            nxt = nxt[~seen[nxt]]
            if len(nxt) == 0:
                break
            seen[nxt] = True
            frontier = nxt
        reached = np.where(seen)[0]
        att = self.parts[PRED_ID["connectedTo"]].in_subjects_many(reached)
        loads = att[self.cols["etype"][att] == CLASS_ID["Load"]]
        sens = self.sensors_of(loads) - self.sensor0
        return len(reached), float(np.nansum(self.cur_val[sens]))

    def top_substations(self, k=20):
        """A4: substations with the largest number of distinct substation links."""
        sl = self.parts[P_SL]
        deg = np.diff(sl.f_ptr)[:self.n_entities].copy()
        for s, ks in sl.d_out.items():
            deg[s] += len(ks)
        idx = np.lexsort((np.arange(len(deg)), -deg))[:k]
        return list(zip(idx.tolist(), deg[idx].tolist()))

    def shortest_hops(self, a, b, t, max_depth=30):
        """A5: minimum number of energised hops between buses a and b at t (-1 if > max_depth)."""
        if a == b:
            return 0
        bn = self.parts[P_BN]
        seen = np.zeros(self.cap, bool)
        seen[a] = True
        frontier = np.array([a], np.int64)
        for d in range(1, max_depth + 1):
            nxt = np.unique(bn.neighbours(frontier, t))
            nxt = nxt[~seen[nxt]]
            if len(nxt) == 0:
                return -1
            if (nxt == b).any():
                return d
            seen[nxt] = True
            frontier = nxt
        return -1

    def branch_sensor(self, br):
        s = self.parts[PRED_ID["monitors"]].in_subjects(br)[0]
        return int(s[0]) if len(s) else -1

    def overloaded_transformers(self, t, window, thr):
        """M2: transformers whose mean |flow|/rating over [t-window, t) exceeds thr."""
        b0, b1 = self.branch_block
        trs = np.arange(b0, b1)[self.cols["etype"][b0:b1] == CLASS_ID["Transformer"]]
        sens = self._branch_sensor_vec[trs - b0]
        has = sens >= 0
        trs, sens = trs[has], sens[has]
        h = np.abs(self.history(sens, t - window, t))
        ld = np.nanmean(h, axis=1) / self.cols["e_rating"][trs]
        sel = ld > thr
        return trs[sel], ld[sel]

    def overload_context(self, t, window, thr):
        """M3: for every transformer returned by M2, the maximum loading of its adjacent monitored branches."""
        trs, _ = self.overloaded_transformers(t, window, thr)
        adj = self.parts[P_ADJ]
        out = []
        b0 = self.branch_block[0]
        for tr in trs.tolist():
            nb = adj.out_rows(tr)[0]
            if len(nb) == 0:
                continue
            sens = self._branch_sensor_vec[nb - b0]
            m = sens >= 0
            if not m.any():
                continue
            h = np.abs(self.history(sens[m], t - window, t))
            valid = ~np.all(np.isnan(h), axis=1)
            if not valid.any():
                continue
            ld = np.full(len(h), -np.inf)
            ld[valid] = np.nanmax(h[valid], axis=1) / self.cols["e_rating"][nb[m][valid]]
            j = int(np.argmax(ld))
            out.append((tr, int(nb[m][j]), float(ld[j])))
        return out

    def finalize_indexes(self, branch_block):
        self.branch_block = branch_block
        b0, b1 = branch_block
        v = np.full(b1 - b0, -1, np.int64)
        mon = self.parts[PRED_ID["monitors"]]
        for br in range(b0, b1):
            s = mon.in_subjects(br)[0]
            if len(s):
                v[br - b0] = s[0]
        self._branch_sensor_vec = v

    def footprint_bytes(self):
        tot = sum(c.nbytes for c in self.chunks) + self.cur_val.nbytes + self.cur_ts.nbytes
        tot += sum(a[:self.n_entities].nbytes for a in self.cols.values())
        for p in self.parts.values():
            for a in (p.src, p.dst, p.vf, p.vt, p.via, p.f_perm, p.f_ptr, p.r_perm, p.r_ptr,
                      p.f_dst, p.f_vf, p.f_vt, p.f_via):
                tot += a.nbytes
        wal = os.path.getsize(self.wal_path)
        return tot, wal


# ---------------------------------------------------------------------------
# rule-based materialisation of derived relations (shared by all systems)
# ---------------------------------------------------------------------------
def materialise(d):
    """Apply ontology.RULES to the explicit facts of a generated dataset.
    Returns pred -> (n,5) int64 array [s, o, vf, vt, via]."""
    blocks = d["meta_blocks"]
    b0, _ = blocks["Branch"]
    ends = {}
    for p in ENDPOINT_PREDS:
        for s, o, _, _ in d[f"fact_{p}"]:
            ends.setdefault(int(s), {})[p] = int(o)
    # in-service intervals from the affects facts
    outs = {}
    for _, br, vf, vt in d["fact_affects"]:
        outs.setdefault(int(br), []).append((int(vf), int(vt)))
    bn = []
    endpoint_of = {}
    for br, e in ends.items():
        x = e.get("fromBus", e.get("hvBus"))
        y = e.get("toBus", e.get("lvBus"))
        endpoint_of[br] = (x, y)
        iv, cur = [], 0
        for a, b in sorted(outs.get(br, [])):
            if a > cur:
                iv.append((cur, a))
            cur = max(cur, b)
        iv.append((cur, INF_T))
        for a, b in iv:
            bn.append((x, y, a, b, br))
            bn.append((y, x, a, b, br))
    # adjacentTo: branches sharing an endpoint bus (structural, always valid)
    at_bus = {}
    for br, (x, y) in endpoint_of.items():
        at_bus.setdefault(x, []).append(br)
        at_bus.setdefault(y, []).append(br)
    adj = set()
    for brs in at_bus.values():
        for i in brs:
            for j in brs:
                if i != j:
                    adj.add((i, j))
    adj = sorted(adj)
    # substationLink
    sub_of = {int(s): int(o) for s, o, _, _ in d["fact_containedIn"]}
    sl = set()
    for br, (x, y) in endpoint_of.items():
        a, b = sub_of[x], sub_of[y]
        if a != b:
            sl.add((a, b)); sl.add((b, a))
    sl = sorted(sl)
    out = {
        "busNeighbor": np.array(bn, np.int64).reshape(-1, 5),
        "adjacentTo": np.array([(i, j, 0, INF_T, -1) for i, j in adj], np.int64).reshape(-1, 5),
        "substationLink": np.array([(i, j, 0, INF_T, -1) for i, j in sl], np.int64).reshape(-1, 5),
    }
    return out


def load_dataset(path):
    import json
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    meta = json.load(open(path.replace(".npz", "_meta.json")))
    d["meta"] = meta
    d["meta_blocks"] = {k: tuple(v) for k, v in meta["blocks"].items()}
    return d
