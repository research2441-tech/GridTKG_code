"""
Construction of the experimental power-grid temporal knowledge graphs.

Topologies come from public test systems shipped with pandapower
(IEEE 118-bus, PEGASE 1354-bus, PEGASE 9241-bus).  The generator
(i) expands the bus-branch model into CIM-typed entities and relations,
(ii) synthesises load, renewable and temperature profiles,
(iii) computes branch flows with the DC power-flow PTDF and multi-outage LODF,
(iv) simulates equipment degradation, storms, forced outages, failures and
     time-based maintenance, and
(v) produces noisy SCADA / PMU / smart-meter measurements with missing values,
     stuck-at-zero sensor dropouts and benign load steps.
All randomness is controlled by a single seed.
"""
import json
import os
import time

import networkx as nx
import numpy as np

from ontology import CLASS_ID, PRED_ID, INF_T

CASES = {"KG-S": "case118", "KG-M": "case1354pegase", "KG-L": "case9241pegase"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_case(name):
    import warnings
    warnings.filterwarnings("ignore")
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.pypower.makePTDF import makePTDF
    net = getattr(pn, name)()
    net.line["in_service"] = True
    net.trafo["in_service"] = True
    pp.rundcpp(net)
    ppc = net._ppc
    lk = net._pd2ppc_lookups
    bus_lookup = lk["bus"]
    nb = ppc["bus"].shape[0]
    slack = int(np.where(ppc["bus"][:, 1] == 3)[0][0])
    ptdf64 = makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"], slack, using_sparse_solver=True)
    l0, l1 = lk["branch"]["line"]
    t0, t1 = lk["branch"]["trafo"]
    br = ppc["branch"]
    f_ppc = br[:, 0].real.astype(int)
    t_ppc = br[:, 1].real.astype(int)
    # pandapower bus index -> ppc bus index; ppc bus index -> compact bus id
    pp_bus = net.bus.index.values
    ppc_of_pp = bus_lookup[pp_bus]
    uniq_ppc = np.unique(ppc_of_pp)
    assert len(uniq_ppc) == nb, "fused buses are not supported"
    bus_of_ppc = np.empty(nb, int)
    bus_of_ppc[ppc_of_pp] = np.arange(len(pp_bus))
    kv = net.bus.vn_kv.values.astype(np.float32)
    # ratings (MVA)
    line_rate = (np.sqrt(3) * net.line.max_i_ka.values * kv[np.searchsorted(pp_bus, net.line.from_bus.values)]
                 * net.line.parallel.values)
    trafo_rate = net.trafo.sn_mva.values * net.trafo.parallel.values
    nbr = br.shape[0]
    order = np.r_[np.arange(l0, l1), np.arange(t0, t1)]
    assert len(order) == nbr
    info = dict(
        n_bus=len(pp_bus), kv=kv,
        br_from=bus_of_ppc[f_ppc], br_to=bus_of_ppc[t_ppc],
        is_trafo=np.r_[np.zeros(l1 - l0, bool), np.ones(t1 - t0, bool)],
        rating=np.r_[line_rate, trafo_rate].astype(np.float32),
        ptdf=None,
        load_bus=np.searchsorted(pp_bus, net.load.bus.values),
        load_p=np.maximum(net.load.p_mw.values, 0.5).astype(np.float32),
        gen_bus=np.searchsorted(pp_bus, np.r_[net.gen.bus.values, net.sgen.bus.values, net.ext_grid.bus.values]),
        gen_pmax=np.r_[np.maximum(net.gen.max_p_mw.fillna(net.gen.p_mw * 1.5).values, net.gen.p_mw.values),
                       np.maximum(net.sgen.max_p_mw.fillna(net.sgen.p_mw * 1.5).values if "max_p_mw" in net.sgen else net.sgen.p_mw.values * 1.5, 1.0),
                       np.full(len(net.ext_grid), np.nan)].astype(np.float32),
        n_sgen=len(net.sgen), n_ext=len(net.ext_grid),
    )
    # PTDF with columns in compact bus order, converted row-block-wise to float32
    col = bus_lookup[pp_bus]
    ptdf = np.empty((nbr, len(pp_bus)), np.float32)
    for i in range(0, nbr, 1000):
        ptdf[i:i + 1000] = ptdf64[i:i + 1000][:, col]
    del ptdf64
    info["ptdf"] = ptdf
    # ext grid capacity: cover the residual of total load
    tot = info["load_p"].sum()
    gp = info["gen_pmax"]
    gp[np.isnan(gp)] = max(0.25 * tot, 10.0)
    gp[gp <= 0] = 1.0
    return info


def _union_find(n, pairs):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def _bridges(n_bus, f, t):
    g = nx.Graph()
    g.add_nodes_from(range(n_bus))
    mult = {}
    for k, (a, b) in enumerate(zip(f, t)):
        key = (min(a, b), max(a, b))
        mult[key] = mult.get(key, 0) + 1
        g.add_edge(*key)
    br = set()
    for a, b in nx.bridges(g):
        key = (min(a, b), max(a, b))
        if mult[key] == 1:
            br.add(key)
    return np.array([(min(a, b), max(a, b)) in br for a, b in zip(f, t)])


def _ar1(rng, n, T, phi, sigma):
    x = np.zeros((n, T), np.float32)
    e = rng.normal(0, sigma, (n, T)).astype(np.float32)
    for k in range(1, T):
        x[:, k] = phi * x[:, k - 1] + e[:, k]
    return x


# ---------------------------------------------------------------------------
# main generator
# ---------------------------------------------------------------------------
def generate(tag, days, step_min, seed=7, out_dir="data", n_regions=None,
             branch_sensor_cov=0.70, pmu_cov=0.10, start_doy=0):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    case = CASES[tag]
    info = _load_case(case)
    nb = info["n_bus"]
    f, t = info["br_from"], info["br_to"]
    nbr = len(f)
    is_tr = info["is_trafo"]
    T = int(days * 24 * 60 / step_min)
    spd = int(24 * 60 / step_min)            # steps per day
    hours = (np.arange(T) * step_min / 60.0).astype(np.float32)

    # ---- substations (buses tied by transformers) and regions (Louvain on substation graph)
    sub_of_bus = _union_find(nb, [(a, b) for a, b, tr in zip(f, t, is_tr) if tr])
    n_sub = sub_of_bus.max() + 1
    g = nx.Graph()
    g.add_nodes_from(range(n_sub))
    for a, b in zip(f, t):
        sa, sb = sub_of_bus[a], sub_of_bus[b]
        if sa != sb:
            g.add_edge(sa, sb)
    comms = nx.community.louvain_communities(g, seed=seed, resolution=0.5 if n_regions is None else 1.0)
    reg_of_sub = np.zeros(n_sub, int)
    for r, c in enumerate(sorted(comms, key=lambda c: -len(c))):
        for s in c:
            reg_of_sub[s] = r
    n_reg = reg_of_sub.max() + 1
    reg_of_bus = reg_of_sub[sub_of_bus]
    reg_of_br = reg_of_bus[f]

    # ---- generation mix: 20 % of conventional units become wind / solar
    n_gen = len(info["gen_bus"])
    gen_kind = np.zeros(n_gen, int)  # 0 thermal, 1 wind, 2 solar
    cand = np.arange(n_gen - info["n_ext"])
    ren = rng.choice(cand, size=max(1, int(0.2 * len(cand))), replace=False)
    gen_kind[ren] = rng.integers(1, 3, size=len(ren))

    # ---- profiles
    doy = start_doy + hours / 24.0
    hod = hours % 24
    dow = (hours // 24) % 7
    n_load = len(info["load_bus"])
    peak_h = rng.normal(18.5, 1.0, n_load)[:, None]
    commercial = rng.random(n_load)[:, None] < 0.35
    daily_res = 0.55 + 0.25 * np.exp(-((hod - peak_h) ** 2) / 8) + 0.12 * np.exp(-((hod - 8) ** 2) / 4)
    daily_com = 0.45 + 0.45 * np.exp(-((hod - 13.5) ** 2) / 18)
    daily = np.where(commercial, daily_com, daily_res)
    weekly = np.where(dow[None, :] >= 5, np.where(commercial, 0.75, 1.05), 1.0)
    seasonal = 1 + 0.12 * np.cos(2 * np.pi * (doy - 15) / 365.0)[None, :]
    load = info["load_p"][:, None] * daily * weekly * seasonal * (1 + _ar1(rng, n_load, T, 0.97, 0.012))
    # benign load steps (industrial switching): 30-60 % change for 2-10 h
    n_steps_ev = int(0.6 * days * max(1, n_load / 100))
    step_events = []
    for _ in range(n_steps_ev):
        i = rng.integers(n_load)
        s0 = rng.integers(T)
        d = int(rng.integers(2, 11) * 60 / step_min)
        mag = rng.choice([-1, 1]) * rng.uniform(0.3, 0.6)
        load[i, s0:s0 + d] *= (1 + mag)
        step_events.append((int(i), int(s0), int(min(T, s0 + d)), float(mag)))
    load = np.maximum(load, 0.05).astype(np.float32)

    wind = np.clip(0.35 + _ar1(rng, n_gen, T, 0.995, 0.03), 0, 1)
    sun = np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0, None)[None, :]
    cloud = np.clip(0.8 + _ar1(rng, n_gen, T, 0.98, 0.04), 0.1, 1)
    avail = np.where(gen_kind[:, None] == 1, wind, np.where(gen_kind[:, None] == 2, sun * cloud, 1.0))
    pmax = info["gen_pmax"][:, None]
    ren_out = np.where(gen_kind[:, None] > 0, pmax * avail, 0.0)
    tot_load = load.sum(0)
    residual = np.maximum(tot_load - ren_out.sum(0), 0)
    therm_cap = np.where(gen_kind == 0, info["gen_pmax"], 0.0)
    gen = ren_out + therm_cap[:, None] / therm_cap.sum() * residual[None, :]
    gen = gen.astype(np.float32)
    temp_reg = (15 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.0)[None, :]
                + 5 * np.sin(2 * np.pi * (hod - 9) / 24.0)[None, :]
                + _ar1(rng, n_reg, T, 0.995, 0.25)).astype(np.float32)

    # ---- injections and pre-outage flows (MW)
    inj = np.zeros((nb, T), np.float32)
    np.add.at(inj, info["gen_bus"], gen)
    np.add.at(inj, info["load_bus"], -load)
    ptdf = info["ptdf"]
    F0 = ptdf @ inj
    rating = np.maximum(info["rating"], np.percentile(np.abs(F0), 99, axis=1) / 0.85)
    rating = np.maximum(rating, 5.0).astype(np.float32)

    # ---- degradation, storms, forced outages, failures, maintenance (daily / hourly)
    bridge = _bridges(nb, f, t)
    age = rng.uniform(1, 45, nbr).astype(np.float32)
    health = rng.uniform(0.45, 1.0, nbr)
    next_maint = rng.uniform(0, 400, nbr)
    exposure = np.where(is_tr, 0.4, 1.0) * rng.uniform(0.6, 1.4, nbr)
    n_days = int(np.ceil(days))
    health_daily = np.zeros((nbr, n_days), np.float32)
    cond_daily = np.full((nbr, n_days), np.nan, np.float32)
    has_cond = is_tr | (rng.random(nbr) < 0.6)
    storm_day = rng.random((n_reg, n_days)) < 1 / 45.0
    status = np.ones((nbr, T), bool)
    outages = []      # (branch, start, end, cause) cause: 0 forced, 1 failure
    maints = []       # (branch, step)
    failures = []     # (branch, day)
    loading_daily = np.zeros((nbr, n_days), np.float32)
    lam0, gam = 1.0e-5, 8.0
    for d in range(n_days):
        sl = slice(d * spd, min(T, (d + 1) * spd))
        ld = np.abs(F0[:, sl]) / rating[:, None]
        loading_daily[:, d] = ld.mean(1)
        stress = np.clip(ld.max(1), 0, 1.5) ** 2 * (1 + 0.03 * np.maximum(temp_reg[reg_of_br, sl].max(1) - 25, 0))
        storm = storm_day[reg_of_br, d]
        dmg = np.where(storm, rng.beta(2, 80, nbr) * exposure, 0.0)
        health -= 0.0007 + 0.0012 * stress + dmg + rng.normal(0, 0.0005, nbr)
        # time-based maintenance (historical policy)
        due = next_maint <= d
        if due.any():
            idx = np.where(due)[0]
            health[idx] = np.minimum(1.0, health[idx] + 0.45)
            next_maint[idx] = d + rng.uniform(150, 450, len(idx))
            for b in idx:
                maints.append((int(b), int(d * spd + rng.integers(spd))))
        health = np.clip(health, 0.0, 1.0)
        health_daily[:, d] = health
        obs = has_cond & (rng.random(nbr) < 0.9)
        cond_daily[obs, d] = (1 - health[obs]) + rng.normal(0, 0.07, obs.sum()) + 0.1 * storm[obs]
        # failures
        lam = lam0 * np.exp(gam * (1 - health)) * (1 + 3 * storm)
        fail = (rng.random(nbr) < lam) & ~bridge & status[:, sl.start]
        for b in np.where(fail)[0]:
            s0 = d * spd + int(rng.integers(spd))
            dur = int(rng.uniform(24, 72) * 60 / step_min)
            outages.append((int(b), s0, min(T, s0 + dur), 1))
            failures.append((int(b), d))
            health[b] = 0.95
        # forced short outages, elevated during storms and at high loading
        rate_h = 6.0e-5 * (1 + 6 * storm) * (1 + 3 * np.maximum(ld.mean(1) - 0.8, 0))
        p_day = 1 - (1 - rate_h) ** 24
        forced = (rng.random(nbr) < p_day) & ~bridge
        for b in np.where(forced)[0]:
            s0 = d * spd + int(rng.integers(spd))
            dur = max(1, int(rng.uniform(1, 8) * 60 / step_min))
            outages.append((int(b), s0, min(T, s0 + dur), 0))
    outages.sort(key=lambda o: o[1])
    for b, s0, s1, _ in outages:
        status[b, s0:s1] = False

    # ---- post-outage flows with the multi-outage LODF; drop outages that island the grid
    F = F0.copy()
    out_sets = {}
    for k in range(T):
        o = np.where(~status[:, k])[0]
        if len(o):
            out_sets.setdefault(tuple(o), []).append(k)
    for oset, steps in out_sets.items():
        o = np.array(oset)
        steps = np.array(steps)
        M = ptdf[:, f[o]] - ptdf[:, t[o]]
        A = np.eye(len(o)) - M[o, :]
        if abs(np.linalg.det(A)) < 1e-6:
            # islanding combination: keep only the first branch of the set out
            o = o[:1]
            M = ptdf[:, f[o]] - ptdf[:, t[o]]
            A = np.eye(1) - M[o, :]
            if abs(A[0, 0]) < 1e-6:
                status[oset[0], steps] = True
                continue
            status[np.array(oset[1:]), steps[:, None]] = True
        F[:, steps] = F0[:, steps] + M @ np.linalg.solve(A, F0[o][:, steps])
        F[o[:, None], steps[None, :]] = 0.0

    # ---- sensors and measurements
    br_sensor = np.where(rng.random(nbr) < branch_sensor_cov)[0]
    pmu_bus = rng.choice(nb, size=max(2, int(pmu_cov * nb)), replace=False)
    # PMU angle channel: nodal potential that reproduces the rating-normalised branch
    # flows in the least-squares sense on the energised graph (reference bus 0)
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    Bs = sp.csr_matrix((np.r_[np.ones(nbr), -np.ones(nbr)], (np.r_[np.arange(nbr), np.arange(nbr)], np.r_[f, t])),
                       shape=(nbr, nb), dtype=np.float64)
    lap = (Bs.T @ Bs).tolil()
    lap[0, :] = 0
    lap[0, 0] = 1
    rhs = np.asarray(Bs.T @ (F / rating[:, None]), dtype=np.float64)
    rhs[0, :] = 0
    ang = (spl.splu(lap.tocsc()).solve(rhs)[pmu_bus] * 5.0).astype(np.float32)

    sens_kind, sens_target, sens_true, sens_scale = [], [], [], []
    for b in br_sensor:
        sens_kind.append(0); sens_target.append(("branch", int(b))); sens_true.append(F[b]); sens_scale.append(rating[b])
    for i in range(n_load):
        sens_kind.append(1); sens_target.append(("load", i)); sens_true.append(load[i]); sens_scale.append(info["load_p"][i])
    for i in range(n_gen):
        sens_kind.append(2); sens_target.append(("gen", i)); sens_true.append(gen[i]); sens_scale.append(info["gen_pmax"][i])
    for j, bb in enumerate(pmu_bus):
        sens_kind.append(3); sens_target.append(("bus", int(bb))); sens_true.append(ang[j]); sens_scale.append(10.0)
    V = np.stack(sens_true).astype(np.float32)
    sc = np.array(sens_scale, np.float32)[:, None]
    V = V + rng.normal(0, 0.01, V.shape).astype(np.float32) * sc
    V[rng.random(V.shape) < 0.005] = np.nan
    ns = V.shape[0]
    dropouts = []
    n_drop = int(0.8 * days * max(1, ns / 300))
    for _ in range(n_drop):
        s = rng.integers(ns)
        s0 = rng.integers(T)
        d = int(rng.integers(1, 7) * 60 / step_min)
        V[s, s0:s0 + d] = 0.0
        dropouts.append((int(s), int(s0), int(min(T, s0 + d))))

    # ---- entity table (ids are contiguous blocks)
    blocks = {}
    etype, e_sub, e_reg, e_kv, e_rating, e_age = [], [], [], [], [], []

    def add(cls, n, sub=None, reg=None, kvv=None, rat=None, ag=None):
        start = len(etype)
        blocks[cls] = (start, start + n)
        etype.extend([CLASS_ID[cls] if isinstance(cls, str) else cls] * n)
        e_sub.extend(list(sub) if sub is not None else [-1] * n)
        e_reg.extend(list(reg) if reg is not None else [-1] * n)
        e_kv.extend(list(kvv) if kvv is not None else [0.0] * n)
        e_rating.extend(list(rat) if rat is not None else [0.0] * n)
        e_age.extend(list(ag) if ag is not None else [0.0] * n)
        return start
    R0 = add("Region", n_reg)
    S0 = add("Substation", n_sub, sub=np.arange(n_sub), reg=reg_of_sub)
    B0 = add("Bus", nb, sub=sub_of_bus, reg=reg_of_bus, kvv=info["kv"])
    line_idx = np.where(~is_tr)[0]
    tr_idx = np.where(is_tr)[0]
    BR0 = len(etype)
    # branches keep their compact index order: lines then transformers
    etype.extend([CLASS_ID["Line"] if not x else CLASS_ID["Transformer"] for x in is_tr])
    e_sub.extend(list(sub_of_bus[f])); e_reg.extend(list(reg_of_br)); e_kv.extend(list(info["kv"][f]))
    e_rating.extend(list(rating)); e_age.extend(list(age))
    blocks["Branch"] = (BR0, BR0 + nbr)
    BK0 = add("Breaker", nbr, sub=sub_of_bus[f], reg=reg_of_br)
    gcls = np.array([CLASS_ID["ThermalUnit"], CLASS_ID["WindUnit"], CLASS_ID["SolarUnit"]])[gen_kind]
    G0 = len(etype)
    etype.extend(list(gcls)); e_sub.extend(list(sub_of_bus[info["gen_bus"]])); e_reg.extend(list(reg_of_bus[info["gen_bus"]]))
    e_kv.extend([0.0] * n_gen); e_rating.extend(list(info["gen_pmax"])); e_age.extend([0.0] * n_gen)
    blocks["Generator"] = (G0, G0 + n_gen)
    L0 = add("Load", n_load, sub=sub_of_bus[info["load_bus"]], reg=reg_of_bus[info["load_bus"]], rat=info["load_p"])
    scls = {0: "SCADA", 1: "SmartMeter", 2: "SCADA", 3: "PMU"}
    SN0 = len(etype)
    target_eid = []
    for k, (kind, (tt, i)) in enumerate(zip(sens_kind, sens_target)):
        etype.append(CLASS_ID[scls[kind]])
        eid = {"branch": BR0, "load": L0, "gen": G0, "bus": B0}[tt] + i
        target_eid.append(eid)
        e_sub.append(e_sub[eid]); e_reg.append(e_reg[eid]); e_kv.append(0.0); e_rating.append(float(sens_scale[k])); e_age.append(0.0)
    blocks["Sensor"] = (SN0, SN0 + ns)
    FE0 = len(etype)
    for b, s0, s1, cause in outages:
        etype.append(CLASS_ID["FaultEvent"]); e_sub.append(e_sub[BR0 + b]); e_reg.append(e_reg[BR0 + b])
        e_kv.append(float(cause)); e_rating.append(0.0); e_age.append(0.0)
    blocks["FaultEvent"] = (FE0, len(etype))
    MR0 = len(etype)
    for b, s in maints:
        etype.append(CLASS_ID["MaintenanceRecord"]); e_sub.append(e_sub[BR0 + b]); e_reg.append(e_reg[BR0 + b])
        e_kv.append(0.0); e_rating.append(0.0); e_age.append(0.0)
    blocks["MaintenanceRecord"] = (MR0, len(etype))

    # ---- explicit facts (temporal quadruples); time unit = step index
    facts = {p: [] for p in PRED_ID}
    for s in range(n_sub):
        facts["locatedIn"].append((S0 + s, R0 + reg_of_sub[s], 0, INF_T))
    for b in range(nb):
        facts["containedIn"].append((B0 + b, S0 + sub_of_bus[b], 0, INF_T))
    for k in range(nbr):
        eid = BR0 + k
        if is_tr[k]:
            hv, lv = (f[k], t[k]) if info["kv"][f[k]] >= info["kv"][t[k]] else (t[k], f[k])
            facts["hvBus"].append((eid, B0 + hv, 0, INF_T))
            facts["lvBus"].append((eid, B0 + lv, 0, INF_T))
            facts["containedIn"].append((eid, S0 + sub_of_bus[f[k]], 0, INF_T))
        else:
            facts["fromBus"].append((eid, B0 + f[k], 0, INF_T))
            facts["toBus"].append((eid, B0 + t[k], 0, INF_T))
        facts["protects"].append((BK0 + k, eid, 0, INF_T))
        facts["containedIn"].append((BK0 + k, S0 + sub_of_bus[f[k]], 0, INF_T))
    for i in range(n_gen):
        facts["connectedTo"].append((G0 + i, B0 + info["gen_bus"][i], 0, INF_T))
    for i in range(n_load):
        facts["connectedTo"].append((L0 + i, B0 + info["load_bus"][i], 0, INF_T))
    for k in range(ns):
        facts["monitors"].append((SN0 + k, target_eid[k], 0, INF_T))
    for j, (b, s0, s1, cause) in enumerate(outages):
        facts["affects"].append((FE0 + j, BR0 + b, s0, s1))
    for j, (b, s) in enumerate(maints):
        facts["maintains"].append((MR0 + j, BR0 + b, s, INF_T))

    meta = dict(tag=tag, case=case, days=days, step_min=step_min, T=T, steps_per_day=spd, seed=seed,
                n_bus=nb, n_branch=nbr, n_line=int((~is_tr).sum()), n_trafo=int(is_tr.sum()),
                n_sub=int(n_sub), n_region=int(n_reg), n_gen=n_gen, n_load=n_load, n_sensor=ns,
                n_pmu=int(len(pmu_bus)), n_branch_sensor=int(len(br_sensor)),
                n_outage=len(outages), n_forced=int(sum(o[3] == 0 for o in outages)),
                n_failure=len(failures), n_maint=len(maints), n_dropout=len(dropouts),
                n_load_step=len(step_events), n_storm_days=int(storm_day.sum()),
                blocks={k: [int(a), int(b)] for k, (a, b) in blocks.items()},
                gen_seconds=round(time.time() - t_start, 1))
    os.makedirs(out_dir, exist_ok=True)
    fact_arr = {}
    for p, lst in facts.items():
        a = np.array(lst, dtype=np.int64).reshape(-1, 4)
        fact_arr[f"fact_{p}"] = a
    np.savez_compressed(
        os.path.join(out_dir, f"{tag}.npz"),
        etype=np.array(etype, np.int16), e_sub=np.array(e_sub, np.int32), e_reg=np.array(e_reg, np.int32),
        e_kv=np.array(e_kv, np.float32), e_rating=np.array(e_rating, np.float32), e_age=np.array(e_age, np.float32),
        V=V, sens_kind=np.array(sens_kind, np.int8), sens_target=np.array(target_eid, np.int64),
        F_true=F.astype(np.float32), status=status, load=load, gen=gen, temp_reg=temp_reg,
        rating=rating, is_trafo=is_tr, br_from=f, br_to=t, reg_of_br=reg_of_br, sub_of_bus=sub_of_bus,
        bridge=bridge, age=age, health_daily=health_daily, cond_daily=cond_daily, has_cond=has_cond,
        loading_daily=loading_daily, storm_day=storm_day, br_sensor=br_sensor, pmu_bus=pmu_bus,
        outages=np.array(outages, np.int64).reshape(-1, 4), maints=np.array(maints, np.int64).reshape(-1, 2),
        failures=np.array(failures, np.int64).reshape(-1, 2), dropouts=np.array(dropouts, np.int64).reshape(-1, 3),
        load_steps=np.array(step_events, np.float64).reshape(-1, 4), next_maint=next_maint, health_end=health,
        exposure=exposure, **fact_arr)
    with open(os.path.join(out_dir, f"{tag}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="KG-S")
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    print(json.dumps(generate(a.tag, a.days, a.step, a.seed, a.out), indent=1))
