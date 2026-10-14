"""Statistics, tables and figures for the system benchmark (db_*.json, ablation_*.json)."""
import json
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridtkg_store import load_dataset, materialise

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data"
FIG = sys.argv[3] if len(sys.argv) > 3 else "figures"
os.makedirs(FIG, exist_ok=True)
SYS = ["gridtkg", "postgres", "duckdb", "polyglot"]
LABEL = dict(gridtkg="GridTKG", postgres="PostgreSQL", duckdb="DuckDB", polyglot="Kùzu+DuckDB")
COL = dict(gridtkg="#1F4E79", postgres="#7F7F7F", duckdb="#C9A227", polyglot="#B65C3A")
QS = ["O1", "O2", "O3", "O4", "A1", "A2", "A3", "A4", "A5", "M1", "M2", "M3", "S1", "S2"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, h


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


out = {"db": {}, "datasets": {}}
for tag in ("KG-S", "KG-M", "KG-L"):
    d = json.load(open(os.path.join(RES, f"db_{tag}.json")))
    o = {"queries": {}, "systems": {}}
    for s in SYS:
        r = d[s]
        mix = r["mixed"]
        o["systems"][s] = dict(version=r["version"], load_s=r["load_seconds"], footprint_mb=r["footprint_bytes"] / 1e6,
                               footprint_after_mb=r["footprint_after_bytes"] / 1e6,
                               mix_ops=ci([m["ops_per_s"] for m in mix]), mix_oltp_p95=float(np.mean([m["oltp_p95"] for m in mix])),
                               mix_olap_mean=float(np.mean([m["olap_mean"] for m in mix])))
    for q in QS:
        qo = {}
        runs = {s: np.array([np.mean(x) for x in d[s]["queries"][q]["lat_ms"]]) for s in SYS}
        for s in SYS:
            allc = np.concatenate([np.array(x) for x in d[s]["queries"][q]["lat_ms"]])
            m, h = ci(runs[s])
            qo[s] = dict(mean=m, ci=h, p50=float(np.percentile(allc, 50)), p95=float(np.percentile(allc, 95)),
                         calls=int(len(allc)))
        ps = []
        for s in SYS[1:]:
            t, p = stats.ttest_ind(runs["gridtkg"], runs[s], equal_var=False)
            qo[s]["welch_t"] = float(t)
            ps.append(p)
        adj = holm(np.array(ps))
        for s, p, pa in zip(SYS[1:], ps, adj):
            qo[s]["p"] = float(p)
            qo[s]["p_holm"] = float(pa)
            qo[s]["speedup"] = qo[s]["mean"] / qo["gridtkg"]["mean"]
        best = min(SYS[1:], key=lambda s: qo[s]["mean"])
        qo["best_baseline"] = best
        qo["speedup_vs_best"] = qo[best]["mean"] / qo["gridtkg"]["mean"]
        o["queries"][q] = qo
    o["agreement"] = {k: int(sum(sum(x) for x in v.values())) for k, v in d["_agreement"].items()}
    o["agreement_total"] = int(sum(len(x) for x in d["_agreement"]["gridtkg"].values()))
    # non-trivial validation answers
    val = d["_validation"]["gridtkg"]
    o["validation_nontrivial"] = dict(M2_nonempty=int(sum(len(x[0]) > 0 for x in val["M2"])),
                                      A5_reachable=int(sum(x >= 0 for x in val["A5"])),
                                      O2_rows=int(sum(len(x[0]) for x in val["O2"])))
    out["db"][tag] = o
    # dataset statistics
    ds = load_dataset(os.path.join(DATA, f"{tag}.npz"))
    der = materialise(ds)
    meta = ds["meta"]
    n_ent = len(ds["etype"])
    n_exp = int(sum(len(ds[k]) for k in ds if k.startswith("fact_")))
    n_der = int(sum(len(v) for v in der.values()))
    bn = der["busNeighbor"]
    b0, b1 = meta["blocks"]["Bus"]
    nb = b1 - b0
    pairs = set()
    for x, y, _, _, _ in bn.tolist():
        pairs.add((min(x, y), max(x, y)))
    deg = np.zeros(nb, int)
    for x, y in pairs:
        deg[x - b0] += 1; deg[y - b0] += 1
    import networkx as nx
    g = nx.Graph(); g.add_nodes_from(range(nb)); g.add_edges_from([(x - b0, y - b0) for x, y in pairs])
    rng = np.random.default_rng(0)
    src = rng.choice(nb, size=min(nb, 200), replace=False)
    ecc = max(max(nx.single_source_shortest_path_length(g, int(s)).values()) for s in src)
    adj = der["adjacentTo"]
    bd = np.bincount(adj[:, 0] - meta["blocks"]["Branch"][0], minlength=meta["n_branch"])
    ent_deg = np.zeros(n_ent, int)
    for k in ds:
        if k.startswith("fact_") and len(ds[k]):
            np.add.at(ent_deg, ds[k][:, 0], 1); np.add.at(ent_deg, ds[k][:, 1], 1)
    for v in der.values():
        np.add.at(ent_deg, v[:, 0], 1); np.add.at(ent_deg, v[:, 1], 1)
    V = ds["V"]
    kv = ds["e_kv"]
    fa, fb, trm = kv[b0 + ds["br_from"]], kv[b0 + ds["br_to"]], ds["is_trafo"]
    viol_line = int(((~trm) & (np.abs(fa - fb) > 0.05 * np.maximum(fa, fb))).sum())
    trafo_eq = int((trm & (fa == fb)).sum())
    out["datasets"][tag] = dict(viol_line=viol_line, trafo_eq=trafo_eq,
        case=meta["case"], buses=nb, branches=meta["n_branch"], lines=meta["n_line"], trafos=meta["n_trafo"],
        substations=meta["n_sub"], regions=meta["n_region"], generators=meta["n_gen"], loads=meta["n_load"],
        sensors=meta["n_sensor"], pmu=meta["n_pmu"], branch_sensors=meta["n_branch_sensor"],
        entities=n_ent, explicit_facts=n_exp, derived_facts=n_der, total_facts=n_exp + n_der,
        meas_rows=int(V.size), meas_missing_pct=float(np.isnan(V).mean() * 100), raw_mb=float(V.nbytes / 1e6),
        outages=meta["n_outage"], failures=meta["n_failure"], maint=meta["n_maint"], dropouts=meta["n_dropout"],
        load_steps=meta["n_load_step"], storm_days=meta["n_storm_days"],
        bus_edges=len(pairs), bus_density=2 * len(pairs) / (nb * (nb - 1)), bus_deg_mean=float(deg.mean()),
        bus_deg_max=int(deg.max()), bus_deg_std=float(deg.std()), bus_diam_lb=int(ecc),
        branch_adj_mean=float(bd.mean()), branch_adj_max=int(bd.max()),
        kg_deg_mean=float(ent_deg.mean()), kg_deg_max=int(ent_deg.max()), T=meta["T"], days=meta["days"],
        step_min=meta["step_min"], mat_s=json.load(open(os.path.join(RES, f"db_{tag}.json")))["_meta"]["materialise_seconds"])
    out["datasets"][tag]["deg_hist"] = np.bincount(deg).tolist()

# ablation
ab = json.load(open(os.path.join(RES, "ablation_KG-L.json")))
abl = {}
for kind in ("full", "no-derived", "row-log"):
    for q, runs in ab[kind].items():
        abl.setdefault(q, {})[kind] = ci([np.mean(x) for x in runs])
for q in abl:
    for kind in ("no-derived", "row-log"):
        if kind in abl[q]:
            f = [np.mean(x) for x in ab["full"][q]]
            k = [np.mean(x) for x in ab[kind][q]]
            abl[q][kind + "_ratio"] = float(np.mean(k) / np.mean(f))
            abl[q][kind + "_p"] = float(stats.ttest_ind(f, k, equal_var=False)[1])
out["ablation"] = abl
out["ablation_agreement"] = ab["_agreement"]
json.dump(out, open(os.path.join(RES, "stats_db.json"), "w"), indent=1, default=float)

# ---------------- figures ----------------
# Fig: degree distribution of the bus graphs
fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))
for ax, tag in zip(axs, ("KG-S", "KG-M", "KG-L")):
    h = np.array(out["datasets"][tag]["deg_hist"], float)
    k = np.arange(len(h))
    m = h > 0
    ax.bar(k[m], h[m] / h.sum(), color="#1F4E79", width=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("energised bus degree k")
    ax.set_ylabel("fraction of buses P(k)")
    s = out["datasets"][tag]
    ax.set_title(f"{tag} ({s['case']}): mean {s['bus_deg_mean']:.2f}, max {s['bus_deg_max']}", fontsize=8.5)
    ax.grid(alpha=0.3, which="both", axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig03_degree.png"), dpi=300)
plt.close(fig)

# Fig: KG-L latency per query and system (log scale, 95 % CI)
o = out["db"]["KG-L"]["queries"]
fig, ax = plt.subplots(figsize=(10.5, 3.9))
w = 0.2
x = np.arange(len(QS))
for i, s in enumerate(SYS):
    m = [o[q][s]["mean"] for q in QS]
    e = [o[q][s]["ci"] for q in QS]
    ax.bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=1.5, color=COL[s], label=LABEL[s], error_kw=dict(lw=0.6))
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(QS)
ax.set_ylabel("mean latency per call (ms, log scale)")
ax.set_xlabel("benchmark operation (Table 5); bars: mean of 30 run means, whiskers: 95% CI")
for xx in (3.5, 8.5, 11.5):
    ax.axvline(xx, color="#999999", lw=0.6, ls=":")
for xx, t in ((1.5, "operational"), (6, "analytical"), (10, "maintenance"), (12.5, "storage")):
    ax.text(xx, ax.get_ylim()[1] * 0.6, t, ha="center", fontsize=8, color="#555555")
ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
ax.grid(axis="y", alpha=0.3, which="major")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig04_latency_KG-L.png"), dpi=300)
plt.close(fig)

# Fig: scalability of selected operations with graph size
sel = ["O2", "O3", "A2", "A3", "A5", "M3"]
fig, axs = plt.subplots(2, 3, figsize=(10, 5.2))
sizes = [out["datasets"][t]["entities"] for t in ("KG-S", "KG-M", "KG-L")]
for ax, q in zip(axs.ravel(), sel):
    for s in SYS:
        m = [out["db"][t]["queries"][q][s]["mean"] for t in ("KG-S", "KG-M", "KG-L")]
        e = [out["db"][t]["queries"][q][s]["ci"] for t in ("KG-S", "KG-M", "KG-L")]
        ax.errorbar(sizes, m, yerr=e, marker="o", ms=3.5, lw=1.2, color=COL[s], label=LABEL[s], capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(q, fontsize=9)
    ax.set_xlabel("KG entities (KG-S, KG-M, KG-L)")
    ax.set_ylabel("latency (ms)")
    ax.grid(alpha=0.3, which="both")
axs[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig05_scalability.png"), dpi=300)
plt.close(fig)
print("ok")
