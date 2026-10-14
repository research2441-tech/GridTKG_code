# -*- coding: utf-8 -*-
"""Collect every number quoted in the manuscript text into numbers.json (single source of truth)."""
import json
import os

import numpy as np
from scipy import stats

R = os.environ.get("RES", ".")
st = json.load(open(os.path.join(R, "stats_db.json")))
S = {}


def sci(x, nd=2):
    m, e = f"{x:.{nd}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def f(x, nd=2):
    return f"{x:,.{nd}f}"


def ms(x):
    """latency formatting with sensible precision"""
    if x < 0.1:
        return f"{x:.3f}"
    if x < 10:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return f"{x:,.0f}"


def sp(x):
    return f"{x:,.1f}" if x < 100 else f"{x:,.0f}"


ds = st["datasets"]
for t, k in (("KG-S", "S"), ("KG-M", "M"), ("KG-L", "L")):
    d = ds[t]
    S[f"ds_density_{k}"] = sci(d['bus_density'])
    S[f"ds_maxdeg_{k}"] = str(d["bus_deg_max"])
    S[f"ds_deg_{k}"] = f"{d['bus_deg_mean']:.2f}"
    S[f"ds_diam_{k}"] = str(d["bus_diam_lb"])
    S[f"ds_kgmax_{k}"] = f"{d['kg_deg_max']:,}"
    S[f"ds_entities_{k}"] = f"{d['entities']:,}"
    S[f"ds_facts_{k}"] = f"{d['total_facts']:,}"
    S[f"ds_rows_{k}"] = f"{d['meas_rows']:,}"
    S[f"ds_mb_{k}"] = f"{d['raw_mb']:.1f}"
degs = [ds[t]["bus_deg_mean"] for t in ds]
S["ds_rows_ratio"] = f"{ds['KG-L']['meas_rows'] / ds['KG-S']['meas_rows']:.0f}"
S["ds_trafo_eq_L"] = str(ds["KG-L"]["trafo_eq"])
S["x_loc"] = str(sum(1 for fn in ("ontology.py", "gridtkg_store.py") for ln in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)) if ln.strip()))
S["x_zenodo"] = "10.5281/zenodo.XXXXXXX"
S["ds_degmin"] = f"{min(degs):.2f}"
S["ds_degmax"] = f"{max(degs):.2f}"
L = ds["KG-L"]
for key in ("n_region", "n_sub", "n_bus", "n_line", "n_trafo", "n_branch", "n_gen", "n_load", "n_sensor", "n_outage", "n_maint"):
    src = dict(n_region="regions", n_sub="substations", n_bus="buses", n_line="lines", n_trafo="trafos",
               n_branch="branches", n_gen="generators", n_load="loads", n_sensor="sensors", n_outage="outages",
               n_maint="maint")[key]
    S[key] = f"{L[src]:,}"

# versions
dbL = st["db"]["KG-L"]
S["pg_version"] = dbL["systems"]["postgres"]["version"].split(" ")[0]
import duckdb, kuzu, torch, pandapower
S["duck_version"] = duckdb.__version__
S["kuzu_version"] = kuzu.__version__
S["torch_version"] = torch.__version__.split("+")[0]
S["pp_version"] = pandapower.__version__

# DB results
for t, k in (("KG-S", "S"), ("KG-M", "M"), ("KG-L", "L")):
    qd = st["db"][t]["queries"]
    for q, v in qd.items():
        for s in ("gridtkg", "postgres", "duckdb", "polyglot"):
            S[f"db_{k}_{q}_{s}"] = ms(v[s]["mean"])
            if s != "gridtkg":
                S[f"db_{k}_{q}_{s}_x"] = sp(v[s]["speedup"])
        S[f"db_{k}_{q}_best"] = dict(postgres="PostgreSQL", duckdb="DuckDB", polyglot="Kùzu+DuckDB")[v["best_baseline"]]
        S[f"db_{k}_{q}_bestx"] = sp(v["speedup_vs_best"])
    sy = st["db"][t]["systems"]
    for s in sy:
        S[f"db_{k}_mix_{s}"] = f"{sy[s]['mix_ops'][0]:,.0f}"
        S[f"db_{k}_mixci_{s}"] = f"{sy[s]['mix_ops'][1]:,.0f}"
        S[f"db_{k}_p95_{s}"] = ms(sy[s]["mix_oltp_p95"])
        S[f"db_{k}_foot_{s}"] = f"{sy[s]['footprint_mb']:,.0f}" if sy[s]['footprint_mb'] >= 10 else f"{sy[s]['footprint_mb']:.1f}"
        S[f"db_{k}_load_{s}"] = f"{sy[s]['load_s']:.1f}"
    S[f"db_{k}_mixx_pg"] = f"{sy['gridtkg']['mix_ops'][0] / sy['postgres']['mix_ops'][0]:.1f}"
pmax = max(v[s]["p_holm"] for v in dbL["queries"].values() for s in ("postgres", "duckdb", "polyglot"))
S["db_pmax"] = sci(pmax, 1)
xs = [v[s]["speedup"] for v in dbL["queries"].values() for s in ("postgres", "duckdb", "polyglot")]
S["db_xmin"] = f"{min(xs):.1f}"
S["db_xmax"] = f"{max(xs):,.0f}"
oper = [dbL["queries"][q]["gridtkg"]["mean"] for q in ("O1", "O2", "O3", "O4")]
S["db_L_oper_max_grid"] = ms(max(oper))
S["db_L_oper_max_pg"] = ms(max(dbL["queries"][q]["postgres"]["mean"] for q in ("O1", "O2", "O3", "O4")))
S["db_L_oper_max_duck"] = ms(max(dbL["queries"][q]["duckdb"]["mean"] for q in ("O1", "O2", "O3", "O4")))
S["db_L_oper_max_poly"] = ms(max(dbL["queries"][q]["polyglot"]["mean"] for q in ("O1", "O2", "O3", "O4")))
S["db_L_S1_rate"] = f"{18514 / dbL['queries']['S1']['gridtkg']['mean'] / 1e3:.1f}"   # million values per second
S["db_L_S1_rate_pg"] = f"{18514 / dbL['queries']['S1']['postgres']['mean']:.0f}"     # thousand values per second
S["db_L_footx_pg"] = f"{dbL['systems']['postgres']['footprint_mb'] / dbL['systems']['gridtkg']['footprint_mb']:.1f}"
S["db_L_meas_mb"] = f"{L['raw_mb']:.0f}"
S["db_agree_total"] = str(sum(st["db"][t]["agreement"]["gridtkg"] for t in st["db"]))
S["db_agree_each"] = str(st["db"]["KG-L"]["agreement_total"])

ab = st["ablation"]
for q, v in ab.items():
    S[f"abl_{q}_full"] = ms(v["full"][0])
    for kind, kk in (("no-derived", "nd"), ("row-log", "rl")):
        if kind in v:
            S[f"abl_{q}_{kk}"] = ms(v[kind][0])
            S[f"abl_{q}_{kk}_x"] = f"{v[kind + '_ratio']:.1f}"


# ---------------------------------------------------------------- learning tasks
LP = os.path.join(R, "learn_results.json")
if os.path.exists(LP):
    Lr = json.load(open(LP))
    for task in ("FD", "PM"):
        if task not in Lr:
            continue
        T = Lr[task]
        tk = task.lower()
        for nm, c in T["counts"].items():
            S[f"{tk}_{nm}_inst"] = f"{c['instances']:,}"
            S[f"{tk}_{nm}_pos"] = f"{c['positives']:,}"
        for m, v in T["models"].items():
            runs = v["runs"]
            keys = ["AUROC", "AP", "train_seconds", "epochs", "params", "infer_ms_per_step"]
            keys += ["FDR", "FAR", "precision", "F1", "false_alarm_ratio", "FDR_at_FAR1e-3", "TP", "FP", "FN"] if task == "FD" else ["recall_at_5pct", "recall_at_1pct"]
            for k in keys:
                a = np.array([r[k] for r in runs], float)
                kk = k.replace("-", "").replace("_at_", "at")
                S[f"{tk}_{m}_{kk}_mean"] = a.mean()
                S[f"{tk}_{m}_{kk}_sd"] = a.std()
        for k, (mean, lo, hi, p) in T["bootstrap_AP_diff"].items():
            kk = k.replace("-", "_")
            S[f"{tk}_boot_{kk}"] = f"{mean:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}"
            S[f"{tk}_boot_{kk}_p"] = p
        if task == "FD":
            for k, v in T["rule"].items():
                S[f"fd_rule_{k}"] = v
    if "FD" in Lr:
        for m in ("MLP", "GRU", "GCN", "GTN"):
            far = np.mean([r["FAR"] for r in Lr["FD"]["models"][m]["runs"]])
            S[f"fd_{m}_fa_day"] = f"{far * 1991 * 24:.1f}"
            S[f"fd_{m}_FARpm"] = f"{far * 1e3:.3f}"
        S["fd_rule_fa_day"] = f"{Lr['FD']['rule']['FAR'] * 1991 * 24:.1f}"
        S["fd_rule_FARpm"] = f"{Lr['FD']['rule']['FAR'] * 1e3:.3f}"
        S["fd_test_hours"] = f"{Lr['FD']['split']['test'][1] - Lr['FD']['split']['test'][0] + 1:,}"
    # pretty strings
    def pm(x, sd, nd=3):
        return f"{x:.{nd}f} ± {sd:.{nd}f}"
    for k in list(S):
        if k.endswith("_mean") and (k.startswith("fd_") or k.startswith("pm_")):
            base = k[:-5]
            S[base] = pm(S[k], S[base + "_sd"], 3)
            S[base + "_m"] = f"{S[k]:.3f}"
    for k in [k for k in S if isinstance(S[k], float)]:
        S[k] = f"{S[k]:.4g}"

# ---------------------------------------------------------------- PPO
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_numbers_ppo.py")).read()) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_numbers_ppo.py")) else None

json.dump(S, open(os.environ.get("NUMBERS", "numbers.json"), "w"), indent=1, ensure_ascii=False)
print(len(S), "numbers")
