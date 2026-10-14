# executed inside make_numbers.py (shares S, R, json, os, np, stats)
_PP = os.path.join(R, "ppo_results.json")
if os.path.exists(_PP):
    _P = json.load(open(_PP))
    _names = ["No maintenance", "Time-based", "Condition-greedy", "GCN-greedy", "GTN-greedy", "PPO-raw", "GTN+PPO"]
    _key = {"No maintenance": "none", "Time-based": "time", "Condition-greedy": "cond", "GCN-greedy": "gcng",
            "GTN-greedy": "gtng", "PPO-raw": "pporaw", "GTN+PPO": "gtnppo"}
    _per = {}
    for n in _names:
        pol = _P["policies"]
        if n in pol:
            rows = [pol[n]["rows"]]
        else:
            rows = [pol[k]["rows"] for k in pol if k.startswith(n + " (seed")]
        if not rows:
            continue
        cost = np.array([[r["cost"] for r in rr] for rr in rows]).mean(0)
        fails = np.array([[r["failures"] for r in rr] for rr in rows]).mean(0)
        maint = np.array([[r["maint"] for r in rr] for rr in rows]).mean(0)
        _per[n] = cost
        k = _key[n]
        S[f"ppo_{k}_cost"] = f"{cost.mean():,.0f}"
        S[f"ppo_{k}_ci"] = f"{stats.t.ppf(0.975, len(cost) - 1) * cost.std(ddof=1) / np.sqrt(len(cost)):,.0f}"
        S[f"ppo_{k}_fail"] = f"{fails.mean():.1f}"
        S[f"ppo_{k}_maint"] = f"{maint.mean():.0f}"
        if len(rows) > 1:
            S[f"ppo_{k}_seedsd"] = f"{np.array([[r['cost'] for r in rr] for rr in rows]).mean(1).std():.1f}"
    if "GTN+PPO" in _per:
        ref = _per["GTN+PPO"]
        for n, c in _per.items():
            k = _key[n]
            if n != "GTN+PPO":
                S[f"ppo_{k}_p"] = sci(stats.wilcoxon(c, ref).pvalue, 1)
                S[f"ppo_red_vs_{k}"] = f"{(1 - ref.mean() / c.mean()) * 100:.1f}"
    if "GTN-greedy" in _per:
        for n in ("Time-based", "Condition-greedy", "GCN-greedy", "No maintenance"):
            S[f"ppo_gtng_red_vs_{_key[n]}"] = f"{(1 - _per['GTN-greedy'].mean() / _per[n].mean()) * 100:.1f}"
            S[f"ppo_gtng_p_vs_{_key[n]}"] = sci(stats.wilcoxon(_per["GTN-greedy"], _per[n]).pvalue, 1)
    S["ppo_curves_updates"] = str(_P["config"]["updates"])
if "fd_GTN_AUROC_m" in S and "ppo_gtng_red_vs_time" in S:
    S["abs_ml"] = (f"In the application layer, a graph temporal network detected branch outages with an AUROC of "
                   f"{S['fd_GTN_AUROC_m']} and an average precision of {S['fd_GTN_AP_m']}, above graph-only and "
                   f"time-only models, and GTN-based risk ranking reduced maintenance cost by "
                   f"{S['ppo_gtng_red_vs_time']}% relative to time-based maintenance; PPO fine-tuning did not "
                   f"improve on this greedy policy.")
    S["concl_ppo"] = (f"Ranking maintenance work by GTN failure risk lowered the episode cost by "
                      f"{S['ppo_gtng_red_vs_time']}% against time-based maintenance with fewer than half of the crew dispatches, whereas "
                      f"PPO fine-tuning did not improve on that greedy rule within the training budget, so the "
                      f"learning-based allocation remains an open question rather than a result.")
