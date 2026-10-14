"""Figures for the analytics layer: outage detection, failure prediction and PPO maintenance."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = sys.argv[2] if len(sys.argv) > 2 else "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
COL = dict(MLP="#9E9E9E", GRU="#C9A227", GCN="#B65C3A", GTN="#1F4E79")


def pr_panel(ax, task, title):
    z = np.load(os.path.join(RES, f"{task}_test_scores.npz"))
    Y, K = z["Y"], z["K"].astype(bool)
    y = Y[K]
    for m in ("MLP", "GRU", "GCN", "GTN"):
        p = z[m][K]
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=COL[m], lw=1.6 if m == "GTN" else 1.1, label=m)
    ax.axhline(y.mean(), color="#666666", ls="--", lw=0.8, label="prevalence")
    ax.set_xlabel("recall (fraction of positives detected)")
    ax.set_ylabel("precision (fraction of alarms that are true)")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)


L = json.load(open(os.path.join(RES, "learn_results.json")))
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
pr_panel(axs[0], "FD", "(a) Outage detection, test split")
fd = L["FD"]["models"]
ms = ["MLP", "GRU", "GCN", "GTN"]
fdr = [np.mean([r["FDR"] for r in fd[m]["runs"]]) for m in ms]
far = [np.mean([r["FAR"] for r in fd[m]["runs"]]) for m in ms]
fdr_sd = [np.std([r["FDR"] for r in fd[m]["runs"]]) for m in ms]
far_sd = [np.std([r["FAR"] for r in fd[m]["runs"]]) for m in ms]
rule = L["FD"]["rule"]
x = np.arange(len(ms) + 1)
axs[1].bar(x, [rule["FDR"]] + fdr, yerr=[0] + fdr_sd, color=["#DDDDDD"] + [COL[m] for m in ms], capsize=2)
axs[1].set_xticks(x); axs[1].set_xticklabels(["Rule"] + ms)
axs[1].set_ylabel("FDR at validation-F1 threshold")
axs[1].set_ylim(0, 1); axs[1].grid(axis="y", alpha=0.3)
axs[1].set_title("(b) Fault detection rate (mean ± s.d.)", fontsize=9)
axs[2].bar(x, np.array([rule["FAR"]] + far) * 1e3, yerr=np.array([0] + far_sd) * 1e3,
           color=["#DDDDDD"] + [COL[m] for m in ms], capsize=2)
axs[2].set_xticks(x); axs[2].set_xticklabels(["Rule"] + ms)
axs[2].set_ylabel("FAR (‰ of normal branch-hours)")
axs[2].grid(axis="y", alpha=0.3)
axs[2].set_title("(c) False alarm rate (mean ± s.d.)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig07_fd.png"), dpi=300)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
pr_panel(axs[0], "PM", "(a) 7-day failure prediction, test split")
pm = L["PM"]["models"]
r5 = [np.mean([r["recall_at_5pct"] for r in pm[m]["runs"]]) for m in ms]
r1 = [np.mean([r["recall_at_1pct"] for r in pm[m]["runs"]]) for m in ms]
xx = np.arange(len(ms))
axs[1].bar(xx - 0.18, r1, 0.36, color=[COL[m] for m in ms], alpha=0.55, label="top 1% per day")
axs[1].bar(xx + 0.18, r5, 0.36, color=[COL[m] for m in ms], label="top 5% per day")
axs[1].set_xticks(xx); axs[1].set_xticklabels(ms)
axs[1].set_ylabel("recall of failing branch-days")
axs[1].set_ylim(0, 1); axs[1].grid(axis="y", alpha=0.3)
axs[1].set_title("(b) Recall at a daily inspection budget", fontsize=9)
axs[1].legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig08_pm.png"), dpi=300)
plt.close(fig)

P = json.load(open(os.path.join(RES, "ppo_results.json")))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw=dict(width_ratios=[1, 1.4]))
for nm, c in (("PPO-raw", "#B65C3A"), ("GTN+PPO", "#1F4E79")):
    cv = np.array(P["curves"][nm])
    u = np.arange(1, cv.shape[1] + 1)
    axs[0].plot(u, cv.mean(0), color=c, label=nm)
    axs[0].fill_between(u, cv.min(0), cv.max(0), color=c, alpha=0.2)
axs[0].set_xlabel("PPO update (4 training episodes each)")
axs[0].set_ylabel("mean episode return (negative cost)")
axs[0].set_title("(a) Training curves (line: mean, band: range of 3 seeds)", fontsize=9)
axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
names = ["No maintenance", "Time-based", "Condition-greedy", "GCN-greedy", "GTN-greedy", "PPO-raw", "GTN+PPO"]
data = []
for n in names:
    if n in P["policies"]:
        data.append([r["cost"] for r in P["policies"][n]["rows"]])
    else:
        rows = [P["policies"][k]["rows"] for k in P["policies"] if k.startswith(n + " (seed")]
        data.append(np.mean([[r["cost"] for r in rr] for rr in rows], 0))
bp = axs[1].boxplot(data, vert=True, patch_artist=True, widths=0.6, showfliers=False, medianprops=dict(color="black"))
for patch, n in zip(bp["boxes"], names):
    patch.set_facecolor("#1F4E79" if n == "GTN+PPO" else "#C9D6E8")
axs[1].set_xticks(range(1, len(names) + 1))
axs[1].set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
axs[1].set_ylabel("total cost per 60-day episode")
axs[1].set_title("(b) Evaluation on 30 common-random-number episodes", fontsize=9)
axs[1].grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig09_ppo.png"), dpi=300)
plt.close(fig)
print("ok")
