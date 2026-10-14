"""
Analytics layer on top of GridTKG: branch outage detection (FD) and
7-day failure prediction for predictive maintenance (PM).

Features are assembled through the GridTKG read API (measurement history,
monitors / connectedTo / endpoint facts, maintains and affects facts) and the
graph is the materialised adjacentTo relation.  Models: MLP, GRU, GCN, GTN
(gtn_models.py) plus a rule baseline for FD.  Chronological 60/20/20 split.

usage: python learn_tasks.py --data data_learn/KG-M.npz --out results --seeds 3
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from gridtkg_store import GridTKG, load_dataset, materialise
from gtn_models import MODELS, norm_adj, n_params
from ontology import PRED_ID, CLASS_ID

torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# feature assembly through the KG store
# ---------------------------------------------------------------------------
def build_store(d, path):
    derived = materialise(d)
    if os.path.exists(path):
        os.remove(path)
    s = GridTKG(path)
    s.bulk_load(d, derived)
    s.finalize_indexes(d["meta_blocks"]["Branch"])
    return s


def kg_context(s, d):
    b0, b1 = d["meta_blocks"]["Branch"]
    n = b1 - b0
    ends = np.array([s.branch_ends(b) for b in range(b0, b1)])
    adj = s.parts[PRED_ID["adjacentTo"]]
    src = adj.src - b0
    dst = adj.dst - b0
    sens = s._branch_sensor_vec
    rating = s.cols["e_rating"][b0:b1]
    is_tr = (s.cols["etype"][b0:b1] == CLASS_ID["Transformer"]).astype(np.float32)
    age = s.cols["e_age"][b0:b1]
    return dict(b0=b0, n=n, ends=ends, src=src, dst=dst, sens=sens, rating=rating, is_tr=is_tr, age=age)


def fd_features(s, d, ctx):
    """Hourly branch features (T, N, 9)."""
    T = d["meta"]["T"]
    n = ctx["n"]
    rating = ctx["rating"]
    sens = ctx["sens"]
    mon = sens >= 0
    flow = np.full((n, T), np.nan, np.float32)
    flow[mon] = s.history(sens[mon], 0, T)
    fl = flow / rating[:, None]
    valid = ~np.isnan(fl)
    f0 = np.nan_to_num(fl)
    prev = np.c_[np.full((n, 1), np.nan, np.float32), fl[:, :-1]]
    dfl = np.nan_to_num(fl - prev)
    zero = valid & (np.abs(np.nan_to_num(fl)) < 0.01)
    lag24 = np.c_[np.full((n, 24), np.nan, np.float32), fl[:, :-24]]
    d24 = np.nan_to_num(np.abs(fl) - np.abs(lag24))
    # endpoint PMU angle change and endpoint load change via KG facts
    s0 = s.sensor0
    kinds = d["sens_kind"]
    tgt = d["sens_target"]
    pmu_k = np.where(kinds == 3)[0]
    bus_ang = {}
    if len(pmu_k):
        A = s.history(pmu_k + s0, 0, T)
        dA = np.abs(np.nan_to_num(np.diff(A, axis=1, prepend=A[:, :1])))
        for j, k in enumerate(pmu_k):
            bus_ang[int(tgt[k])] = dA[j]
    load_k = np.where(kinds == 1)[0]
    Lh = s.history(load_k + s0, 0, T)
    dL = np.abs(np.nan_to_num(np.diff(Lh, axis=1, prepend=Lh[:, :1])))
    ct = s.parts[PRED_ID["connectedTo"]]
    bus_dl = {}
    for j, k in enumerate(load_k):
        bus = int(ct.out_rows(int(tgt[k]))[0][0])
        bus_dl[bus] = bus_dl.get(bus, 0) + dL[j]
    ang = np.zeros((n, T), np.float32)
    dlo = np.zeros((n, T), np.float32)
    for i, (x, y) in enumerate(ctx["ends"]):
        for b in (x, y):
            if b in bus_ang:
                ang[i] += bus_ang[b]
            if b in bus_dl:
                dlo[i] += bus_dl[b]
    dlo /= rating[:, None]
    X = np.stack([f0, np.repeat(mon[:, None], T, 1).astype(np.float32), valid.astype(np.float32), dfl,
                  zero.astype(np.float32), ang, dlo, np.repeat(ctx["is_tr"][:, None], T, 1), d24], -1)
    X = np.clip(X, -5, 5).astype(np.float32)
    return np.ascontiguousarray(X.transpose(1, 0, 2)), flow  # (T, N, F)


def fd_labels(d):
    st = d["status"]
    prev = np.c_[np.ones((st.shape[0], 1), bool), st[:, :-1]]
    onset = prev & ~st
    keep = st | onset           # drop hours of continuing outages
    return onset.T.astype(np.float32), keep.T


def pm_features(s, d, ctx, flow):
    """Daily branch features (D, N, 11)."""
    T = d["meta"]["T"]
    D = T // 24
    n = ctx["n"]
    ld = np.abs(flow[:, :D * 24]).reshape(n, D, 24) / ctx["rating"][:, None, None]
    with np.errstate(all="ignore"):
        lmean = np.nan_to_num(np.nanmean(ld, 2))
        lmax = np.nan_to_num(np.nanmax(ld, 2))
    mon = (ctx["sens"] >= 0).astype(np.float32)
    cond = d["cond_daily"][:, :D]
    cmask = (~np.isnan(cond)).astype(np.float32)
    reg = d["reg_of_br"]
    temp = d["temp_reg"][:, :D * 24].reshape(-1, D, 24).max(2)
    tfeat = (temp[reg] - 15) / 10.0
    storm = d["storm_day"][reg][:, :D].astype(np.float32)
    # days since maintenance from KG 'maintains' facts; outages in last 30 days from 'affects' facts
    b0 = ctx["b0"]
    last_m = np.full(n, -400.0)
    mt = s.parts[PRED_ID["maintains"]]
    ev_m = sorted(zip((mt.vf // 24).tolist(), (mt.dst - b0).tolist()))
    dsm = np.zeros((n, D), np.float32)
    j = 0
    for day in range(D):
        while j < len(ev_m) and ev_m[j][0] <= day:
            last_m[ev_m[j][1]] = ev_m[j][0]
            j += 1
        dsm[:, day] = np.minimum(day - last_m, 730) / 365.0
    af = s.parts[PRED_ID["affects"]]
    out_cnt = np.zeros((n, D), np.float32)
    for br, vf in zip((af.dst - b0).tolist(), (af.vf // 24).tolist()):
        out_cnt[br, vf:min(D, vf + 30)] += 1
    X = np.stack([lmean, lmax, np.repeat(mon[:, None], D, 1), np.nan_to_num(cond), cmask, tfeat, storm, dsm,
                  np.repeat((ctx["age"] / 50.0)[:, None], D, 1), np.repeat(ctx["is_tr"][:, None], D, 1),
                  out_cnt / 3.0], -1)
    return np.ascontiguousarray(np.clip(X, -5, 5).astype(np.float32).transpose(1, 0, 2))


def pm_labels(d, H=7):
    T = d["meta"]["T"]
    D = T // 24
    n = d["status"].shape[0]
    y = np.zeros((D, n), np.float32)
    for b, day in d["failures"]:
        y[max(0, day - H):day, b] = 1.0
    up = d["status"][:, 23:D * 24:24].T          # in service at the end of each day
    return y, up


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def make_batch(X, times, W):
    return torch.from_numpy(np.stack([X[t - W + 1:t + 1] for t in times]))


def predict(model, X, A, times, W, bs=16):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(times), bs):
            out.append(torch.sigmoid(model(make_batch(X, times[i:i + bs], W), A)).numpy())
    return np.concatenate(out)


def train(name, X, Y, K, A, tr, va, W, seed, epochs, lr=2e-3, bs=8, hidden=48, pos_cap=200.0, balance=True,
          patience=4, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    F = X.shape[-1]
    model = MODELS[name](F, hidden) if name != "MLP" else MODELS[name](F, 64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos = Y[tr][K[tr]].mean()
    pw = torch.tensor(min(pos_cap, (1 - pos) / max(pos, 1e-6)))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    has_pos = tr[(Y[tr] * K[tr]).sum(1) > 0]
    no_pos = tr[(Y[tr] * K[tr]).sum(1) == 0]
    best, best_state, bad = -1, None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        if balance:
            neg = rng.choice(no_pos, size=min(len(no_pos), max(len(has_pos), 64)), replace=False)
            ep_t = rng.permutation(np.r_[has_pos, neg])
        else:
            ep_t = rng.permutation(tr)
        tl = 0.0
        for i in range(0, len(ep_t), bs):
            tt = ep_t[i:i + bs]
            xb = make_batch(X, tt, W)
            yb = torch.from_numpy(Y[tt])
            kb = torch.from_numpy(K[tt].astype(np.float32))
            loss = (lossf(model(xb, A), yb) * kb).sum() / kb.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(tt)
        p = predict(model, X, A, va, W)
        yv, kv = Y[va][K[va]], K[va]
        ap = average_precision_score(yv, p[kv]) if yv.sum() > 0 else 0.0
        hist.append(dict(epoch=ep, train_loss=tl / len(ep_t), val_ap=float(ap)))
        if log:
            print(f"    {name} s{seed} ep{ep:2d} loss {tl / len(ep_t):.4f} valAP {ap:.4f}", flush=True)
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def op_point(yv, pv, yt, pt):
    """Threshold maximising F1 on validation, applied to test."""
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    tau = th[int(np.argmax(f1[:-1]))] if len(th) else 0.5
    return tau, rates(yt, pt >= tau)


def rates(y, a):
    y = y.astype(bool)
    tp = int((a & y).sum()); fp = int((a & ~y).sum()); fn = int((~a & y).sum()); tn = int((~a & ~y).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, FDR=tp / max(tp + fn, 1), FAR=fp / max(fp + tn, 1),
                precision=tp / max(tp + fp, 1), F1=2 * tp / max(2 * tp + fp + fn, 1),
                false_alarm_ratio=fp / max(tp + fp, 1))


def recall_at_budget(Y, P, K, frac):
    """Fraction of positive asset-days ranked inside the daily top-`frac` of risk scores."""
    hit = tot = 0
    for y, p, k in zip(Y, P, K):
        if y[k].sum() == 0:
            continue
        pk, yk = p[k], y[k]
        m = max(1, int(np.ceil(frac * len(pk))))
        top = np.argsort(-pk)[:m]
        hit += yk[top].sum()
        tot += yk.sum()
    return float(hit / max(tot, 1))


def boot_diff(y, k, pa, pb, metric, n=1000, seed=0):
    """Paired bootstrap over time units of metric(pa) - metric(pb)."""
    rng = np.random.default_rng(seed)
    T = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, T, T)
        yy, kk = y[idx][k[idx]], k[idx]
        if yy.sum() == 0:
            continue
        vals.append(metric(yy, pa[idx][kk]) - metric(yy, pb[idx][kk]))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float((vals <= 0).mean())


def run_task(task, X, Y, K, A, W, seeds, epochs, out, extra_eval=None):
    T = len(X)
    t_tr, t_va = int(0.6 * T), int(0.8 * T)
    tr = np.arange(W - 1 + (24 if task == "FD" else 0), t_tr)
    va = np.arange(t_tr, t_va)
    te = np.arange(t_va, T)
    res = dict(split=dict(train=[int(tr[0]), int(tr[-1])], val=[int(va[0]), int(va[-1])], test=[int(te[0]), int(te[-1])]),
               counts={}, models={})
    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        res["counts"][nm] = dict(instances=int(K[idx].sum()), positives=int((Y[idx] * K[idx]).sum()))
    preds = {}
    for name in ("MLP", "GRU", "GCN", "GTN"):
        runs = []
        P_seeds = []
        for seed in range(seeds):
            t0 = time.time()
            model, hist = train(name, X, Y, K, A, tr, va, W, seed, epochs, log=True)
            tt = time.time() - t0
            pv = predict(model, X, A, va, W)
            pt = predict(model, X, A, te, W)
            yv, kv = Y[va][K[va]], K[va]
            yt, kt = Y[te][K[te]], K[te]
            r = dict(seed=seed, train_seconds=tt, epochs=len(hist), params=n_params(model),
                     AUROC=float(roc_auc_score(yt, pt[kt])), AP=float(average_precision_score(yt, pt[kt])))
            t1 = time.perf_counter()
            predict(model, X, A, te[:64], W)
            r["infer_ms_per_step"] = (time.perf_counter() - t1) * 1e3 / 64
            if task == "FD":
                tau, op = op_point(yv, pv[kv], yt, pt[kt])
                r.update(threshold=float(tau), **op)
                tau3 = np.quantile(pv[kv][yv == 0], 1 - 1e-3)
                r["FDR_at_FAR1e-3"] = rates(yt, pt[kt] >= tau3)["FDR"]
            else:
                r["recall_at_5pct"] = recall_at_budget(Y[te], pt, K[te], 0.05)
                r["recall_at_1pct"] = recall_at_budget(Y[te], pt, K[te], 0.01)
            r["history"] = hist
            runs.append(r)
            P_seeds.append(pt)
            if seed == 0:
                torch.save(model.state_dict(), os.path.join(out, f"{task}_{name}_seed0.pt"))
            print(f"  {task} {name} seed {seed}: AUROC {r['AUROC']:.4f} AP {r['AP']:.4f} ({tt:.0f}s)", flush=True)
        res["models"][name] = dict(runs=runs)
        preds[name] = np.mean(P_seeds, 0)
    # paired bootstrap of AP: GTN vs each comparison model (seed-averaged scores)
    res["bootstrap_AP_diff"] = {}
    for name in ("MLP", "GRU", "GCN"):
        res["bootstrap_AP_diff"][f"GTN-{name}"] = boot_diff(Y[te], K[te], preds["GTN"], preds[name], average_precision_score)
    if extra_eval:
        res.update(extra_eval(Y, K, te, va))
    np.savez_compressed(os.path.join(out, f"{task}_test_scores.npz"), te=te, Y=Y[te], K=K[te], **preds)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_learn/KG-M.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs_fd", type=int, default=12)
    ap.add_argument("--epochs_pm", type=int, default=40)
    ap.add_argument("--tasks", default="FD,PM")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load_dataset(a.data)
    t0 = time.time()
    s = build_store(d, os.path.join(a.out, "learn_wal.log"))
    ctx = kg_context(s, d)
    A = norm_adj(ctx["src"], ctx["dst"], ctx["n"])
    Xfd, flow = fd_features(s, d, ctx)
    feat_s = time.time() - t0
    out = dict(meta=dict(d["meta"], feature_seconds=feat_s, n_nodes=int(ctx["n"]), n_edges=int(len(ctx["src"]))))
    path = os.path.join(a.out, "learn_results.json")
    if os.path.exists(path):
        out.update({k: v for k, v in json.load(open(path)).items() if k in ("FD", "PM")})
    if "FD" in a.tasks:
        Y, K = fd_labels(d)
        # rule baseline: zero reading on a previously loaded monitored branch
        def rule_eval(Y, K, te, va):
            x = Xfd
            alarm = (x[:, :, 4] > 0) & (np.abs(x[:, :, 0] - x[:, :, 3]) > 0.05)
            yt, kt = Y[te][K[te]], K[te]
            return dict(rule=rates(yt, alarm[te][kt]))
        out["FD"] = run_task("FD", Xfd, Y, K, A, 4, a.seeds, a.epochs_fd, a.out, rule_eval)
        json.dump(out, open(path, "w"), indent=1)
    if "PM" in a.tasks:
        Xpm = pm_features(s, d, ctx, flow)
        Y, K = pm_labels(d)
        np.savez_compressed(os.path.join(a.out, "pm_features.npz"), X=Xpm, Y=Y, K=K)
        out["PM"] = run_task("PM", Xpm, Y, K, A, 14, a.seeds, a.epochs_pm, a.out)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
