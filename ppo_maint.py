"""
Maintenance-crew allocation with PPO on top of the GridTKG analytics layer.

Each day the operator can send at most C crews; a crew restores the health of
one in-service branch.  Asset degradation, storms, failures and condition
readings follow the same stochastic model as grid_generator.py; exogenous
loading and temperature are replayed from the KG-M learning dataset.

Policies compared (30 common-random-number evaluation episodes each):
  none, time-based, condition-greedy, GCN-greedy, GTN-greedy (expected-cost),
  PPO-raw (no GTN risk), GTN+PPO.

usage: python ppo_maint.py --data data_learn/KG-M.npz --res results --seeds 3
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from gtn_models import GTN, GCN, norm_adj
from gridtkg_store import load_dataset
from learn_tasks import build_store, kg_context

torch.set_num_threads(2)

H_EP = 60          # days per episode
C_CREW = 5         # crews per day
C_MAINT = 0.5      # cost of one maintenance action
C_FAIL0, C_FAIL1 = 10.0, 40.0   # failure cost = C_FAIL0 + C_FAIL1 * criticality
W_PM = 14


class MaintEnv:
    def __init__(self, d, ctx, Xpm, start_days):
        self.n = ctx["n"]
        self.D = Xpm.shape[0]
        self.Xpm = Xpm
        self.reg = d["reg_of_br"]
        self.n_reg = int(self.reg.max()) + 1
        self.has_cond = d["has_cond"]
        self.exposure = d["exposure"]
        F = d["F_true"]
        T = F.shape[1]
        Dn = T // 24
        r = d["rating"][:, None, None]
        ldm = (np.abs(F[:, :Dn * 24]).reshape(self.n, Dn, 24) / r)
        self.lmax_true = ldm.max(2)
        self.temp_max = d["temp_reg"][:, :Dn * 24].reshape(self.n_reg, Dn, 24).max(2)
        fm = np.abs(F).mean(1)
        crit = fm / fm.max()
        crit[d["bridge"]] = 1.0
        self.crit = crit.astype(np.float32)
        self.fail_cost = C_FAIL0 + C_FAIL1 * self.crit
        self.health_daily = d["health_daily"]
        self.start_days = start_days
        self.age = ctx["age"]
        self.is_tr = ctx["is_tr"]

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.d0 = int(self.rng.choice(self.start_days))
        self.t = 0
        self.h = self.health_daily[:, self.d0].astype(np.float64).copy()
        # window of past PM features (taken from the dataset up to d0)
        self.win = self.Xpm[self.d0 - W_PM + 1:self.d0 + 1].copy()
        self.dsm = self.win[-1, :, 7] * 365.0
        self.out = np.zeros(self.n, int)
        self.recent_fail = np.zeros((30, self.n), np.float32)
        self.cond_hist = self.win[:, :, 3].copy()
        self.cmask_hist = self.win[:, :, 4].copy()
        return self.obs_window()

    def obs_window(self):
        return self.win

    def step(self, maint_idx):
        rng = self.rng
        day = (self.d0 + 1 + self.t) % self.D
        cost = 0.0
        maint_idx = [i for i in maint_idx if self.out[i] == 0]
        if maint_idx:
            self.h[maint_idx] = np.minimum(1.0, self.h[maint_idx] + 0.45)
            self.dsm[maint_idx] = 0.0
            cost += C_MAINT * len(maint_idx)
        storm = rng.random(self.n_reg) < 1 / 45.0
        st_b = storm[self.reg]
        stress = np.clip(self.lmax_true[:, day], 0, 1.5) ** 2 * (1 + 0.03 * np.maximum(self.temp_max[self.reg, day] - 25, 0))
        dmg = np.where(st_b, rng.beta(2, 80, self.n) * self.exposure, 0.0)
        self.h = np.clip(self.h - (0.0007 + 0.0012 * stress + dmg + rng.normal(0, 0.0005, self.n)), 0, 1)
        lam = 1.0e-5 * np.exp(8.0 * (1 - self.h)) * (1 + 3 * st_b)
        fail = (rng.random(self.n) < lam) & (self.out == 0)
        nf = int(fail.sum())
        cost += float(self.fail_cost[fail].sum())
        self.h[fail] = 0.95
        self.out = np.maximum(self.out - 1, 0)
        self.out[fail] = 2
        self.dsm += 1
        self.recent_fail = np.roll(self.recent_fail, 1, 0)
        self.recent_fail[0] = fail
        obs_c = self.has_cond & (rng.random(self.n) < 0.9)
        cond = np.where(obs_c, (1 - self.h) + rng.normal(0, 0.07, self.n) + 0.1 * st_b, 0.0)
        x = self.Xpm[day].copy()                       # replayed loading / temperature columns
        x[:, 3] = cond
        x[:, 4] = obs_c
        x[:, 5] = (self.temp_max[self.reg, day] - 15) / 10.0
        x[:, 6] = st_b
        x[:, 7] = np.minimum(self.dsm, 730) / 365.0
        x[:, 10] = self.recent_fail.sum(0) / 3.0
        self.win = np.concatenate([self.win[1:], np.clip(x, -5, 5)[None]], 0).astype(np.float32)
        self.t += 1
        done = self.t >= H_EP
        return self.win, -cost, done, dict(failures=nf, maint=len(maint_idx), cost=cost,
                                         fail_cost=float(self.fail_cost[fail].sum()),
                                         crit_failed=float(self.crit[fail].sum()))


# ---------------------------------------------------------------------------
class RiskModel:
    """Frozen PM model (GTN or GCN) + Platt calibration on the validation split."""

    def __init__(self, kind, path, A, Xpm, Y, K, va):
        F = Xpm.shape[-1]
        self.m = (GTN if kind == "GTN" else GCN)(F, 48)
        self.m.load_state_dict(torch.load(path))
        self.m.eval()
        self.A = A
        logits = []
        with torch.no_grad():
            for i in range(0, len(va), 16):
                tt = va[i:i + 16]
                xb = torch.from_numpy(np.stack([Xpm[t - W_PM + 1:t + 1] for t in tt]))
                logits.append(self.m(xb, A).numpy())
        z = np.concatenate(logits)[K[va]]
        y = Y[va][K[va]]
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e3).fit(z[:, None], y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def __call__(self, win):
        with torch.no_grad():
            z = self.m(torch.from_numpy(win[None]), self.A).numpy()[0]
        return z, 1 / (1 + np.exp(-(self.a * z + self.b)))


def asset_obs(env, win, risk=None):
    cond = win[:, :, 3]
    cm = win[:, :, 4]
    last = cond[-1]
    c7 = (cond[-7:] * cm[-7:]).sum(0) / np.maximum(cm[-7:].sum(0), 1)
    cols = [last, cm[-1], c7, win[-1, :, 7], win[-7:, :, 0].mean(0), env.crit, (env.out > 0).astype(np.float32),
            env.is_tr, env.age / 50.0]
    if risk is not None:
        z, p = risk
        cols += [np.clip(z / 5.0, -3, 3), np.clip(p * 10, 0, 5), np.clip(p * env.fail_cost / C_MAINT / 10.0, 0, 10)]
    return np.stack(cols, -1).astype(np.float32)


class Policy(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(f, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.skip = nn.Sequential(nn.Linear(2 * f + 1, h), nn.Tanh(), nn.Linear(h, 1))
        self.value = nn.Sequential(nn.Linear(2 * f + 1, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def logits(self, o, g, avail):
        s = self.score(o).squeeze(-1)
        s = torch.where(avail, s, torch.full_like(s, -1e9))
        sk = self.skip(g).expand(*s.shape[:-1], C_CREW)
        return torch.cat([s, sk], -1)

    def g(self, o, tfrac):
        return torch.cat([o.mean(-2), o.max(-2).values, tfrac.unsqueeze(-1)], -1)


def pl_logprob(logits, acts):
    """Plackett-Luce log-probability of ordered selections acts (B, C)."""
    lp = 0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    ent = 0
    for k in range(acts.shape[-1]):
        l = logits.masked_fill(mask, -1e9)
        logp = torch.log_softmax(l, -1)
        a = acts[:, k]
        lp = lp + logp.gather(-1, a[:, None]).squeeze(-1)
        if k == 0:
            pr = logp.exp()
            ent = -(pr * logp).sum(-1)
        mask = mask.clone()
        mask[torch.arange(len(a)), a] = True
    return lp, ent


def run_episode(env, seed, chooser, risk_model=None):
    win = env.reset(seed)
    tot = dict(cost=0.0, failures=0, maint=0, fail_cost=0.0, crit_failed=0.0)
    traj = []
    done = False
    while not done:
        risk = risk_model(win) if risk_model is not None else None
        idx, extra = chooser(env, win, risk)
        win2, r, done, info = env.step(idx)
        traj.append((extra, r, done))
        for k in tot:
            tot[k] += info[k]
        win = win2
    return tot, traj


# ---- heuristic policies ----------------------------------------------------
def ch_none(env, win, risk):
    return [], None


def ch_time(env, win, risk):
    cand = np.where(env.out == 0)[0]
    return cand[np.argsort(-env.dsm[cand])[:C_CREW]].tolist(), None


def ch_cond(env, win, risk):
    o = asset_obs(env, win)
    sc = np.where(env.out == 0, o[:, 2] * env.fail_cost, -np.inf)
    return np.argsort(-sc)[:C_CREW].tolist(), None


def ch_expected(env, win, risk):
    """Expected-cost greedy: maintain if calibrated 7-day failure probability x failure cost exceeds the action cost."""
    _, p = risk
    gain = p * env.fail_cost
    gain[env.out > 0] = -np.inf
    top = np.argsort(-gain)[:C_CREW]
    return [int(i) for i in top if gain[i] > C_MAINT], None


# ---- PPO ---------------------------------------------------------------------
def collect_teacher(env, teacher, risk_model, use_risk, rng, episodes):
    """Roll out a greedy teacher and record (obs, avail, tfrac, ordered action padded with skip tokens)."""
    data = []
    for _ in range(episodes):
        win = env.reset(int(rng.integers(1 << 30)))
        done = False
        while not done:
            risk = risk_model(win) if risk_model is not None else None
            o = torch.from_numpy(asset_obs(env, win, risk if use_risk else None))
            avail = torch.from_numpy(env.out == 0)
            tf = torch.tensor(env.t / H_EP, dtype=torch.float32)
            idx, _ = teacher(env, win, risk)
            act = list(idx)[:C_CREW] + [env.n + k for k in range(C_CREW - len(idx))]
            data.append((o, avail, tf, torch.tensor(act)))
            win, _, done, _ = env.step(idx)
    return data


def behaviour_clone(pol, data, steps=300, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    O = torch.stack([d[0] for d in data]); AV = torch.stack([d[1] for d in data])
    TF = torch.stack([d[2] for d in data]); ACT = torch.stack([d[3] for d in data])
    for _ in range(steps):
        mb = torch.randint(0, len(data), (32,), generator=g)
        lg = pol.logits(O[mb], pol.g(O[mb], TF[mb]), AV[mb])
        lp, _ = pl_logprob(lg, ACT[mb])
        loss = -lp.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


def ppo_train(env, risk_model, use_risk, seed, updates=40, eps_per_update=4, log=None, teacher=None,
              bc_episodes=4, lr=1e-4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = 12 if use_risk else 9
    pol = Policy(f)
    if teacher is not None:
        data = collect_teacher(env, teacher, risk_model, use_risk, rng, bc_episodes)
        bc_loss = behaviour_clone(pol, data, seed=seed)
        if log:
            print(f"    BC warm start: {len(data)} teacher steps, final NLL {bc_loss:.2f}", flush=True)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gamma, lam, clip = 0.99, 0.95, 0.2
    curve = []
    for u in range(updates):
        buf = []
        ep_ret = []
        for e in range(eps_per_update):
            win = env.reset(int(rng.integers(1 << 30)))
            done = False
            R = 0.0
            while not done:
                risk = risk_model(win) if use_risk else None
                o = torch.from_numpy(asset_obs(env, win, risk))
                avail = torch.from_numpy(env.out == 0)
                tf = torch.tensor(env.t / H_EP, dtype=torch.float32)
                with torch.no_grad():
                    g = pol.g(o, tf)
                    lg = pol.logits(o, g, avail)
                    gum = -torch.log(-torch.log(torch.rand_like(lg).clamp_min(1e-12)))
                    act = torch.topk(lg + gum, C_CREW).indices
                    lp, _ = pl_logprob(lg[None], act[None])
                    v = pol.value(g)
                idx = [int(a) for a in act if a < env.n]
                _, r, done, _ = env.step(idx)
                buf.append(dict(o=o, avail=avail, tf=tf, act=act, lp=lp[0], v=float(v), r=r / 50.0, done=done))
                R += r
            ep_ret.append(R)
        # GAE
        adv = np.zeros(len(buf))
        last = 0.0
        for i in reversed(range(len(buf))):
            nv = 0.0 if buf[i]["done"] else buf[i + 1]["v"]
            delta = buf[i]["r"] + gamma * nv - buf[i]["v"]
            last = delta + gamma * lam * (0.0 if buf[i]["done"] else last)
            adv[i] = last
        ret = adv + np.array([b["v"] for b in buf])
        adv_t = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
        ret_t = torch.tensor(ret, dtype=torch.float32)
        O = torch.stack([b["o"] for b in buf]); AV = torch.stack([b["avail"] for b in buf])
        TF = torch.stack([b["tf"] for b in buf]); ACT = torch.stack([b["act"] for b in buf])
        LP0 = torch.stack([b["lp"] for b in buf])
        for _ in range(4):
            perm = torch.randperm(len(buf))
            for i in range(0, len(buf), 60):
                mb = perm[i:i + 60]
                g = pol.g(O[mb], TF[mb])
                lg = pol.logits(O[mb], g, AV[mb])
                lp, ent = pl_logprob(lg, ACT[mb])
                ratio = torch.exp(lp - LP0[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
                vloss = ((pol.value(g).squeeze(-1) - ret_t[mb]) ** 2).mean()
                loss = -torch.min(s1, s2).mean() + 0.5 * vloss - 0.01 * ent.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
        curve.append(float(np.mean(ep_ret)))
        if log:
            print(f"    PPO{'+GTN' if use_risk else '-raw'} s{seed} upd {u:2d} mean return {np.mean(ep_ret):8.1f}", flush=True)
    return pol, curve


def ppo_chooser(pol, use_risk):
    def ch(env, win, risk):
        o = torch.from_numpy(asset_obs(env, win, risk if use_risk else None))
        avail = torch.from_numpy(env.out == 0)
        tf = torch.tensor(env.t / H_EP, dtype=torch.float32)
        with torch.no_grad():
            g = pol.g(o, tf)
            lg = pol.logits(o, g, avail)
            # greedy decoding of the Plackett-Luce policy at evaluation time
            act = torch.topk(lg, C_CREW).indices
        return [int(a) for a in act if a < env.n], None
    return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_learn/KG-M.npz")
    ap.add_argument("--res", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--updates", type=int, default=15)
    ap.add_argument("--skip_heuristics", action="store_true")
    ap.add_argument("--eval_eps", type=int, default=30)
    a = ap.parse_args()
    d = load_dataset(a.data)
    s = build_store(d, os.path.join(a.res, "ppo_wal.log"))
    ctx = kg_context(s, d)
    A = norm_adj(ctx["src"], ctx["dst"], ctx["n"])
    z = np.load(os.path.join(a.res, "pm_features.npz"))
    Xpm, Y, K = z["X"], z["Y"], z["K"]
    D = len(Xpm)
    va = np.arange(int(0.6 * D), int(0.8 * D))
    test_days = np.arange(int(0.8 * D), D - 1)
    train_days = np.arange(W_PM, int(0.6 * D))
    gtn = RiskModel("GTN", os.path.join(a.res, "PM_GTN_seed0.pt"), A, Xpm, Y, K, va)
    gcn = RiskModel("GCN", os.path.join(a.res, "PM_GCN_seed0.pt"), A, Xpm, Y, K, va)
    env_tr = MaintEnv(d, ctx, Xpm, train_days)
    env_te = MaintEnv(d, ctx, Xpm, test_days)
    out = dict(config=dict(H=H_EP, crews=C_CREW, c_maint=C_MAINT, c_fail=[C_FAIL0, C_FAIL1], updates=a.updates,
                           eps_per_update=4, gamma=0.99, gae_lambda=0.95, clip=0.2, lr=1e-4, seeds=a.seeds,
                           bc_teacher=dict(PPO_raw="Condition-greedy", GTN_PPO="GTN-greedy"), bc_episodes=4,
                           bc_steps=300, bc_lr=1e-3,
                           eval_episodes=a.eval_eps, platt_gtn=[gtn.a, gtn.b], platt_gcn=[gcn.a, gcn.b]),
               policies={}, curves={})
    eval_seeds = [900_000 + i for i in range(a.eval_eps)]

    def evaluate(name, chooser, risk_model):
        rows = []
        t0 = time.time()
        for sd in eval_seeds:
            tot, _ = run_episode(env_te, sd, chooser, risk_model)
            rows.append(tot)
        out["policies"][name] = dict(rows=rows, seconds=time.time() - t0)
        c = np.array([r["cost"] for r in rows])
        print(f"  {name:18s} cost {c.mean():8.1f} ± {c.std():6.1f}  failures {np.mean([r['failures'] for r in rows]):5.2f} "
              f"maint {np.mean([r['maint'] for r in rows]):6.1f}", flush=True)

    prev = os.path.join(a.res, "ppo_results.json")
    if a.skip_heuristics and os.path.exists(prev):
        old = json.load(open(prev))
        for k in ("No maintenance", "Time-based", "Condition-greedy", "GCN-greedy", "GTN-greedy"):
            out["policies"][k] = old["policies"][k]
    else:
        evaluate("No maintenance", ch_none, None)
        evaluate("Time-based", ch_time, None)
        evaluate("Condition-greedy", ch_cond, None)
        evaluate("GCN-greedy", ch_expected, gcn)
        evaluate("GTN-greedy", ch_expected, gtn)
        json.dump(out, open(prev, "w"), indent=1)
    for use_risk, nm, teacher in ((False, "PPO-raw", ch_cond), (True, "GTN+PPO", ch_expected)):
        curves = []
        for sd in range(a.seeds):
            t0 = time.time()
            pol, curve = ppo_train(env_tr, gtn, use_risk, sd, a.updates, log=True, teacher=teacher)
            curves.append(curve)
            evaluate(f"{nm} (seed {sd})", ppo_chooser(pol, use_risk), gtn if use_risk else None)
            out["policies"][f"{nm} (seed {sd})"]["train_seconds"] = time.time() - t0
            torch.save(pol.state_dict(), os.path.join(a.res, f"ppo_{nm.replace('+', '_')}_seed{sd}.pt"))
            json.dump(out, open(os.path.join(a.res, "ppo_results.json"), "w"), indent=1)
        out["curves"][nm] = curves
    json.dump(out, open(os.path.join(a.res, "ppo_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
