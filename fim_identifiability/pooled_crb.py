import glob
import os
import numpy as np
from sim_envs import make_env
from hessian import hessian_func
import argparse
from collections import defaultdict

def build_logp_pendulum(states, actions, dt, sigma2=1.0):
    s = states.astype(np.float64)
    sin_t    = s[:-1, 1]
    tdot_t   = s[:-1, 2]
    tdot_tp1 = s[1:, 2]
    tau_t    = actions.astype(np.float64)

    def logp(g, l):
        mu = tdot_t - dt*(g/l)*sin_t + dt*tau_t/(1.0*l**2)   # mass1 = 1
        return -0.5/sigma2 * np.sum((tdot_tp1 - mu)**2)
    return logp

# def build_logp_driven(states, actions, g, l, dt, drive_omega, drive_amp, damping):
#     s = states.astype(np.float64)
#     sin_t    = s[:-1, 1]
#     tdot_t   = s[:-1, 2]
#     tdot_tp1 = s[1:, 2]
#     tau_t    = actions.astype(np.float64)
#     T = s.shape[0]
#     t_arr = np.arange(T - 1) * dt

#     def logp(g, l, sigma2=1.0):
#         mu = (tdot_t
#               - dt * (g / l) * sin_t
#               - dt * damping * tdot_t
#               + dt * drive_amp * np.cos(drive_omega * t_arr)
#               + dt * tau_t / (l ** 2))
#         return -0.5 / sigma2 * np.sum((tdot_tp1 - mu) ** 2)
#     return logp

def build_logp_driven_ratio(states, actions, dt, drive_omega, drive_amp, damping):
    s = states.astype(np.float64)
    sin_t, tdot_t, tdot_tp1 = s[:-1,1], s[:-1,2], s[1:,2]
    t_arr = np.arange(s.shape[0]-1) * dt
    def logp(r, sigma2=1.0):                       # r = g/l, single param
        mu = (tdot_t - dt*r*sin_t - dt*damping*tdot_t
              + dt*drive_amp*np.cos(drive_omega*t_arr))
        return -0.5/sigma2 * np.sum((tdot_tp1 - mu)**2)
    return logp

def ratio_fisher(F):
    ev = np.linalg.eigvalsh(F)
    lam_max = ev[-1]
    crb_ratio = 1.0 / lam_max if lam_max > 1e-12 else np.inf
    return lam_max, crb_ratio

def build_logp_cartpole(states, actions, dt, mass1, mass2, sigma2=1.0):
    s = states.astype(np.float64)
    th        = np.arctan2(s[:-1, 1], s[:-1, 0])
    thdot     = s[:-1, 2]
    x         = s[:-1, 3]
    xdot      = s[:-1, 4]
    thdot_tp1 = s[1:, 2]
    xdot_tp1  = s[1:, 4]
    a = actions.astype(np.float64)
    max_action = float(np.abs(a).max()) + 1.0

    def one_step(g, l_full):
        env = make_env("CartPoleSim", gravity=g, mass1=mass1, length=l_full,
                       dt=dt, max_action=max_action, damping=0, seed=35, mass2=mass2)
        mu_th = np.empty_like(thdot); mu_x = np.empty_like(xdot)
        for t in range(len(th)):
            env.theta, env.theta_dot = th[t], thdot[t]
            env.x, env.x_dot = x[t], xdot[t]
            env.env_init = True
            sn = env.step(a[t])
            mu_th[t] = sn[2]; mu_x[t] = sn[4]
        return mu_th, mu_x

    def logp(g, l):
        mu_th, mu_x = one_step(g, l)
        return -0.5/sigma2 * (np.sum((thdot_tp1 - mu_th)**2)
                            + np.sum((xdot_tp1  - mu_x)**2))
    return logp


def fim_for_file(path, h=1e-2):

    d = np.load(path,)
    all_states  = d["states"]
    all_actions = d["actions"]
    g  = float(d["gravity"]); l = float(d["length"]); dt = float(d["dt"])
    is_cart = "CartPoleSim" in os.path.basename(path)      
    is_driv = 'DrivenPendulumSim' in os.path.basename(path)
    N = all_states.shape[0]
    F_total = np.zeros((2, 2))
    for n in range(N):
        if is_cart:
            m1 = float(d["mass1"]); m2 = float(d["mass2"])
            logp = build_logp_cartpole(all_states[n], all_actions[n], dt, m1, m2)
        elif is_driv:
            dr_omega = float(d["drive_omega"]); dr_amp = float(d["drive_amp"]); damp = float(d['damping'])
            logp = build_logp_driven_ratio(all_states[n], all_actions[n], dt, dr_omega, dr_amp, damp)
        else:
            logp = build_logp_pendulum(all_states[n], all_actions[n], dt)
        F_total += -hessian_func(logp, g, l, h*g, h*l)
    return F_total, (g, l), N


def _safe_crb_cond(F):
    ev = np.linalg.eigvalsh(F)
    lam_min, lam_max = ev[0], ev[1]
    ok = lam_min > 0 
    if not ok:
        return np.inf, np.inf, np.nan, False
    crb = np.linalg.inv(F)
    return crb[0, 0], crb[1, 1], lam_max / lam_min, True


def pooled_crb(pattern, h=1e-2, verbose=True):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match: {pattern}")

    F_pool = np.zeros((2, 2))
    per_file_crb = []
    n_broken = 0
    for f in files:
        F, (g, l), N = fim_for_file(f, h=h)
        cg, cl, cond, ok = _safe_crb_cond(F)
        if ok:
            F_pool += F
            per_file_crb.append((cg, cl))
        else:
            n_broken += 1
        if verbose:
            tag = "" if ok else " BROKEN FIM (non-PD), skipped"
            condstr = f"{cond:.2f}" if ok else "  n/a"
            print(f"  g={g:6.3f} l={l:6.3f}  N={N}  cond={condstr}{tag}")

    cg_pool, cl_pool, cond_pool, ok_pool = _safe_crb_cond(F_pool)
    pf = np.array(per_file_crb) if per_file_crb else np.array([[np.nan, np.nan]])
    cg_pt, cl_pt = np.nanmean(pf[:, 0]), np.nanmean(pf[:, 1])

    print(f"\n[{len(files)} files, {n_broken} broken]  pattern: {os.path.basename(pattern)}")
    print(f"  POOLED        CRB_g={cg_pool:.3f}  CRB_l={cl_pool:.3f}  cond={cond_pool:.2f}")
    print(f"  PER-TRAJ mean CRB_g={cg_pt:.3f}  CRB_l={cl_pt:.3f}")
    return {"pooled": (cg_pool, cl_pool, cond_pool),
            "per_traj": (cg_pt, cl_pt), "n": len(files), "n_broken": n_broken}


def parse_config(path):
    d = np.load(path)
    return float(d["gravity"]), float(d["length"]), float(d["drive_omega"])

def driven_crb_analysis(pattern, h=1e-2):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)

    by_config = defaultdict(list)
    for f in files:
        g, l, w = parse_config(f)
        by_config[(g, l)].append((w, f))
    config_results = []
    for (g, l), items in sorted(by_config.items()):
        items.sort()
        omega_0 = (g / l) ** 0.5
        
        print(f"\n=== config g={g} l={l}  (omega_0={omega_0:.3f}) ===")
        F_config = np.zeros((2, 2))
        per_omega = []
        for w, f in items:
            F, _, N = fim_for_file(f, h=h)
            F_config += F
            lam_max, crb_r = ratio_fisher(F)
            per_omega.append((w, lam_max, crb_r))
            print(f"  omega_d={w:.3f}  Fisher_ratio={lam_max:.4f}  CRB_ratio={crb_r:.4f}")
        # if (g, l) == (9.8, 2.0):
        #     print(f"\n--- (9.8, 2.0) full Fisher curve, omega_0={(g/l)**0.5:.3f} ---")
        #     for w, lam, crb in sorted(per_omega):
        #         bar = "#" * int(lam / max(x[1] for x in per_omega) * 40)   # crude text bar
        #         print(f"  omega_d={w:.3f}  Fisher={lam:8.2f}  {bar}")
        lam_max_p, crb_r_p = ratio_fisher(F_config)
        print(f"  POOLED over omega_d: Fisher_ratio={lam_max_p:.4f}  CRB_ratio={crb_r_p:.4f}")

        omega_0 = (g / l) ** 0.5
        w_star = max(per_omega, key=lambda x: x[1])[0]
        print(f"  --> Fisher_ratio peaks at omega_d={w_star:.3f}  (omega_0={omega_0:.3f})")
        r_true = g / l
        config_results.append((g, l, r_true, crb_r_p))
    print("\n=== cross-config ratio separability ===")
    for i in range(len(config_results)):
        for j in range(i+1, len(config_results)):
            gi,li,ri,crbi = config_results[i]
            gj,lj,rj,crbj = config_results[j]
            sep = abs(ri - rj) / np.sqrt(crbi + crbj)     # separation in sigma units
            print(f"  ({gi},{li}) r={ri:.2f} vs ({gj},{lj}) r={rj:.2f}: {sep:.1f} sigma apart")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default = 'datasets')
    ap.add_argument("--holdg", default = 9.8, type = float)
    ap.add_argument("--holdl", default = 10, type = float)
    ap.add_argument("--driven", action = "store_true")
    args = ap.parse_args()
    DATA = os.path.join(os.path.dirname(__file__), "..", args.dir)
    impulse = os.path.join(DATA, "impulse_policy")
    if args.driven:
        driv = os.path.join(DATA, "driven")
        print("------combined------")
        print(os.path.join(driv, f"*"))
        driven_crb_analysis(os.path.join(driv, f"*"))

    else:
        blocks = [
            ("WHITE  Pendulum", DATA, "PendulumSim"),
            ("WHITE  CartPole", DATA, "CartPoleSim"),
            ("SPARSE Pendulum", impulse, "PendulumSim"),
            ("SPARSE CartPole", impulse, "CartPoleSim"),
        ]
        for label, d, env in blocks:
            print("\n" + "=" * 60)
            print(label)
            print("=" * 60)
            print("--- combined (all configs) ---")
            pooled_crb(os.path.join(d, f"*{env}*"))
            print("\n--- vary-g, hold-l (l=10) ---")
            pooled_crb(os.path.join(d, f"*{env}*length{args.holdl}*"))
            print("\n--- vary-l, hold-g (g=9.8) ---")
            pooled_crb(os.path.join(d, f"*{env}*grav*{args.holdg}*"))