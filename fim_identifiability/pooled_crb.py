import glob
import os
import numpy as np
from sim_envs import make_env
from hessian import hessian_func


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

    d = np.load(path)
    all_states  = d["states"]
    all_actions = d["actions"]
    g  = float(d["gravity"]); l = float(d["length"]); dt = float(d["dt"])
    is_cart = "CartPoleSim" in os.path.basename(path)
    if is_cart:
        m1 = float(d["mass1"]); m2 = float(d["mass2"])

    N = all_states.shape[0]
    F_total = np.zeros((2, 2))
    for n in range(N):
        if is_cart:
            logp = build_logp_cartpole(all_states[n], all_actions[n], dt, m1, m2)
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
            condstr = f"{cond:.2e}" if ok else "  n/a"
            print(f"  g={g:6.3f} l={l:6.3f}  N={N}  cond={condstr}{tag}")

    cg_pool, cl_pool, cond_pool, ok_pool = _safe_crb_cond(F_pool)
    pf = np.array(per_file_crb) if per_file_crb else np.array([[np.nan, np.nan]])
    cg_pt, cl_pt = np.nanmean(pf[:, 0]), np.nanmean(pf[:, 1])

    print(f"\n[{len(files)} files, {n_broken} broken]  pattern: {os.path.basename(pattern)}")
    print(f"  POOLED        CRB_g={cg_pool:.3e}  CRB_l={cl_pool:.3e}  cond={cond_pool:.2e}")
    print(f"  PER-TRAJ mean CRB_g={cg_pt:.3e}  CRB_l={cl_pt:.3e}")
    return {"pooled": (cg_pool, cl_pool, cond_pool),
            "per_traj": (cg_pt, cl_pt), "n": len(files), "n_broken": n_broken}


if __name__ == "__main__":
    DATA = os.path.join(os.path.dirname(__file__), "..", "datasets")
    impulse = os.path.join(DATA, "impulse_policy")

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
        pooled_crb(os.path.join(d, f"*{env}*length10.0*"))
        print("\n--- vary-l, hold-g (g=9.8) ---")
        pooled_crb(os.path.join(d, f"*{env}*grav*9.80*"))