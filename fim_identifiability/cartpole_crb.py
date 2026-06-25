import numpy as np
from sim_envs import make_env
from collector import collect_trajectories
import matplotlib.pyplot as plt
from hessian import hessian_func


def one_step_accels(g, l_full, thdot, xdot, th, x, a, dt, max_action):
    env_params = {"gravity": g, "mass1": 0.1, "length": l_full, "dt": dt,
                  "max_action": max_action, "damping": 0, "seed": 35, "mass2": 1.0}
    env = make_env("CartPoleSim", **env_params)
    mu_thdot = np.empty_like(thdot); mu_xdot = np.empty_like(xdot)
    for t in range(len(th)):
        env.theta, env.theta_dot = th[t], thdot[t]
        env.x, env.x_dot = x[t], xdot[t]
        env.env_init = True
        s_next = env.step(a[t])
        mu_thdot[t] = s_next[2]; mu_xdot[t] = s_next[4]
    return mu_thdot, mu_xdot


def logp(g, l_full, thdot, xdot, th, x, thdot_tp1, xdot_tp1, sigma2, a, dt, max_action):
    mu_th, mu_x = one_step_accels(g, l_full, thdot, xdot, th, x, a, dt, max_action)
    return -0.5/sigma2 * (np.sum((thdot_tp1 - mu_th)**2) + np.sum((xdot_tp1 - mu_x)**2))


if __name__ == "__main__":
    g0, l0, dt = 9.8, 10.0, 0.01
    h = 1e-2
    amplitudes = [0, 1, 5, 10, 50, 100]
    base = {"gravity": g0, "mass1": 0.1, "length": l0, "dt": dt,
            "damping": 0, "seed": 35, "mass2": 1.0}

    CRB_g, CRB_l = [], []
    for A in amplitudes:
        env = make_env("CartPoleSim", **{**base, "max_action": A})
        states, actions, _ = collect_trajectories(
            env, num_trajectories=1, episode_time=1000, policy_seed=30,
            save=False, impulse_policy=True)
        s = states[0].astype(np.float64)
        a = actions[0].astype(np.float64)

        th    = np.arctan2(s[:-1, 1], s[:-1, 0])
        thdot = s[:-1, 2]; x = s[:-1, 3]; xdot = s[:-1, 4]
        thdot_tp1 = s[1:, 2]; xdot_tp1 = s[1:, 4]

        func_args = {"thdot": thdot, "xdot": xdot, "th": th, "x": x,
                     "thdot_tp1": thdot_tp1, "xdot_tp1": xdot_tp1,
                     "sigma2": 1.0, "a": a, "dt": dt, "max_action": A}

        F = -hessian_func(logp, g0, l0, h*g0, h*l0, **func_args)
        F_inv = np.linalg.inv(F)
        CRB_g.append(F_inv[0, 0]); CRB_l.append(F_inv[1, 1])
        print(f"A={A:4d}  CRB_g={F_inv[0,0]:.3e}  CRB_l={F_inv[1,1]:.3e}")

    plt.figure()
    plt.plot(amplitudes, CRB_l, marker='o', label='CRB(l)')
    plt.plot(amplitudes, CRB_g, marker='x', label='CRB(g)')
    plt.yscale('log'); plt.xlabel('max_action (forcing amplitude)')
    plt.ylabel('Cramér-Rao bound'); plt.title('CartPole')
    plt.legend(); plt.show()