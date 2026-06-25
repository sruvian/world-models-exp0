import numpy as np
from sim_envs import make_env
from collector import collect_trajectories
import matplotlib.pyplot as plt
from hessian import hessian_func

def logp(g, l, sigma2, tdot_t, dt, tau_t, tdot_tp1):
    mu = tdot_t - dt*(g/l)*sin_t + dt*tau_t/(1.0*l**2)
    return -0.5/sigma2 * np.sum((tdot_tp1 - mu)**2)

if __name__ == "__main__":

    g0, l0, dt, mass1 = 9.8, 10.0, 0.01, 1.0
    env_params = {"gravity": g0, "mass1": mass1, "length": l0, "dt": dt, "max_action": 1, "damping": 0, "seed": 35, "mass2": 0}
    pendulum_env = make_env("PendulumSim", **env_params)

    collector_params = {"num_trajectories": 1, "episode_time": 1000, "policy_seed": 30, "save": False,}
    states, actions, _ = collect_trajectories(pendulum_env, **collector_params)
    s = states[0].astype(np.float64)
    theta_dot = s[:-1, 2]
    sin_theta = s[:-1, 1]
    theta_dot_p1 = s[1:, 2]

    h = 1e-2
    amplitudes = [0, 1, 5, 10, 50, 100]
    CRB_l, CRB_g = [], []

    for A in amplitudes:

        env_params_A = {**env_params, "max_action": A}
        env_A = make_env("PendulumSim", **env_params_A)
        states, actions, _ = collect_trajectories(
            env_A, num_trajectories=1, episode_time=1000, policy_seed=30, save=False, impulse_policy=True)

        s = states[0].astype(np.float64)
        sin_t   = s[:-1, 1]
        tdot_t  = s[:-1, 2]
        tdot_tp1 = s[1:, 2]
        tau_t   = actions[0].astype(np.float64)
        sigma2  = 1.0
        func_args = {"sigma2": sigma2, "tdot_t": tdot_t, "dt": dt, "tau_t": tau_t, "tdot_tp1": tdot_tp1}


        F = -hessian_func(logp, g0, l0, h*g0, h*l0, **func_args)
        F_inv = np.linalg.inv(F)
        CRB_g.append(F_inv[0, 0])    
        CRB_l.append(F_inv[1, 1])    
        print(f"A={A:4d}  CRB_g={F_inv[0,0]:.3e}  CRB_l={F_inv[1,1]:.3e}")

    plt.figure()
    plt.plot(amplitudes, CRB_l, marker='o', label='CRB(l)')
    plt.plot(amplitudes, CRB_g, marker='x', label='CRB(g)')
    plt.yscale('log'); plt.xlabel('max_action (forcing amplitude)')
    plt.ylabel('Cramér-Rao bound'); plt.legend(); plt.show()