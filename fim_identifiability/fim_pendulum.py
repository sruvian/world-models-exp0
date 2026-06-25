import numpy as np
from sim_envs import make_env
from collector import collect_trajectories
import matplotlib.pyplot as plt
from hessian import hessian_func

def logp(g, l, mass1, theta_dot, dt, sin_theta, actions, sigma2, theta_dot_p1):
    ml2= 1/(mass1 * l**2)
    mu_t = theta_dot - (dt * (g / l) * sin_theta) + (dt * actions/ml2)
    return (-0.5/ sigma2) * (np.sum(theta_dot_p1 - mu_t))**2

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
    
    sigma2 = 1

    func_args = {"mass1": mass1, "theta_dot": theta_dot, "dt": dt, "sin_theta": sin_theta, "actions": actions,
                 "sigma2": sigma2, "theta_dot_p1":theta_dot_p1}
    
    lambda_ratio = []
    hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    for h in hs:
        F = -hessian_func(logp, g0, l0, h*g0, h*l0, **func_args)        
        eigvals, eigvecs = np.linalg.eigh(F)    
        lambda_ratio.append(np.log(np.abs(eigvals[1]/ (eigvals[0]))))
        v_null = eigvecs[:, 0]                  
        target = np.array([g0/l0, 1.0]); target /= np.linalg.norm(target)
        align = abs(np.dot(v_null, target))     
        print(f"h={h:.0e}  eigvals={eigvals}  null·target={align:.4f}")

    plt.figure()
    plt.plot(hs, lambda_ratio, marker='o')
    plt.xscale('log')
    plt.xlabel('h')
    plt.ylabel(r'$\log(\lambda_1/\lambda_2)$')
    plt.show()