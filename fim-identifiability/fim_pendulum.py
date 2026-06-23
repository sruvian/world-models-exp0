import numpy as np
from sim_envs import make_env
from collector import collect_trajectories
import matplotlib.pyplot as plt


if __name__ == "__main__":

    g0, l0, dt = 9.8, 10.0, 0.01
    env_params = {"gravity": g0, "mass1": 1.0, "length": l0, "dt": dt, "max_action": 1, "damping": 0, "seed": 35, "mass2": 0}
    pendulum_env = make_env("PendulumSim", **env_params)

    collector_params = {"num_trajectories": 1, "episode_time": 1000, "policy_seed": 30, "save": False,}
    states, actions, _ = collect_trajectories(pendulum_env, **collector_params)
    s = states[0].astype(np.float64)
    theta_dot = s[:-1, 2]
    sin_theta = s[:-1, 1]
    theta_dot_p1 = s[1:, 2]
    
    sigma2 = np.var(theta_dot_p1 - theta_dot)
    def logp(g, l):
        mu_t = theta_dot - (0.01 * (g / l) * sin_theta)
        return (-0.5/ sigma2) * (np.sum(theta_dot_p1 - mu_t))**2
    
    def hessian(g, l, h_g, h_l):
        f = logp
        d_gg = (f(g+h_g, l) - 2*f(g, l) + f(g-h_g, l)) / (h_g**2)
        d_ll = (f(g, l+h_l) - 2*f(g, l) + f(g, l-h_l)) / (h_l**2)
        d_gl = (f(g+h_g, l+h_l) - f(g+h_g, l-h_l)
                - f(g-h_g, l+h_l) + f(g-h_g, l-h_l)) / (4*h_g*h_l )
        return np.array([[d_gg, d_gl], [d_gl, d_ll]])
    lambda_ratio = []
    hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    for h in hs:
        F = -hessian(g0, l0, h*g0, h*l0)        
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
    
