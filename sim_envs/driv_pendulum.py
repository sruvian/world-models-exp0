import numpy as np
from .base_environment import Environment


class DrivenPendulumSim(Environment):

    def __init__(self, gravity:float, mass1:float, length:float, max_action:float, damping:float,
                 dt:float, seed:int, drive_omega: float, drive_amp: float,**kwargs):
        self.gravity = gravity #m/s
        self.mass1 = mass1 # Kg
        self.length = length #metres
        self.dt = dt #seconds
        self.max_action = max_action #Newton metres
        self.damping = damping
        self.name = "driv_pendulum"
        self.drive_omega = drive_omega
        self.drive_amp = drive_amp

        self.gl = self.gravity/ self.length
        self.ml2 = self.mass1 * (self.length**2)

        self.theta = 0.0
        self.theta_dot = 0.0
        self.t = 0.0  
        self.env_init = False
        self.env_seed = seed
        self.rng = np.random.default_rng(seed)
    
    def step(self, action: float):
        if not self.env_init:
            raise ValueError("Call reset() before calling step")
        action = float(np.clip(action, -self.max_action, self.max_action))

        drive = self.drive_amp * np.cos(self.drive_omega * self.t)
        theta_ddot = (- self.gl * np.sin(self.theta)
                    - self.damping * self.theta_dot
                    + drive
                    + action / self.ml2)
        self.theta_dot += theta_ddot * self.dt
        self.theta     += self.theta_dot * self.dt
        self.t         += self.dt
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi
        return self.get_state()

    def reset(self):
        self.theta = self.rng.uniform(-np.pi, np.pi)
        self.theta_dot = self.rng.uniform(-1.0, 1.0)
        self.t = 0.0
        self.env_init = True
        return self.get_state()

    def get_state(self):
        if not self.env_init:
            raise ValueError("Call reset() before calling get_state()")
        phase = self.drive_omega * self.t
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot,
                        np.cos(phase), np.sin(phase)])
    
    def get_metadata(self):
        return {
            "name": self.name,
            "dt": self.dt,
            "env_seed": self.env_seed,
            "params": {"gravity": self.gravity, "length": self.length, "mass1":
                       self.mass1, "drive_omega": self.drive_omega, "drive_amp": self.drive_amp, "damping": self.damping},
            "readout_targets": ["g_over_l", "drive_omega",],
            "probe_targets": ["cos_theta", "sin_theta", "theta_dot", "cos_phase", "sin_phase", "gravity", "length", "g_over_l", "sqrt_g_over_l", "drive_omega"]
                }