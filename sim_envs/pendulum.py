import numpy as np


class PendulumSim():

    def __init__(self, gravity: float, mass1: float, length: float, dt: float, max_action: float, damping: float, seed: int, mass2 = 0) -> None:
        
        # Physical parameters

        self.gravity = gravity #m/s
        self.mass1 = mass1 # Kg
        self.length = length #metres
        self.dt = dt #seconds
        self.max_action = max_action #Newton metres
        self.damping = damping


        self.gl = self.gravity/ self.length
        self.ml2 = self.mass1 * (self.length**2)
        
        # Internal state
        self.theta = 0.0
        self.theta_dot = 0.0
        self.env_init = False
        self.env_seed = seed
        self.rng = np.random.default_rng(seed)

    def step(self, action: float)-> np.ndarray:
        if not self.env_init:
            raise ValueError("Call reset() before calling step")
        action = float(action)
        action = np.clip(action, -self.max_action, self.max_action)        
        theta_ddot: float = - self.gl * np.sin(self.theta) + (action / self.ml2) - (self.damping * self.theta_dot / self.ml2)
        
        self.theta_dot += theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt

        self.theta = ((self.theta + np.pi) % (2*np.pi)) - np.pi

        return self.get_state()

    def reset(self)-> np.ndarray:
        self.theta = self.rng.uniform(-np.pi, np.pi)
        self.theta_dot = self.rng.uniform(-1.0, 1.0)
        self.env_init = True
        return self.get_state()

    def get_state(self) -> np.ndarray:
        if not self.env_init:
            raise ValueError("Call reset() before calling get_state()")
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])
    
    def get_metadata(self) -> dict:
        return {
            "gravity": self.gravity,
            "mass1": self.mass1,
            "mass2": 0.0,          # placeholder for consistency
            "length": self.length,
            "dt": self.dt,
            "damping": self.damping,
            "env_seed": self.env_seed,
        }