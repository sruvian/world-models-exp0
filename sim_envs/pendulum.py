import numpy as np


class PendulumSim():

    def __init__(self, gravity: float, pen_mass: float, pen_length: float, dt: float, max_torque: float, damping: int, seed: int) -> None:
        
        # Physical parameters

        self.gravity = gravity #m/s
        self.pen_mass = pen_mass # Kg
        self.pen_length = pen_length #metres
        self.dt = dt #seconds
        self.max_torque = max_torque #Newton metres
        self.damping = damping


        self.gl = self.gravity/ self.pen_length
        self.ml2 = self.pen_mass * (self.pen_length**2)
        
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
        action = np.clip(action, -self.max_torque, self.max_torque)        
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
        if self.env_init:
            return np.array([self.theta, self.theta_dot])
        else:
            raise ValueError("Reset the environment to begin simulation")
        
    
    