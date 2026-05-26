import numpy as np


class CartPoleSim():

    def __init__(self, gravity: float, mass1: float, mass2: float,length:float, dt: float, max_action: float, damping: float, seed: int ) -> None:
        
        self.gravity = gravity
        self.mass1 = mass1
        self.mass2 = mass2
        self.length = length * 0.5
        self.mcmp = self.mass2 + self.mass1
        self.dt = dt
        self.damping = damping
        self.max_action = max_action
        
        self.mpl = self.mass1 * self.length
        self.theta = 0.0
        self.theta_dot = 0.0

        self.x = 0.0
        self.x_dot = 0.0

        self.env_init = False
        self.env_seed = seed

        self.rng = np.random.default_rng(seed)

    def step(self, action: float)-> np.ndarray:
        if not self.env_init:
            raise ValueError("Call reset() before calling step")
        action = float(action)
        action = np.clip(action, -self.max_action, self.max_action)

        cos_th = np.cos(self.theta)
        sin_th = np.sin(self.theta)

        mpl_sin = self.mpl * (self.theta_dot ** 2) * sin_th

        th_temp = (action + mpl_sin) / (self.mcmp)
        th_num = (self.gravity * sin_th) - (cos_th * th_temp)
        th_den = (4/3) -( (self.mass1 *(cos_th ** 2)) / self.mcmp )

        theta_ddot = th_num / (self.length * th_den)
        x_ddot = (action + mpl_sin - (self.mpl * theta_ddot * cos_th)) / self.mcmp

        self.theta_dot += theta_ddot * self.dt
        self.theta_dot -= self.damping * self.theta_dot * self.dt
        self.theta += self.theta_dot * self.dt

        self.x_dot += x_ddot * self.dt
        self.x_dot -= self.damping * self.x_dot * self.dt
        self.x += self.x_dot * self.dt

        self.theta = ((self.theta + np.pi) % (2*np.pi)) - np.pi
        self.x = np.clip(self.x, -4.8, 4.8)
        return self.get_state()

    def reset(self)-> np.ndarray:
        self.theta = self.rng.uniform(-np.pi, np.pi)
        self.theta_dot = self.rng.uniform(-1.0, 1.0)

        self.x = self.rng.uniform(-2.4, 2.4)
        self.x_dot = self.rng.uniform(-0.5, 0.5)

        self.env_init = True
        return self.get_state()

    def get_state(self) -> np.ndarray:
        if not self.env_init:
            raise ValueError("Call reset() before calling get_state()")
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot, self.x, self.x_dot])
    
    def get_metadata(self) -> dict:
        return {
            "gravity": self.gravity,
            "mass1": self.mass1,
            "mass2": self.mass2,
            "length": self.length * 2,  # full length
            "dt": self.dt,
            "damping": self.damping,
            "env_seed": self.env_seed,
        }