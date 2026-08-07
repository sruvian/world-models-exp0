import numpy as np

class Environment:
    def __init__(self, dt: float, seed: int, **kwargs):
        self.dt = dt
        self.env_seed = seed
        self.rng = np.random.default_rng(seed)
        self.env_init = False
        self.name = "base"

    def step(self, action): raise NotImplementedError
    def reset(self):        raise NotImplementedError
    def get_state(self):    raise NotImplementedError
    def get_metadata(self): raise NotImplementedError