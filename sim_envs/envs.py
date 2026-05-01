from .pendulum import PendulumSim
from .cartpole import CartPoleSim

envs = {"PendulumSim": PendulumSim, "CartpoleSim": CartPoleSim}

def make_env(env_name, **kwargs):
    if env_name not in envs:
        raise ValueError("Environment unavailable")
    
    environment = envs[env_name]

    return environment(**kwargs)
