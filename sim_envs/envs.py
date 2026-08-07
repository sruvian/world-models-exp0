from .pendulum import PendulumSim
from .cartpole import CartPoleSim
from .driv_pendulum import DrivenPendulumSim
from .base_environment import Environment
from typing import TypeVar

E = TypeVar("E", bound=Environment)
envs = {"PendulumSim": PendulumSim, "CartPoleSim": CartPoleSim, "DrivenPendulumSim": DrivenPendulumSim}

def make_env(env_name, **kwargs)-> E:
    if env_name not in envs:
        raise ValueError("Environment unavailable")
    
    environment = envs[env_name]

    return environment(**kwargs)
