import matplotlib.pyplot as plt
from .collect import collect_trajectories
from sim_envs import PendulumSim

env = PendulumSim(42)
states, actions = collect_trajectories(env, 2, 2000, 200)

print(states.shape)
print(actions.shape)
print(states.mean(axis=(0,1)))
print(states.std(axis=(0,1)))