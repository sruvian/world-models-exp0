import matplotlib.pyplot as plt
from collector import collect_trajectories
from sim_envs import make_env

env = make_env("PendulumSim", gravity = 9.8, pen_mass = 1.0, pen_length = 10, dt = 0.01, max_torque = 10, damping = 0, seed =  42)
states, actions, metadata = collect_trajectories(env, 2, 2000, 200, False)

print(states.shape)
print(actions.shape)
print(states.mean(axis=(0,1)))
print(states.std(axis=(0,1)))
print(f"Metadata: {metadata}")