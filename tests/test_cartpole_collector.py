import numpy as np
import matplotlib.pyplot as plt
from collector import collect_trajectories
from sim_envs import make_env

env = make_env("CartPoleSim",
    gravity=9.8, cart_mass=1.0, pole_mass=0.1,
    pole_length=1.0, dt=0.02, max_force=10.0,
    damping=0.0, seed=42
)

states, actions, metadata = collect_trajectories(env, 2, 500, 35, False)
print(f"States shape: {states.shape}")   # (2, 500, 5)
print(f"Actions shape: {actions.shape}") # (2, 500)
print(f"State means: {states.mean(axis=(0,1))}")
print(f"State stds: {states.std(axis=(0,1))}")
print(f"Metadata: {metadata}")