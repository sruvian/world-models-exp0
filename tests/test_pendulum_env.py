import numpy as np
import matplotlib.pyplot as plt
from sim_envs import make_env

env = make_env("PendulumSim", gravity = 9.8, pen_mass = 1.0, pen_length = 10, dt = 0.01, max_torque = 10, damping = 0, seed =  42)

initial_state = env.reset()
print(f"Initial Theta: {initial_state[0]}, Initial Ang Velocity: {initial_state[1]}")

states = []
for _ in range(2000):
    states.append(env.get_state())
    env.step(0.0)

states = np.array(states)

plt.figure()
plt.title("Angle over time")
plt.plot(states[:, 0])

plt.figure()
plt.title("Angular velocity over time")
plt.plot(states[:, 1])

plt.show()