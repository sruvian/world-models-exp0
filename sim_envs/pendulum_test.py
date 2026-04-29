import numpy as np
import matplotlib.pyplot as plt
from pendulum import PendulumSim

env = PendulumSim(42)

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