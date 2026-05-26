import numpy as np
import matplotlib.pyplot as plt
from sim_envs import make_env

env = make_env("CartPoleSim",
    gravity=9.8, cart_mass=1.0, pole_mass=0.1,
    pole_length=1.0, dt=0.02, max_force=10.0,
    damping=0.0, seed=42
)

state = env.reset()
print(f"Initial state: cos_th={state[0]:.3f} sin_th={state[1]:.3f} "
      f"th_dot={state[2]:.3f} x={state[3]:.3f} x_dot={state[4]:.3f}")

states = []
for _ in range(500):
    states.append(env.get_state())
    env.step(0.0)

states = np.array(states)
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(states[:, 0], label='cos θ')
axes[0].plot(states[:, 1], label='sin θ')
axes[0].legend()
axes[0].set_title("Angular position")

axes[1].plot(states[:, 2])
axes[1].set_title("Angular velocity θ̇")

axes[2].plot(states[:, 3], label='x')
axes[2].plot(states[:, 4], label='ẋ')
axes[2].legend()
axes[2].set_title("Cart position and velocity")

plt.tight_layout()
plt.show()