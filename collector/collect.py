
import numpy as np
import os
from tqdm import tqdm

def collect_trajectories(env, num_trajectories: int, episode_time: int, policy_seed:int, save: bool = True):

    rng = np.random.default_rng(policy_seed)
    states = []
    actions = []
    
    
    for trajectory in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        init_state =  env.reset()
        trajectory_states = []
        trajectory_actions = []
        trajectory_states.append(init_state)
        for time_step in range(episode_time):
            action = rng.uniform(-env.max_torque, env.max_torque)
            trajectory_actions.append(action)
            next_state = env.step(action)
            trajectory_states.append(next_state)
        
        states.append(trajectory_states) # Shape: (N, T+1, 3) due to initial state state
        actions.append(trajectory_actions) # Shape: (N, T)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)


    
    if save:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        datasets_dir = os.path.join(base_dir, "..", "datasets")

        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)

        save_file = os.path.join(datasets_dir,
                                f"v0_{env.__class__.__name__}_N{num_trajectories}_T{episode_time}"
                                f"_env{env.env_seed}_pol{policy_seed}_dt{env.dt:.3f}_damp{env.damping:.3f}"
                                f"_grav{env.gravity:.2f}_mass{env.pen_mass:.2f}_length{env.pen_length:.1f}.npz")

        np.savez(save_file,
                states = states,
                actions = actions,
                num_trajectories = num_trajectories,
                episode_time = episode_time,
                env_seed = env.env_seed,
                pol_seed = policy_seed,
                dt = env.dt,
                damping = env.damping,
                gravity=env.gravity,
                mass=env.pen_mass,
                length=env.pen_length,)

    return (states, actions)