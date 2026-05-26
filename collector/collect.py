
import numpy as np
import os
from tqdm import tqdm

from sim_envs.pendulum import PendulumSim

def collect_trajectories(env, num_trajectories: int, episode_time: int, policy_seed:int, save: bool = True)-> tuple[np.ndarray, np.ndarray, dict[str, int|float]]:

    rng = np.random.default_rng(policy_seed)
    states = []
    actions = []
    
    
    for trajectory in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        init_state =  env.reset()
        trajectory_states = []
        trajectory_actions = []
        trajectory_states.append(init_state)
        for time_step in range(episode_time):
            max_action = getattr(env, 'max_torque', getattr(env, 'max_force', 1.0))
            action = rng.uniform(-max_action, max_action)
            trajectory_actions.append(action)
            next_state = env.step(action)
            trajectory_states.append(next_state)
        
        states.append(trajectory_states) # Shape: (N, T+1, 3) due to initial state state
        actions.append(trajectory_actions) # Shape: (N, T)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    metadata = {
        "num_trajectories": num_trajectories,
        "episode_time": episode_time,
        "pol_seed": policy_seed,
        **env.get_metadata()
    }

    if save:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        datasets_dir = os.path.join(base_dir, "..", "datasets")

        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
        meta = env.get_metadata()
        meta_str = "_".join(f"{k}{v:.3f}" if isinstance(v, float) else f"{k}{v}"
                            for k, v in meta.items()
                            if k not in ("env_seed", "dt", "damping"))
        save_file = os.path.join(datasets_dir,
            f"v0_{env.__class__.__name__}_N{num_trajectories}_T{episode_time}"
            f"_env{meta['env_seed']}_pol{policy_seed}"
            f"_dt{meta['dt']:.3f}_damp{meta['damping']:.3f}"
            f"_{meta_str}.npz")

        np.savez(save_file,
                states = states,
                actions = actions,
                **metadata)

    return states, actions, metadata