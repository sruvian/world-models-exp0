
import numpy as np
import os
from tqdm import tqdm


class SparseImpulsePolicy:
    def __init__(self, max_action: float, dt: float = 0.01, min_gap: float = 0.05, max_gap: float = 0.2, min_duration: float = 0.01, max_duration: float = 0.05,
                 seed: int = 35) -> None:
        
        self.max_action = max_action

        self.dt = dt

        self.min_gap = min_gap
        self.max_gap = max_gap

        self.min_duration = min_duration
        self.max_duration = max_duration
        
        self.rng = np.random.default_rng(seed)
        
        self._schedule_next()

    def _schedule_next(self) -> None:
        
        gap = self.rng.uniform(self.min_gap, self.max_gap)
        
        self.steps_until_impulse = int(gap / self.dt)
        self.impulse_steps_remaining = 0
        self.current_action = 0.0

    def __call__(self) -> float:
        if self.impulse_steps_remaining > 0:
            self.impulse_steps_remaining -= 1
            if self.impulse_steps_remaining == 0:
                self._schedule_next()
            return self.current_action

        if self.steps_until_impulse > 0:
            self.steps_until_impulse -= 1
            return 0.0

        duration = self.rng.uniform(self.min_duration, self.max_duration)
        self.impulse_steps_remaining = max(1, int(duration / self.dt))
        self.current_action = self.rng.uniform(-self.max_action, self.max_action)
        return self.current_action

    def reset(self) -> None:
        self._schedule_next()


def collect_trajectories(env, num_trajectories: int, episode_time: int, policy_seed:int, save: bool = True, 
                         impulse_policy: bool = False,
                         min_duration: float = 0.01, max_duration: float = 0.05)-> tuple[np.ndarray, np.ndarray, dict[str, int|float]]:

    rng = np.random.default_rng(policy_seed)
    states = []
    actions = []
    policy = None

    max_action = getattr(env, 'max_torque', getattr(env, 'max_force', 1.0))
    if impulse_policy:
        T = 2 * np.pi * np.sqrt(env.length / env.gravity)
        min_gap = 0.05 * T
        max_gap = 0.15 * T
        policy = SparseImpulsePolicy(max_action, getattr(env, 'dt', 0.01), min_gap, max_gap, min_duration, max_duration, policy_seed)
        
    else:
        rng = np.random.default_rng(policy_seed)
    
    for trajectory in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        init_state =  env.reset()
        trajectory_states = []
        trajectory_actions = []
        trajectory_states.append(init_state)
        for time_step in range(episode_time):

            if policy is not None:
                action = policy()
            else:
                action = rng.uniform(-env.max_action, env.max_action)
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

        if impulse_policy:
            datasets_dir = os.path.join(base_dir, "..", "datasets/impulse_policy")
        else:
            datasets_dir = os.path.join(base_dir, "..", "datasets/")

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