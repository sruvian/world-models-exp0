import glob, re
from pathlib import Path
import numpy as np
from collector.collect import collect_trajectories
from sim_envs import make_env
import torch

COLLECTOR = {"num_trajectories": 50, "episode_time": 100, "policy_seed": 35,
             "save": False, "impulse_policy": False}


def gl_from_filename(path):
    name = Path(path).name
    g_match = re.search(r'gravity(\d+\.?\d*)', name) or re.search(r'grav(\d+\.?\d*)', name)
    l_match = re.search(r'length(\d+\.?\d*)', name)
    if g_match is None or l_match is None:
        raise ValueError(f"Could not parse g/l from filename: {name}")
    return float(g_match.group(1)), float(l_match.group(1))


def _make_env_for_config(env_name, g, l, seed):
    if env_name == "CartPoleSim":
        return make_env("CartPoleSim", gravity=g, mass1=0.1, mass2=1.0,
                        length=l, dt=0.01, max_action=10.0, damping=0.0, seed=seed)
    return make_env("PendulumSim", gravity=g, mass1=1.0, mass2=0.0,
                    length=l, dt=0.01, max_action=10.0, damping=0.0, seed=seed)


def load_from_files(regime, env, impulse_policy):
    base = "datasets/impulse_policy" if impulse_policy else "datasets"
    if regime == "combined":
        pattern = f"{base}/*{env}*"
    elif regime == "holdg":
        pattern = f"{base}/*{env}*gra*9.8*"
    elif regime == "holdl":
        pattern = f"{base}/*{env}*len*10*"
    else:
        raise ValueError(f"load_from_files called with non-multi regime: {regime}")

    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No datasets matched: {pattern}")

    states, gravities, lengths = [], [], []
    for f in files:
        data = np.load(f)
        g, l = gl_from_filename(f)
        states.append(data["states"])
        gravities.append(g)
        lengths.append(l)
    return states, gravities, lengths




def get_data(cfg):
    regime = cfg["regime"]
    env    = cfg["env"]
    impulse = cfg["impulse"]

    if regime in ("combined", "holdg", "holdl"):
        return load_from_files(regime, env, impulse)
    else:
        return generate_fresh(env, cfg["g"], cfg["l"], impulse)
    
def _collect(g, l, env, impulse=False, seed=100, n_traj=None, steps=None, policy_seed=None):
    env_obj = _make_env_for_config(env, g, l, seed=seed)
    s, a, _ = collect_trajectories(
        env_obj,
        n_traj or COLLECTOR["num_trajectories"],
        steps or COLLECTOR["episode_time"],
        policy_seed or COLLECTOR["policy_seed"],
        COLLECTOR["save"],
        impulse,
    )
    return torch.from_numpy(s).float(), torch.from_numpy(a).float()

def generate_fresh(env, g, l, impulse_policy):
    s, _ = _collect(g, l, env, impulse=impulse_policy)
    return [s.numpy()], [float(g)], [float(l)]

def collect_for_config(g, l, env, impulse=False, seed=100,
                       n_traj=None, steps=None, policy_seed=None):
    return _collect(g, l, env, impulse=impulse, seed=seed,
                    n_traj=n_traj, steps=steps, policy_seed=policy_seed)