import glob
from pathlib import Path
import numpy as np
import torch
from sim_envs import make_env
from collector import collect_trajectories
from .utils import ENV_NAMES

COLLECTOR = {"num_trajectories": 50, "episode_time": 100, "policy_seed": 35,
             "save": False, "impulse_policy": False}


def config_from_file(path):
    d = np.load(path)
    reserved = {"states", "actions", "readout_targets", "probe_targets", "name"}
    params = {}
    for k in d.files:
        if k in reserved or d[k].ndim != 0:
            continue
        v = d[k]
        if not np.issubdtype(v.dtype, np.number):
            continue
        params[k] = float(v)
    targets = d["readout_targets"].tolist() if "readout_targets" in d.files else []
    probes  = d["probe_targets"].tolist()  if "probe_targets"  in d.files else []
    return params, targets, probes

def make_env_from_params(env_name, params, seed, **overrides):
    return make_env(env_name, **{**params, "seed": seed, **overrides})


def _dataset_glob(env, impulse_policy, base_dir):
    base = f"{base_dir}/impulse_policy" if impulse_policy else f"{base_dir}"
    return f"{base}/{ENV_NAMES[env]}/*"


def load_by_regime(env, regime, impulse_policy=False,
                   hold_param=None, hold_value=None, tol=1e-6, base_dir = "datasets"):
    files = sorted(glob.glob(_dataset_glob(env, impulse_policy, base_dir)))
    if not files:
        raise FileNotFoundError(f"No datasets matched env={env} impulse={impulse_policy}")

    states, params_list = [], []
    for f in files:
        params, _, _ = config_from_file(f)
        if regime == "hold":
            if hold_param is None:
                raise ValueError("regime='hold' requires hold_param and hold_value")
            if abs(params.get(hold_param, float("inf")) - hold_value) > tol:
                continue
        elif regime != "combined":
            raise ValueError(f"unknown regime: {regime}")
        states.append(np.load(f)["states"])
        params_list.append(params)

    if not states:
        raise FileNotFoundError(f"No files after filter: {regime} {hold_param}={hold_value}")
    return states, params_list


def _collect(env, params, seed=100, n_traj=None, steps=None,
             policy_seed=None, impulse=False, max_action=10.0):
    env_obj = make_env_from_params(env, params, seed=seed, max_action=max_action)
    s, a, _ = collect_trajectories(
        env = env_obj,
        num_trajectories=n_traj or COLLECTOR["num_trajectories"],
        episode_time= steps or COLLECTOR["episode_time"],
        policy_seed = policy_seed or COLLECTOR["policy_seed"],
        save = COLLECTOR["save"],
        impulse_policy=impulse,
    )
    return torch.from_numpy(s).float(), torch.from_numpy(a).float()


def collect_for_config(env, params, impulse=False, seed=100,
                       n_traj=None, steps=None, policy_seed=None, max_action=10.0):
    return _collect(env, params, seed=seed, n_traj=n_traj, steps=steps,
                    policy_seed=policy_seed, impulse=impulse, max_action=max_action)


def generate_fresh(env, params, impulse_policy):
    s, _ = _collect(env, params, impulse=impulse_policy)
    return [s.numpy()], [params]


def get_data(cfg):
    env, regime, impulse = cfg["env"], cfg["regime"], cfg["impulse"]
    if regime == "combined":
        states, params_list = load_by_regime(env, "combined", impulse)
    elif regime == "hold":
        states, params_list = load_by_regime(env, "hold", impulse,
                                             hold_param=cfg["hold_param"], hold_value=cfg["hold_value"])
    elif regime == "single":
        states, params_list = generate_fresh(env, cfg["params"], impulse)
    else:
        raise ValueError(f"unknown regime: {regime}")
    probe_targets = make_env_from_params(env, params_list[0], seed=0, max_action=0).get_metadata()["probe_targets"]
    return states, params_list, probe_targets