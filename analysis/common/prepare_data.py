import numpy as np
import torch
from .activations import collect_activations
from .utils import STATE_CHANNELS, DERIVED_TARGETS



def build_targets(states, params_flat, probe_targets, env_name):

    channels = STATE_CHANNELS.get(env_name, {})
    t = {}
    for name in probe_targets:
        if name in channels:
            t[name] = states[:, channels[name]]
        elif name in DERIVED_TARGETS:
            t[name] = _derived_per_sample(name, params_flat)
    return t


def _derived_per_sample(name, params_flat):
    fn = DERIVED_TARGETS[name]
    keys = params_flat.keys()
    N = len(next(iter(params_flat.values())))
    out = np.empty(N, dtype=np.float32)
    try:
        out = fn({k: params_flat[k] for k in keys}).astype(np.float32)
    except Exception:
        for i in range(N):
            out[i] = fn({k: params_flat[k][i] for k in keys})
    return out


def prepare_probe_data(model, all_states, params_list, probe_targets, env_name):

    cur_list, nxt_list, cfg_list = [], [], []
    param_keys = sorted({k for p in params_list for k in p})
    param_accum = {k: [] for k in param_keys}

    for ci, (s, params) in enumerate(zip(all_states, params_list)):
        D = s.shape[-1]
        cur = s[:, :-1, :].reshape(-1, D)
        nxt = s[:, 1:,  :].reshape(-1, D)
        n = cur.shape[0]

        cur_list.append(cur)
        nxt_list.append(nxt)
        cfg_list.append(np.full(n, ci, dtype=int))
        for k in param_keys:
            param_accum[k].append(np.full(n, float(params.get(k, np.nan))))

    current = np.concatenate(cur_list, axis=0).astype(np.float32)
    next_   = np.concatenate(nxt_list, axis=0).astype(np.float32)
    config_labels = np.concatenate(cfg_list)
    params_flat = {k: np.concatenate(v) for k, v in param_accum.items()}

    action = torch.zeros(len(current), 1)
    acts, timesteps = collect_activations(model, torch.from_numpy(current), action)

    current_targets = build_targets(current, params_flat, probe_targets, env_name)
    next_targets    = build_targets(next_,   params_flat, probe_targets, env_name)

    return acts, timesteps, current_targets, next_targets, config_labels