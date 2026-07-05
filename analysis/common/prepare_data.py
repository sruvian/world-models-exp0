# common/prepare_data.py
import numpy as np
import torch
from .regime import probeable_vars, INTERVENTION
from .activations import collect_activations

def build_targets(states, g_flat, l_flat, regime, is_cartpole):
    """Targets ONLY for variables that vary in this regime (via probeable_vars)."""
    vars_here = probeable_vars(regime, is_cartpole)
    t = {}
    if "theta" in vars_here:
        t["theta"] = np.arctan2(states[:, 1], states[:, 0])
    if "theta_dot" in vars_here:
        t["theta_dot"] = states[:, 2]
    if "x" in vars_here:
        t["x"] = states[:, 3]
    if "x_dot" in vars_here:
        t["x_dot"] = states[:, 4]
    if "gravity" in vars_here:
        t["gravity"] = g_flat
    if "length" in vars_here:
        t["length"] = l_flat
    if "g_over_l" in vars_here:
        t["g_over_l"] = g_flat / l_flat
    if "sqrt_l_over_g" in vars_here:
        t["sqrt_l_over_g"] = np.sqrt(l_flat / g_flat)
    return t

def prepare_probe_data(model, all_states, gravities, lengths, regime,
                       is_cartpole=False):
    cur_list, nxt_list, g_list, l_list, cfg_list = [], [], [], [], []

    for ci, (s, g, l) in enumerate(zip(all_states, gravities, lengths)):
        D = s.shape[-1]
        cur = s[:, :-1, :].reshape(-1, D)
        nxt = s[:, 1:,  :].reshape(-1, D)
        n = cur.shape[0]

        cur_list.append(cur)
        nxt_list.append(nxt)
        g_list.append(np.full(n, float(g)))
        l_list.append(np.full(n, float(l)))
        cfg_list.append(np.full(n, ci, dtype=int))

    current = np.concatenate(cur_list, axis=0).astype(np.float32)
    next_   = np.concatenate(nxt_list, axis=0).astype(np.float32)
    g_flat  = np.concatenate(g_list)
    l_flat  = np.concatenate(l_list)
    config_labels = np.concatenate(cfg_list)
    action = torch.zeros(len(current), 1)
    acts, timesteps = collect_activations(model, torch.from_numpy(current), action)

    current_targets = build_targets(current, g_flat, l_flat, regime, is_cartpole)
    next_targets    = build_targets(next_,   g_flat, l_flat, regime, is_cartpole)

    return acts, timesteps, current_targets, next_targets, config_labels