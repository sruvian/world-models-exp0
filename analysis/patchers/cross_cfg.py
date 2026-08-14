
import numpy as np
import torch
from analysis.common.regime import checkable_vars, intervention_for
from analysis.common.data_provider import collect_for_config
from models.wmodel import WorldModelRSSM
from analysis.common.utils import rollout_state

GRAVITIES = [5.0, 9.8, 15.0]
LENGTHS   = [2.0, 10.0, 18.0]

HOLDG_FIXED_G = 9.8
HOLDL_FIXED_L = 10.0


def _config_pairs(variable, regime):
    pairs = []

    if variable == "gravity":
        held_lengths = [HOLDL_FIXED_L] if regime == "holdl" else LENGTHS
        for l in held_lengths:
            for i, g_src in enumerate(GRAVITIES):
                for g_tgt in GRAVITIES[i+1:]:
                    pairs.append(((g_src, l), (g_tgt, l)))

    elif variable == "length":
        held_gravities = [HOLDG_FIXED_G] if regime == "holdg" else GRAVITIES
        for g in held_gravities:
            for i, l_src in enumerate(LENGTHS):
                for l_tgt in LENGTHS[i+1:]:
                    pairs.append(((g, l_src), (g, l_tgt)))

    return pairs


N_RAND_DIM_DRAWS = 10

def _apply_patch(z_tgt, z_src, dims, mode, latent_dim, rng, is_rssm, space="z"):
    if is_rssm:
        h_ptc, z_ptc = z_tgt[0].clone(), z_tgt[1].clone()
        if space == "h":
            ptc, src = h_ptc, z_src[0]
        else:
            ptc, src = z_ptc, z_src[1]
        dim_total = ptc.shape[1]

        if mode == "real":
            ptc[:, dims] = src[:, dims]
        elif mode == "rand_values":
            rv = torch.from_numpy(
                rng.standard_normal((ptc.shape[0], len(dims))).astype(np.float32)
            ).to(ptc.device) * src.std()
            ptc[:, dims] = rv
        elif mode == "rand_dims":
            rand_dims = rng.choice(dim_total, size=len(dims), replace=False)
            ptc[:, rand_dims] = src[:, rand_dims]
        return (h_ptc, z_ptc)
    else:
        z_ptc = z_tgt.clone()
        if mode == "real":
            z_ptc[:, dims] = z_src[:, dims]
        elif mode == "rand_values":
            rv = torch.from_numpy(
                rng.standard_normal((z_ptc.shape[0], len(dims))).astype(np.float32)
            ).to(z_ptc.device) * z_src.std()
            z_ptc[:, dims] = rv
        elif mode == "rand_dims":
            rand_dims = rng.choice(latent_dim, size=len(dims), replace=False)
            z_ptc[:, rand_dims] = z_src[:, rand_dims]
        return z_ptc


def cross_config_patch(model, direction, meta, cfg, writer, device, rng,
                       top_ks=(1,2,3,4,5), n_traj=50, steps=100,
                       representation="computational", t_roll=None):
    env_name = cfg["env"]
    variable = meta["variable"]
    if intervention_for(variable, env_name)["type"] != "config":
        return
    if env_name != "PendulumSim" and env_name != "CartPoleSim":
        return

    is_rssm = isinstance(model, WorldModelRSSM)
    space = "h" if representation == "rollout_h" else "z"
    if space == "h" and not is_rssm:
        return

    for (g_src, l_src), (g_tgt, l_tgt) in _config_pairs(variable, cfg["regime"]):
        src_params = {"gravity": g_src, "length": l_src, "mass1": 1.0, "dt": 0.01, "damping": 0.0}
        tgt_params = {"gravity": g_tgt, "length": l_tgt, "mass1": 1.0, "dt": 0.01, "damping": 0.0}
        if env_name == "CartPoleSim":
            src_params.update(mass1=0.1, mass2=1.0); tgt_params.update(mass1=0.1, mass2=1.0)
        src_s, _ = collect_for_config(env_name, src_params, impulse=cfg["impulse"],
                                      seed=4200, n_traj=n_traj, steps=steps)
        tgt_s, _ = collect_for_config(env_name, tgt_params, impulse=cfg["impulse"],
                                      seed=4200, n_traj=n_traj, steps=steps)
        if representation.startswith("rollout_"):
            T = t_roll if t_roll else steps - 1
            z_src = rollout_state(model, src_s, T, device)
            z_tgt = rollout_state(model, tgt_s, T, device)
            n = (z_src[0] if is_rssm else z_src).shape[0]
        else:
            D = src_s.shape[-1]
            n = min(src_s.reshape(-1, D).shape[0], tgt_s.reshape(-1, D).shape[0])
            z_src = model.encode_computational(src_s.reshape(-1, D).float().to(device)[:n])
            z_tgt = model.encode_computational(tgt_s.reshape(-1, D).float().to(device)[:n])

        action = torch.zeros(n, 1, device=device)
        latent_dim = (z_src[1].shape[1] if is_rssm else z_src.shape[1])

        s_tgt = model.decode_computational(model.step_computational(z_tgt, action))
        s_src = model.decode_computational(model.step_computational(z_src, action))
        base_dist = ((s_tgt[:, :2] - s_src[:, :2])**2).mean().sqrt().item()

        for top_k in top_ks:
            dims = np.argsort(np.abs(direction))[-top_k:]
            for mode in ["real", "rand_values", "rand_dims"]:
                n_draws = N_RAND_DIM_DRAWS if mode == "rand_dims" else 1
                patch_dists = []
                for _ in range(n_draws):
                    z_patched = _apply_patch(z_tgt, z_src, dims, mode,
                                             latent_dim, rng, is_rssm, space)
                    s_patch = model.decode_computational(
                        model.step_computational(z_patched, action))
                    patch_dists.append(
                        ((s_patch[:, :2] - s_src[:, :2])**2).mean().sqrt().item())
                patch_dist = float(np.mean(patch_dists))
                writer.writerow([
                    meta["checkpoint"], variable,
                    f"src_g{g_src}_l{l_src}", f"tgt_g{g_tgt}_l{l_tgt}",
                    cfg["latent"], cfg["k"], top_k, mode,
                    round(base_dist - patch_dist, 6), round(base_dist, 6),
                    round(patch_dist, 6),
                ])