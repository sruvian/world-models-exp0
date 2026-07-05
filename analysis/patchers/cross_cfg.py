
import numpy as np
import torch
from analysis.common.regime import checkable_vars, INTERVENTION
from analysis.common.data_provider import collect_for_config

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


def cross_config_patch(model, direction, meta, cfg, writer, device, rng,
                       top_ks=(1, 2, 3, 4, 5), n_traj=50, steps=100):
    variable = meta["variable"]
    if INTERVENTION[variable]["type"] != "config":
        return

    for (g_src, l_src), (g_tgt, l_tgt) in _config_pairs(variable, cfg["regime"]):
        src_s, _ = collect_for_config(g_src, l_src, cfg["env"], cfg["impulse"],
                                      seed=4200, n_traj=n_traj, steps=steps)
        tgt_s, _ = collect_for_config(g_tgt, l_tgt, cfg["env"], cfg["impulse"],
                                      seed=4200, n_traj=n_traj, steps=steps)
        D = src_s.shape[-1]
        src_flat = src_s.reshape(-1, D).float().to(device)
        tgt_flat = tgt_s.reshape(-1, D).float().to(device)

        n = min(src_flat.shape[0], tgt_flat.shape[0])
        src_flat, tgt_flat = src_flat[:n], tgt_flat[:n]

        z_src = model.encode_computational(src_flat)
        z_tgt = model.encode_computational(tgt_flat)
        action = torch.zeros(n, 1, device=device)

        for top_k in top_ks:
            dims = np.argsort(np.abs(direction))[-top_k:]

            z_patched = z_tgt.clone()
            z_patched[:, dims] = z_src[:, dims]
            s_tgt   = model.decode_computational(model.step_computational(z_tgt, action))
            s_patch = model.decode_computational(model.step_computational(z_patched, action))
            s_src   = model.decode_computational(model.step_computational(z_src, action))

            base_dist  = ((s_tgt[:, :2]   - s_src[:, :2]) ** 2).mean().sqrt().item()
            patch_dist = ((s_patch[:, :2] - s_src[:, :2]) ** 2).mean().sqrt().item()
            shift = base_dist - patch_dist

            writer.writerow([
                meta["checkpoint"], variable,
                f"src_g{g_src}_l{l_src}", f"tgt_g{g_tgt}_l{l_tgt}",
                cfg["latent"], cfg["k"], top_k,
                round(shift, 6), round(base_dist, 6), round(patch_dist, 6),
            ])