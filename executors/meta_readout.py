import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from analysis import get_data
from analysis import parse_model, probeable_vars, make_metadata, INTERVENTION, prepare_probe_data, integrated_gradients, iter_model_groups, load_model
from models.model import make_model

def readout_probe(z, target, **ctx):
    finite = np.isfinite(z).all(axis=1) & np.isfinite(target)
    if not finite.any():
        return None                          
    z, target = z[finite], target[finite]
    if np.abs(z).max() > 1e4:
        return None
    if z.shape[0] > 100000:
        idx = np.random.default_rng(0).choice(z.shape[0], 100000, replace=False)
        z, target = z[idx], target[idx]
    return Ridge(alpha=ctx.get("alpha", 10.0)).fit(z, target).coef_

def generate_latents_rollout(model, states: torch.Tensor, actions: torch.Tensor,
                             full: bool = True, T_roll: int | None = None, probe_target = "full") -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        N, T, _ = states.shape
        T_roll = T - 1 if T_roll is None else min(T_roll, T - 1)

        comp = model.encode_computational(states[:, 0, :])
        outs = []
        for t in range(T_roll):
            a = actions[:, t]
            if a.dim() == 1:
                a = a.unsqueeze(-1)

            comp = model.step_computational(comp, a)
            obs_comp = model.encode_computational(states[:, t + 1, :])
            if isinstance(comp, tuple):
                h_acc, _ = comp
                _, z_obs = obs_comp if isinstance(obs_comp, tuple) else (None, obs_comp)
                comp = (h_acc, z_obs)
                if probe_target == "full":
                    outs.append(torch.cat(comp, dim=-1) if full else z_obs)
                elif probe_target == 'h':
                    outs.append(h_acc)
                elif probe_target == 'z':
                    outs.append(z_obs)
            else:
                outs.append(model.probe_representation(comp) if not full else comp)

        return torch.stack(outs, dim=1)

def readout_ig(z, target, **ctx):
    if ctx["channel"] is None:
        return None
    model, channel, action = ctx["model"], ctx["channel"], ctx["action"]
    z_t = torch.tensor(z, dtype=torch.float32)
    if z_t.shape[0] > 500:
        idx = torch.randperm(z_t.shape[0])[:500]
        z_t = z_t[idx]
    baseline = z_t.mean(0, keepdim=True)
    out_func = lambda zz: model.decode_computational(model.step_computational(zz, action.expand(zz.shape[0], -1)))[:, channel]
    ig = integrated_gradients(out_func, z_t, baseline, num_steps=50)
    lhs = ig.sum(-1)
    rhs = out_func(z_t) - out_func(baseline)
    completeness_err = (lhs - rhs).abs().mean()
    print(f"IG completeness error for channel {channel}: {completeness_err:.4f}")
    return ig.mean(0).detach().numpy()

def build_targets(states, g_t, l_t, regime, is_cartpole, repr="latent", T_roll = 1):
        vars_here = probeable_vars(regime, is_cartpole)
        t = {}
        if repr == 'latent_full':
            s = states[:, 1:T_roll + 1, :]
            g = g_t[:, 1:T_roll + 1]
            l = l_t[:, 1:T_roll + 1]
            if "cos_theta" in vars_here:      t["cos_theta"] =  s[:, :, 0].reshape(-1).numpy()
            if "sin_theta" in vars_here:      t["sin_theta"] =  s[:, :, 1].reshape(-1).numpy()
            if "theta_dot" in vars_here:  t["theta_dot"] = s[:, :, 2].reshape(-1).numpy()
            if "x" in vars_here:          t["x"] = s[:, :, 3].reshape(-1).numpy()
            if "x_dot" in vars_here:      t["x_dot"] = s[:, :, 4].reshape(-1).numpy()
            if "gravity" in vars_here:    t["gravity"] = g.reshape(-1).numpy()
            if "length" in vars_here:     t["length"] = l.reshape(-1).numpy()
            if "g_over_l" in vars_here:   t["g_over_l"] = (g / l).reshape(-1).numpy()
            if "sqrt_l_over_g" in vars_here: t["sqrt_l_over_g"] = torch.sqrt(l / g).reshape(-1).numpy()
        
        if repr == "latent":
            if "cos_theta" in vars_here:      t["cos_theta"] =  states[:, :, 0].reshape(-1).numpy()
            if "sin_theta" in vars_here:      t["sin_theta"] =  states[:, :, 1].reshape(-1).numpy()
            if "theta_dot" in vars_here:  t["theta_dot"] = states[:, :, 2].reshape(-1).numpy()
            if "x" in vars_here:          t["x"] = states[:, :, 3].reshape(-1).numpy()
            if "x_dot" in vars_here:      t["x_dot"] = states[:, :, 4].reshape(-1).numpy()
            if "gravity" in vars_here:    t["gravity"] = g_t.reshape(-1).numpy()
            if "length" in vars_here:     t["length"] = l_t.reshape(-1).numpy()
            if "g_over_l" in vars_here:   t["g_over_l"] = (g_t / l_t).reshape(-1).numpy()
            if "sqrt_l_over_g" in vars_here: t["sqrt_l_over_g"] = torch.sqrt(l_t / g_t).reshape(-1).numpy()
        elif repr == 'state':
            if "gravity" in vars_here:    t["gravity"] = g_t.reshape(-1).numpy()
            if "length" in vars_here:     t["length"] = l_t.reshape(-1).numpy()
            if "g_over_l" in vars_here:   t["g_over_l"] = (g_t / l_t).reshape(-1).numpy()
            if "sqrt_l_over_g" in vars_here: t["sqrt_l_over_g"] = torch.sqrt(l_t / g_t).reshape(-1).numpy()
        return t
READOUT_METHODS = {"probe": readout_probe, "ig": readout_ig}




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--readout_type", required=True, choices=list(READOUT_METHODS))
    ap.add_argument("--save_dir", default="readouts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--representation", default="computational",
                choices=["computational", "rollout_h", "rollout_z", "rollout_full"])
    ap.add_argument("--t_roll", type=int, default=None)
    args = ap.parse_args()

    groups = iter_model_groups(args.models_dir)
    for (env_tag, policy_tag, model_tag), model_files in groups.items():
        out_sub = Path(args.save_dir) / f"{env_tag}_{policy_tag}_{model_tag}"
        out_sub.mkdir(parents=True, exist_ok=True)

        for mf in model_files:
            cfg = parse_model(Path(mf))
            if cfg["regime"] is None:
                continue
            is_cp = cfg["env"] == "CartPoleSim"
            model = load_model(mf, cfg, args.device)

            states, gravities, lengths = get_data(cfg)
            acts, timesteps, cur_t, nxt_t, config_labels = prepare_probe_data(
                model, states, gravities, lengths, cfg["regime"], is_cartpole=is_cp)

            vars_here = probeable_vars(cfg["regime"], is_cp)

            if args.representation == "computational":
                z = acts["computational"]
                z = z.numpy() if hasattr(z, "numpy") else z
                ts = timesteps.get("computational", "current")
                targets = cur_t if ts == "current" else nxt_t
            else:
                probe = {"rollout_h": "h", "rollout_z": "z",
                         "rollout_full": "full"}[args.representation]

                z_parts, s_parts, g_parts, l_parts = [], [], [], []
                T_eff = None
                for s_np, g, l in zip(states, gravities, lengths):
                    s = torch.from_numpy(s_np).float()
                    a = torch.zeros(s.shape[0], s.shape[1], 1)
                    z_roll = generate_latents_rollout(model, s, a,
                                                      T_roll=args.t_roll,
                                                      probe_target=probe)
                    T_eff = z_roll.shape[1]
                    z_parts.append(z_roll.reshape(-1, z_roll.shape[-1]))
                    s_parts.append(s[:, 1:T_eff + 1, :])
                    n_rows = z_roll.shape[0] * T_eff
                    g_parts.append(np.full(n_rows, float(g), dtype=np.float32))
                    l_parts.append(np.full(n_rows, float(l), dtype=np.float32))

                z = torch.cat(z_parts, 0).numpy()
                s_all = torch.cat(s_parts, 0)
                g_flat = np.concatenate(g_parts)
                l_flat = np.concatenate(l_parts)

                targets = {}
                if "cos_theta" in vars_here: targets["cos_theta"] = s_all[:, :, 0].reshape(-1).numpy()
                if "sin_theta" in vars_here: targets["sin_theta"] = s_all[:, :, 1].reshape(-1).numpy()
                if "theta_dot" in vars_here: targets["theta_dot"] = s_all[:, :, 2].reshape(-1).numpy()
                if "x" in vars_here:         targets["x"] = s_all[:, :, 3].reshape(-1).numpy()
                if "x_dot" in vars_here:     targets["x_dot"] = s_all[:, :, 4].reshape(-1).numpy()
                if "gravity" in vars_here:   targets["gravity"] = g_flat
                if "length" in vars_here:    targets["length"] = l_flat
                if "g_over_l" in vars_here:  targets["g_over_l"] = g_flat / l_flat
                if "sqrt_l_over_g" in vars_here: targets["sqrt_l_over_g"] = np.sqrt(l_flat / g_flat)

            action = torch.zeros(1, 1)

            for variable in vars_here:
                target = targets[variable]
                assert z.shape[0] == target.shape[0], \
                    f"{variable}: z rows {z.shape[0]} != target rows {target.shape[0]}"
                ctx = {"model": model, "channel": INTERVENTION[variable].get("channel"),
                       "action": action, "alpha": 10.0}
                direction = READOUT_METHODS[args.readout_type](z, target, **ctx)
                if direction is None:
                    print(f"  [skip] {variable}: non-finite latents")
                    continue
                meta = make_metadata(args.readout_type, variable, args.representation,
                                     cfg["model_name"], cfg["latent"], cfg["regime"], cfg["seed"])
                meta["checkpoint"] = str(Path(mf))
                meta["t_roll"] = args.t_roll or 0
                meta["dim"] = int(z.shape[-1])
                np.savez(out_sub / f"{args.readout_type}_{args.representation}_{variable}_{Path(mf).stem}.npz",
                         direction=direction, **meta)
        print(f"[{env_tag}_{policy_tag}_{model_tag}] {len(model_files)} models")