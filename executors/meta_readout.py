import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from analysis.common.data_provider import get_data
from analysis.common.prepare_data import prepare_probe_data
from analysis.common.regime import probeable_vars, make_metadata, intervention_for
from analysis.common.utils import parse_model, load_model, iter_model_groups, STATE_CHANNELS, DERIVED_TARGETS
from analysis.patchers.int_gradients import integrated_gradients
from models.model import make_model
from sklearn.linear_model import LinearRegression


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
    
def pred_accel(pred_traj, dt = 0.01):
    theta = torch.arctan2(pred_traj[:, :, 1], pred_traj[:, :, 0])
    theta_u = np.unwrap(theta.detach().numpy(), axis=1)
    thetadot = pred_traj[:, :, 2].detach().numpy()
    theta_ddot = np.gradient(thetadot, dt, axis=1, edge_order=1)
    sin_th = np.sin(theta_u)

    if not (np.isfinite(theta_ddot).all() and np.isfinite(sin_th).all()):
        return float("nan"), float("nan")

    X = (-sin_th).reshape(-1, 1)
    y = theta_ddot.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    float(reg.coef_[0][0])

def pred_accel_pointwise(pred_traj, dt=0.01):
    thetadot = pred_traj[:, :, 2]
    theta_ddot = (thetadot[:, 1] - thetadot[:, 0]) / dt
    return theta_ddot

def readout_ig(z, target, **ctx):
    model, channel, action = ctx["model"], ctx["channel"], ctx["action"]
    z_t = torch.tensor(z, dtype=torch.float32)
    if z_t.shape[0] > 500:
        idx = torch.randperm(z_t.shape[0])[:500]
        z_t = z_t[idx]
    baseline = z_t.mean(0, keepdim=True)
    if channel is None:
        def out_func(zz):
            s_now  = model.decode_computational(zz)
            s_next = model.decode_computational(model.step_computational(zz, action.expand(zz.shape[0], -1)))
            theta_ddot = (s_next[:, 2] - s_now[:, 2]) / 0.01
            return theta_ddot 
    else:
        out_func = lambda zz: model.decode_computational(model.step_computational(zz, action.expand(zz.shape[0], -1)))[:, channel]
    ig = integrated_gradients(out_func, z_t, baseline, num_steps=50)
    lhs = ig.sum(-1)
    rhs = out_func(z_t) - out_func(baseline)
    rel_err = (lhs - rhs).abs().mean() / (rhs.abs().mean() + 1e-8)
    completeness_err = (lhs - rhs).abs().mean()
    print(f"IG completeness error for channel {channel}: {completeness_err:.4f}| Relative Error: {rel_err:.4f}")
    if channel is None:
        with torch.inference_mode():
            sin_th = model.decode_computational(z_t)[:, 1]
        w = -sin_th / (sin_th.pow(2).sum() + 1e-8)
        ig_gl = (w.unsqueeze(-1) * ig).sum(0)
        return ig_gl.detach().numpy()
    return ig.mean(0).detach().numpy()

def _build_probe_targets(states, params_arr, env_name, probe_targets,
                         regime, hold_param, repr, T_roll=1):
    channels = STATE_CHANNELS.get(env_name, {})
    vars_here = probeable_vars(probe_targets, env_name, regime, hold_param)
    t = {}

    if repr == "latent_full":
        s = states[:, 1:T_roll + 1, :]
        P = {k: v[:, 1:T_roll + 1] for k, v in params_arr.items()}
    else:
        s = states
        P = params_arr

    for name in vars_here:
        if name in channels:
            if repr == "state":
                continue
            t[name] = s[:, :, channels[name]].reshape(-1).numpy()
        elif name in DERIVED_TARGETS:
            val = DERIVED_TARGETS[name](P)
            t[name] = np.asarray(val).reshape(-1)
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
            model = load_model(mf, cfg, device=args.device)

            states, params_list, probe_targets = get_data(cfg)
            acts, timesteps, cur_t, nxt_t, config_labels = prepare_probe_data(
                model, states, params_list, probe_targets, cfg["env"])

            vars_here = probeable_vars(probe_targets, cfg["env"], cfg["regime"], cfg.get("hold_param"))
            

            if args.representation == "computational":
                z = acts["computational"]
                z = z.numpy() if hasattr(z, "numpy") else z
                ts = timesteps.get("computational", "current")
                targets = cur_t if ts == "current" else nxt_t
                # with torch.inference_mode():
                #     z_t = torch.tensor(z, dtype=torch.float32)
                #     a0 = torch.zeros(z_t.shape[0], 1)
                #     s_next_pred = model.decode_computational(model.step_computational(z_t, a0))
                #     theta_dot_pred = s_next_pred[:, 2].numpy()
            else:
                probe = {"rollout_h": "h", "rollout_z": "z",
                         "rollout_full": "full"}[args.representation]

                z_parts, s_parts, g_parts, l_parts = [], [], [], []
                T_eff = None
                for s_np, params in zip(states, params_list):
                    g, l = params["gravity"], params["length"]
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
                # if "theta_dot" in vars_here: targets["theta_dot"] = theta_dot_pred
                if "x" in vars_here:         targets["x"] = s_all[:, :, 3].reshape(-1).numpy()
                if "x_dot" in vars_here:     targets["x_dot"] = s_all[:, :, 4].reshape(-1).numpy()
                if "gravity" in vars_here:   targets["gravity"] = g_flat
                if "length" in vars_here:    targets["length"] = l_flat
                if "g_over_l" in vars_here:  targets["g_over_l"] = g_flat / l_flat
                if "sqrt_l_over_g" in vars_here: targets["sqrt_l_over_g"] = np.sqrt(l_flat / g_flat)

            action = torch.zeros(1, 1)

            for variable in vars_here:
                target = targets[variable]
                if variable not in ["g_over_l", "theta_dot"]:
                    continue
                # if "theta_dot" in vars_here:
                #     targets["theta_dot"] = theta_dot_pred
                assert z.shape[0] == target.shape[0], \
                    f"{variable}: z rows {z.shape[0]} != target rows {target.shape[0]}"
                ctx = {"model": model, "channel": intervention_for(variable, env_name=cfg["env"]),
                       "action": action, "alpha": 10.0}
                direction = READOUT_METHODS[args.readout_type](z, target, **ctx)
                if direction is None:
                    print(f"  [skip] {variable}: non-finite latents")
                    continue
                meta = make_metadata(args.readout_type, variable, args.representation,
                     cfg["model_name"], cfg["latent"], cfg["regime"], cfg["seed"],
                     cfg["env"])
                meta["checkpoint"] = str(Path(mf))
                meta["t_roll"] = args.t_roll or 0
                meta["dim"] = int(z.shape[-1])
                np.savez(out_sub / f"{args.readout_type}_{args.representation}_{variable}_{Path(mf).stem}.npz",
                         direction=direction, **meta)
        print(f"[{env_tag}_{policy_tag}_{model_tag}] {len(model_files)} models")