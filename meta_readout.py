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


READOUT_METHODS = {"probe": readout_probe, "ig": readout_ig}




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--readout_type", required=True, choices=list(READOUT_METHODS))
    ap.add_argument("--out_dir", default="readouts")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    groups = iter_model_groups(args.models_dir)
    for (env_tag, policy_tag, model_tag), model_files in groups.items():
        out_sub = Path(args.out_dir) / f"{env_tag}_{policy_tag}_{model_tag}"
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

            z = acts["computational"]
            z = z.numpy() if hasattr(z, "numpy") else z
            action = torch.zeros(1, 1)

            for variable in probeable_vars(cfg["regime"], is_cp):
                ts = timesteps.get("computational", "current")
                target = (cur_t if ts == "current" else nxt_t)[variable]
                ctx = {"model": model, "channel": INTERVENTION[variable].get("channel"),
                       "action": action, "alpha": 10.0}
                direction = READOUT_METHODS[args.readout_type](z, target, **ctx)
                if direction is None:
                    print(f"  [skip] {variable}: non-finite latents")
                    continue
                meta = make_metadata(args.readout_type, variable, "computational",
                                     cfg["model_name"], cfg["latent"], cfg["regime"], cfg["seed"])
                meta["checkpoint"] = str(Path(mf))
                np.savez(out_sub / f"{args.readout_type}_{variable}_{Path(mf).stem}.npz",
                         direction=direction, **meta)
        print(f"[{env_tag}_{policy_tag}_{model_tag}] {len(model_files)} models")