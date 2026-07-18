
import argparse, glob, csv
from pathlib import Path
import numpy as np
import torch
from analysis.common.utils import parse_model, load_model, TAGS
from analysis.common.regime import checkable_vars, INTERVENTION
from analysis.common.data_provider import collect_for_config
from analysis.do_checker.checker import DoChecker
from analysis.do_checker.val_suite import ValidationSuite

ALL_CONFIGS = [(5.0,2.0),(5.0,10.0),(5.0,18.0),(9.8,2.0),(9.8,10.0),
               (9.8,18.0),(15.0,2.0),(15.0,10.0),(15.0,18.0)]
HOLDG_FIXED_G, HOLDL_FIXED_L = 9.8, 10.0

def config_dict(env, g, l):
    if env == "CartPoleSim":
        return dict(gravity=g, mass1=0.1, mass2=1.0, length=l, dt=0.01, max_action=10.0, damping=0.0)
    return dict(gravity=g, mass1=1.0, mass2=0.0, length=l, dt=0.01, max_action=10.0, damping=0.0)


def source_target_configs(variable, cfg):
    iv = INTERVENTION[variable]
    regime = cfg["regime"]

    if iv["type"] == "config":

        if variable == "gravity":
            l_use = HOLDL_FIXED_L if regime == "holdl" else 10.0
            return (5.0, l_use), (15.0, l_use)
        else:  # length
            g_use = HOLDG_FIXED_G if regime == "holdg" else 9.8
            return (g_use, 2.0), (g_use, 18.0)
    else:
        if regime == "combined":
            g, l = 9.8, 10.0
        elif regime == "holdg":
            g, l = HOLDG_FIXED_G, 10.0
        elif regime == "holdl":
            g, l = 9.8, HOLDL_FIXED_L
        else:
            g, l = cfg["g"], cfg["l"]
        return (g, l), (g, l)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--save_dir", required = True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--manifold_mult", type=float, default=2.0)
    args = ap.parse_args()

    csv_handles = {}
    model_cache = {}
    for rf in glob.glob(str(Path(args.readouts_dir) / "**" / "*.npz"), recursive=True):
        d = np.load(rf, allow_pickle=True)
        direction = d["direction"]
        meta = {k: d[k].item() for k in d.files if k != "direction"}
        variable = meta["variable"]

        cfg_stub = {"regime": meta["regime"]}
        if variable not in checkable_vars(meta["regime"], is_cartpole=("cartpole" in meta.get("checkpoint","").lower())):
            continue

        model_file = Path(meta["checkpoint"])
        if not model_file.exists():
            print(f"[skip] model not found: {meta['checkpoint']}"); continue
        mf = str(model_file)
        
        cfg = parse_model(Path(mf))
        is_cp = cfg["env"] == "CartPoleSim"
        model = load_model(mf, cfg, args.device)
        if mf not in model_cache:
            model_cache[mf] = load_model(mf, cfg, args.device)
        model = model_cache[mf]

        checker = DoChecker(
            encode=model.encode_computational,
            step=model.step_computational,
            decode=model.decode_computational,
            env=cfg["env"], manifold_mult=args.manifold_mult)
        suite = ValidationSuite(checker)

        (gs, ls), (gt, lt) = source_target_configs(variable, cfg)
        source_config = config_dict(cfg["env"], gs, ls)
        target_config = config_dict(cfg["env"], gt, lt)
        iv = INTERVENTION[variable]
        channel = iv.get("channel")

        src_s, src_a = collect_for_config(gs, ls, cfg["env"], cfg["impulse"], seed=4200, n_traj=20, steps=50)
        source_states = src_s.reshape(-1, src_s.shape[-1]).float()
        actions = src_a.reshape(-1, 1).float()
        calibration_pool = source_states

        if iv["type"] == "state":
            channel = iv["channel"]
            target_states = source_states.clone()
            delta = 1.0
            target_states[:, channel] = source_states[:, channel] + delta
        else:
            target_states = None


        result = suite.report(
            source_states[:len(actions)], source_config, target_config, channel,
            direction, actions, calibration_pool, target_states)

        env_tag = "cartpole" if is_cp else "pendulum"
        policy_tag = "sparse" if cfg["impulse"] else "noise"
        group = f"{env_tag}_{policy_tag}_{TAGS[cfg['model_name']]}"
        if group not in csv_handles:
            path = Path(f"{args.save_dir}/comparator_{group}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            hdr = not path.exists()
            fh = open(path, "a", newline="")
            w = csv.writer(fh)
            if hdr:
                w.writerow(["checkpoint","variable","latent_dim","k","regime",
                            "ceiling_err","dz_opt_cossim","analytical_search_gap","dy_opt_norm",
                            "dz_probe_cossim","dzopt_probe_cossim","survival","probe_slope","clears_null",
                            "null_95","null_mean","null_max","null_nonnan",
                            "probe_target_value","null_target_mean",
                            "pc1_var","pc1_probe_cos"])
            csv_handles[group] = (fh, w)
        fh, w = csv_handles[group]
        w.writerow([
            meta["checkpoint"], variable, cfg["latent"], cfg["k"], cfg["regime"],
            result["ceiling_err"], result["dz_opt_cossim"], result.get("analytical_search_gap"),
            result["dy_opt_norm"], result["dz_probe_cossim"], result["dzopt_probe_cossim"],
            result["probe_survival"], result["probe_slope"], result["clears_null"],
            result["null_95"], result.get("null_mean"), result.get("null_max"), result.get("null_nonnan"),
            result.get("probe_target_value"), result.get("null_target_mean"),
            result.get("pc1_var"), result.get("pc1_probe_cos"),
        ])
        fh.flush()
        print(f"[{meta['checkpoint']}] {variable}: ceiling={result['ceiling_err']:.4f} "
              f"probe_slope={result['probe_slope']:.4f} clears_null={result['clears_null']}")

    for fh, _ in csv_handles.values():
        fh.close()