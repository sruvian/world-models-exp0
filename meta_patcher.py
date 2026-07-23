# meta_patcher.py
import argparse, glob, csv
from pathlib import Path
import numpy as np
import torch
from analysis.common.utils import parse_model, load_model
from analysis.common.data_provider import collect_for_config
from analysis.patchers.activation_patching import patch_trajectories
from analysis.common.utils import TAGS
from analysis.patchers import cross_config_patch 

ALL_CONFIGS = [(5.0,2.0),(5.0,10.0),(5.0,18.0),(9.8,2.0),(9.8,10.0),
               (9.8,18.0),(15.0,2.0),(15.0,10.0),(15.0,18.0)]

def dims_from_direction(direction, top_k):
    return np.argsort(np.abs(direction))[-top_k:]

def run_activation_patch(model, direction, meta, cfg, writer, device, rng):
    eval_configs = ALL_CONFIGS if cfg["flag"] else [(cfg["g"], cfg["l"])]
    for g_eval, l_eval in eval_configs:
        is_ood = (g_eval, l_eval) not in ALL_CONFIGS
        src_s, _ = collect_for_config(g_eval, l_eval, cfg["env"], cfg["impulse"], seed=4200)
        tgt_s, _ = collect_for_config(g_eval, l_eval, cfg["env"], cfg["impulse"], seed=1000000)
        D = src_s.shape[-1]
        src_flat = src_s.reshape(-1, D).float().to(device)
        tgt_flat = tgt_s.reshape(-1, D).float().to(device)

        for top_k in [1, 2, 3, 4, 5]:
            dims = dims_from_direction(direction, top_k)
            for mode in ["real", "rand_values", "rand_dims"]:
                shift, shift_src, bl, pt, bl_s, pt_s = patch_trajectories(
                    model, src_flat, tgt_flat, dims, patch_mode=mode, rng=rng)
                writer.writerow([
                    meta["checkpoint"], meta["variable"], f"g{g_eval}_l{l_eval}",
                    cfg["latent"], cfg["k"], top_k, is_ood, mode,
                    round(shift,6), round(shift_src,6), round(bl,6), round(pt,6),
                    round(bl_s,6), round(pt_s,6),
                ])
                
PATCH_METHODS = {
    "activation": run_activation_patch,
    "cross_config": cross_config_patch
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts_dir", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--save_dir",required = True)
    parser.add_argument("--method", required=True, choices=list(PATCH_METHODS))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rng = np.random.default_rng(35)
    model_cache = {}

    
    readout_files = glob.glob(str(Path(args.readouts_dir) / "**" / "*.npz"), recursive=True)

    csv_handles = {}
    for rf in readout_files:
        d = np.load(rf, allow_pickle=True)
        direction = d["direction"]
        meta = {k: d[k].item() for k in d.files if k != "direction"}

        variable = meta["variable"]
        if variable in ("g_over_l", "sqrt_l_over_g"):
            continue

        checkpoint = meta["checkpoint"]
        model_file = Path(checkpoint)
        if not model_file.exists():
            hits = glob.glob(str(Path(args.models_dir) / "**" / Path(checkpoint).name), recursive=True)
            if not hits:
                print(f"[skip] model not found: {checkpoint}"); continue
            if len(hits) > 1:
                print(f"[warn] {len(hits)} matches for {Path(checkpoint).name}, using {hits[0]}")
            model_file = Path(hits[0])

        cfg = parse_model(model_file)
        if cfg["regime"] is None:
            continue
        model_key = str(model_file)
        if model_key not in model_cache:
            model_cache[model_key] = load_model(model_file, cfg, args.device)
        model = model_cache[model_key]

        env_tag = "cartpole" if cfg["env"]=="CartPoleSim" else "pendulum"
        policy_tag = "sparse" if cfg["impulse"] else "noise"
        
        group = f"{env_tag}_{policy_tag}_{TAGS[cfg['model_name']]}"
        if group not in csv_handles:
            path = Path(f"{args.save_dir}/{args.method}_{group}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            hdr = not path.exists()
            fh = open(path, "a", newline="")
            w = csv.writer(fh)
            if hdr:
                if args.method == "cross_config":
                    w.writerow(["checkpoint","target_var","src_config","tgt_config",
                                "latent_dim","k","top_k","patch_mode",
                                "shift","base_dist","patch_dist"])
                else:
                    w.writerow(["checkpoint","target_var","eval_config","latent_dim","k",
                                "top_k","is_ood","patch_mode","shift","shift_source",
                                "baseline_err","patched_err","baseline_err_source","patched_err_source"])
            csv_handles[group] = (fh, w)
        fh, writer = csv_handles[group]

        print(f"[{checkpoint}] var={variable} method={args.method}")
        PATCH_METHODS[args.method](model, direction, meta, cfg, writer, args.device, rng)
        fh.flush()

    for fh, _ in csv_handles.values():
        fh.close()
    print("Done.")