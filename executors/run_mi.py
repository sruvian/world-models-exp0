from __future__ import annotations
import argparse, glob, csv
from pathlib import Path
import numpy as np

from analysis.common.utils import parse_model, load_model, TAGS, ENV_NAMES, STATE_CHANNELS, make_model, HIDDEN_DIM
from analysis.common.data_provider import get_data
from analysis.probes.linear_probe import stratified_probe_split
from analysis.probes.smile import mi_dissociation
from analysis.probes.smile import estimate_mi_smile

PARAM_TARGETS = ["g_over_l", "sqrt_g_over_l", "gravity", "length", "drive_omega"]
CONTROL_TARGETS = ["theta_dot", "sin_theta", "cos_phase"]


def _random_model(cfg, device):
    state_dim = len(STATE_CHANNELS[cfg["env"]])
    m = make_model(cfg["model_name"], state_dim=state_dim, action_dim=1,
                   hidden_dim=HIDDEN_DIM.get(cfg["model_name"], 64), latent_dim=cfg["latent"])
    m.eval()
    return m.to(device)


def _normalize_regime(cfg):
    r = cfg.get("regime")
    if r == "holdg":
        cfg["regime"], cfg["hold_param"], cfg["hold_value"] = "hold", "gravity", 9.8
    elif r == "holdl":
        cfg["regime"], cfg["hold_param"], cfg["hold_value"] = "hold", "length", 10.0
    else:
        cfg.setdefault("hold_param", None)
    return cfg


def _get_latents_and_targets(model, states, params_list, probe_targets, cfg, representation, roll, probe_target):
    train_z, _val_z, train_t, _val_t = stratified_probe_split(
        model, states, params_list, cfg["env"], probe_targets,
        regime=cfg["regime"], hold_param=cfg.get("hold_param"),
        representation=representation, roll=roll, probe_target= probe_target)
    return train_z, train_t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--representation", default="latent")
    ap.add_argument("--roll", type=int, default=200)
    ap.add_argument("--n_sub", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--probe_target", type = str, default = "h")

    args = ap.parse_args()

    model_files = glob.glob(str(Path(args.models_dir) / "**" / "*.pt"), recursive=True)
    rng = np.random.default_rng(0)

    for mf in model_files:
        cfg = parse_model(Path(mf))
        if cfg.get("regime") is None:
            continue
        cfg = _normalize_regime(cfg)

        print(f"\n[{Path(mf).name}]")
        model    = load_model(mf, cfg, device=args.device)
        rand_mod = _random_model(cfg, args.device)

        states, params_list, probe_targets = get_data(cfg)

        z_tr, t_tr = _get_latents_and_targets(model,    states, params_list, probe_targets,
                                              cfg, args.representation, args.roll, args.probe_target)
        z_rd, _t_rd = _get_latents_and_targets(rand_mod, states, params_list, probe_targets,
                                               cfg, args.representation, args.roll, args.probe_target)

        n = z_tr.shape[0]
        if n > args.n_sub:
            idx = rng.choice(n, args.n_sub, replace=False)
        else:
            idx = np.arange(n)
        z_tr_s, z_rd_s = z_tr[idx], z_rd[idx]

        want = [t for t in (PARAM_TARGETS + CONTROL_TARGETS) if t in t_tr]
        targets = {name: np.asarray(t_tr[name])[idx] for name in want}
        if not any(t in targets for t in CONTROL_TARGETS):
            print("  [warn] no observable positive control present — parameter null "
                  "will be uninterpretable; check probe_targets.")

        results = mi_dissociation(z_tr_s, z_rd_s, targets,
                                  seeds=tuple(args.seeds),
                                  steps=args.steps, device=args.device)

        env_tag = ENV_NAMES[cfg["env"]]
        policy_tag = "sparse" if cfg["impulse"] else "noise"
        group = f"{env_tag}_{policy_tag}_{TAGS[cfg['model_name']]}"
        out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"smile_{group}.csv"
        hdr = not csv_path.exists()
        with open(csv_path, "a", newline="") as fh:
            w = csv.writer(fh)
            if hdr:
                w.writerow(["checkpoint", "latent", "k", "regime", "target",
                            "is_control",
                            "trained_mi", "trained_sd", "random_mi", "random_sd",
                            "gain", "gain_sd"])
            for name, r in results.items():
                is_ctrl = name in CONTROL_TARGETS
                w.writerow([Path(mf).name, cfg["latent"], cfg["k"], cfg["regime"],
                            name, int(is_ctrl),
                            round(r["trained_mi"], 5), round(r["trained_sd"], 5),
                            round(r["random_mi"], 5), round(r["random_sd"], 5),
                            round(r["gain"], 5), round(r["gain_sd"], 5)])

        hist_path = out_dir / f"smile_hist_{group}_{Path(mf).stem}.npz"
        hist_dump = {}
        for name in targets:            
            h_tr = estimate_mi_smile(z_tr_s, targets[name],
                                     steps=args.steps, seed=args.seeds[0], device=args.device)["history"]
            hist_dump[f"{name}_trained"] = h_tr
        np.savez(hist_path, **hist_dump)
        print(f"  wrote {csv_path.name} and {hist_path.name}")

    print("Done.")