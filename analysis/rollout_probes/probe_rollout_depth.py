from pathlib import Path
import glob
import csv
import argparse
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from analysis.common import parse_model, iter_model_groups, load_model, collect_for_config


def probe_at_depth(z: np.ndarray, target: np.ndarray,
                   alpha: float = 10.0) -> tuple[float, float]:

    if not np.isfinite(z).all() or not np.isfinite(target).all():
        return float('nan'), float('nan')

    z = np.clip(z, -1e6, 1e6)
    n = z.shape[0]
    train_idx = int(0.8 * n)
    
    train_z, val_z = z[:train_idx], z[train_idx:]
    train_t, val_t = target[:train_idx], target[train_idx:]
    
    if val_t.std() < 1e-6:
        return 0.0, 0.0
    
    probe = Ridge(alpha=alpha)
    probe.fit(train_z, train_t)
    r2 = r2_score(val_t, probe.predict(val_z))
    
    shuffled = train_t.copy()
    np.random.shuffle(shuffled)
    probe_s = Ridge(alpha=alpha)
    probe_s.fit(train_z, shuffled)
    r2_s = r2_score(val_t, probe_s.predict(val_z))
    
    return round(float(r2), 6), round(float(r2_s), 6)


ALL_CONFIGS = [
    (5.0,  2.0), (5.0,  10.0), (5.0,  18.0),
    (9.8,  2.0), (9.8,  10.0), (9.8,  18.0),
    (15.0, 2.0), (15.0, 10.0), (15.0, 18.0),
]

PROBE_DEPTHS = [0, 1, 3, 5, 10, 15, 25, 50]

def eval_cfg(cfg):
    if cfg["regime"] == "combined":
        return ALL_CONFIGS
    elif cfg["regime"] == "holdg":
        return [(9.8, l) for l in [2.0, 10.0, 18.0]]
    elif cfg["regime"] == "holdl":
        return [(g, 10.0) for g in [5.0, 9.8, 15.0]]
    else:
        return [(cfg["g"], cfg["l"])]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    for (env_tag, policy_tag, model_tag), files in iter_model_groups(args.models_dir).items():
        csv_path = Path(f"{args.out_dir}/probe_rollout_depth_{env_tag}_{policy_tag}_{model_tag}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        csv_file = open(csv_path, "a", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow([
                "checkpoint", "model_config", "eval_config", "latent_dim", "k", "depth",
                "r2_cos_theta", "r2s_cos_theta", "delta_cos_theta", "r2_sin_theta", "r2s_sin_theta", "delta_sin_theta",
                "r2_thetadot", "r2s_thetadot", "delta_thetadot",
                "r2_x", "r2s_x", "delta_x",
                "r2_xdot", "r2s_xdot", "delta_xdot",
            ])

        impulse = (policy_tag == "sparse")
        for mf in files:
            cfg = parse_model(Path(mf))
            print(f"\n[{Path(mf).name}]")
            model = load_model(mf, cfg, args.device)
            state_dim = 5 if cfg["env"] == "CartPoleSim" else 3
            eval_configs = eval_cfg(cfg)

            for g_eval, l_eval in eval_configs:
                states_t, actions_t = collect_for_config(g_eval, l_eval, cfg["env"], impulse)
                with torch.inference_mode():
                    comp_current = model.encode_computational(states_t[:, 0, :])

                for depth in PROBE_DEPTHS:
                    if depth > 0:
                        with torch.inference_mode():
                            prev = PROBE_DEPTHS[PROBE_DEPTHS.index(depth) - 1]
                            for step in range(depth - prev):
                                t_idx = prev + step
                                if t_idx >= actions_t.shape[1]:
                                    break
                                a = actions_t[:, t_idx].unsqueeze(-1)
                                comp_current = model.step_computational(comp_current, a)

                    if depth >= states_t.shape[1]:
                        break
                    with torch.inference_mode():
                        z_np = model.probe_representation(comp_current).numpy()
                    s_true = states_t[:, depth, :]

                    cos_theta_true = s_true[:, 0].numpy()
                    sin_theta_true = s_true[:, 1].numpy()
                    thetadot_true = s_true[:, 2].numpy()
                    r2_cth, r2s_cth = probe_at_depth(z_np, cos_theta_true, args.alpha)
                    r2_sth, r2s_sth = probe_at_depth(z_np, sin_theta_true, args.alpha)
                    r2_td, r2s_td = probe_at_depth(z_np, thetadot_true, args.alpha)
                    delta_cth = round(r2_cth - r2s_cth, 6) if not (np.isnan(r2_cth) or np.isnan(r2s_cth)) else float('nan')
                    delta_sth = round(r2_sth - r2s_sth, 6) if not (np.isnan(r2_sth) or np.isnan(r2s_sth)) else float('nan')
                    delta_td = round(r2_td - r2s_td, 6) if not (np.isnan(r2_td) or np.isnan(r2s_td)) else float('nan')

                    if state_dim == 5:
                        x_true, xdot_true = s_true[:, 3].numpy(), s_true[:, 4].numpy()
                        r2_x, r2s_x = probe_at_depth(z_np, x_true, args.alpha)
                        r2_xd, r2s_xd = probe_at_depth(z_np, xdot_true, args.alpha)
                        delta_x = round(r2_x - r2s_x, 6) if not (np.isnan(r2_x) or np.isnan(r2s_x)) else float('nan')
                        delta_xd = round(r2_xd - r2s_xd, 6) if not (np.isnan(r2_xd) or np.isnan(r2s_xd)) else float('nan')
                    else:
                        r2_x = r2s_x = delta_x = r2_xd = r2s_xd = delta_xd = float('nan')

                    writer.writerow([
                        Path(mf).name, cfg["config"], f"g{g_eval}_l{l_eval}",
                        cfg["latent"], cfg["k"], depth,
                        r2_cth, r2s_cth, delta_cth, r2_sth, r2s_sth, delta_sth, r2_td, r2s_td, delta_td,
                        r2_x, r2s_x, delta_x, r2_xd, r2s_xd, delta_xd,
                    ])
                    csv_file.flush()
        csv_file.close()
        print(f"[{env_tag}_{policy_tag}_{model_tag}] done -> {csv_path}")