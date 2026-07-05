from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import torch
import numpy as np
import os
import csv
import argparse
from pathlib import Path
from models import make_model
from analysis.common import parse_model, load_model, iter_model_groups, get_data, HIDDEN_DIM
from analysis.common import probeable_vars

def generate_latents(model, states: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        N, T, state_dim = states.shape
        flat = states.reshape(-1, state_dim)
        z = model.encode_computational(flat)
        return z.reshape(N, T, -1)

def run_probe(train_z: np.ndarray, val_z: np.ndarray,
              train_target: np.ndarray, val_target: np.ndarray,
              label: str, writer, meta: dict, alpha: float) -> tuple[float, float, float]:

    if not np.isfinite(train_z).all() or not np.isfinite(val_z).all():
        print(f"{label}: SKIPPED (NaN/inf in latents)")
        return float('nan'), float('nan'), float('nan')

    if np.abs(train_z).max() > 1e4:
        print(f"{label}: SKIPPED (latent explosion > 1e4)")
        return float('nan'), float('nan'), float('nan')
    
    probe = Ridge(alpha=alpha)
    probe.fit(train_z, train_target)
    r2 = r2_score(val_target, probe.predict(val_z))

    mi = float(mutual_info_regression(train_z, train_target, n_neighbors=3).mean())

    shuffled = train_target.copy()
    np.random.shuffle(shuffled)
    probe_shuffled = Ridge(alpha=alpha)
    probe_shuffled.fit(train_z, shuffled)
    r2_shuffled = r2_score(val_target, probe_shuffled.predict(val_z))
    delta = r2 - r2_shuffled

    print(f"{label}: R2={r2:.4f} | shuffled={r2_shuffled:.4f} | delta={delta:.4f} | MI={mi:.4f}")
    
    writer.writerow([
        meta["checkpoint"], meta["config"], meta["latent"], meta["k"],
        label, round(r2, 4), round(r2_shuffled, 4), round(delta, 4), round(mi, 4)
    ])

    return r2, r2_shuffled, mi


def stratified_probe_split(model, all_states, gravities, lengths,
                           is_cartpole, train_frac=0.8, regime="combined"):
    train_s, val_s, train_g, val_g, train_l, val_l = [], [], [], [], [], []

    for states_c, g_c, l_c in zip(all_states, gravities, lengths):

        n = states_c.shape[0]
        ti = int(train_frac * n)
        N, T = states_c.shape[0], states_c.shape[1]

        g_arr = np.full((N, T), float(g_c), dtype=np.float32)
        l_arr = np.full((N, T), float(l_c), dtype=np.float32)

        train_s.append(states_c[:ti]);  val_s.append(states_c[ti:])
        train_g.append(g_arr[:ti]);      val_g.append(g_arr[ti:])
        train_l.append(l_arr[:ti]);      val_l.append(l_arr[ti:])

    train_states = torch.from_numpy(np.concatenate(train_s, 0)).float()
    val_states   = torch.from_numpy(np.concatenate(val_s, 0)).float()
    train_gt = torch.from_numpy(np.concatenate(train_g, 0)).float()
    val_gt   = torch.from_numpy(np.concatenate(val_g, 0)).float()
    train_lt = torch.from_numpy(np.concatenate(train_l, 0)).float()
    val_lt   = torch.from_numpy(np.concatenate(val_l, 0)).float()

    train_z = generate_latents(model, train_states)
    val_z   = generate_latents(model, val_states)
    latent_dim = train_z.shape[-1]
    train_z_flat = train_z.reshape(-1, latent_dim).numpy()
    val_z_flat   = val_z.reshape(-1, latent_dim).numpy()

    def build_targets(states, g_t, l_t, regime):
        vars_here = probeable_vars(regime, is_cartpole)
        t = {}
        if "theta" in vars_here:      t["theta"] = torch.atan2(states[:, :, 1], states[:, :, 0]).reshape(-1).numpy()
        if "theta_dot" in vars_here:  t["theta_dot"] = states[:, :, 2].reshape(-1).numpy()
        if "x" in vars_here:          t["x"] = states[:, :, 3].reshape(-1).numpy()
        if "x_dot" in vars_here:      t["x_dot"] = states[:, :, 4].reshape(-1).numpy()
        if "gravity" in vars_here:    t["gravity"] = g_t.reshape(-1).numpy()
        if "length" in vars_here:     t["length"] = l_t.reshape(-1).numpy()
        if "g_over_l" in vars_here:   t["g_over_l"] = (g_t / l_t).reshape(-1).numpy()
        if "sqrt_l_over_g" in vars_here: t["sqrt_l_over_g"] = torch.sqrt(l_t / g_t).reshape(-1).numpy()
        return t

    return (train_z_flat, val_z_flat,
            build_targets(train_states, train_gt, train_lt, regime),
            build_targets(val_states, val_gt, val_lt, regime))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    for (env_tag, policy_tag, model_tag), files in iter_model_groups(args.models_dir).items():
        tag = f"{env_tag}_{policy_tag}_{model_tag}"
        output_csv = f"probe_results/linear_probe_{tag}.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        write_header = not os.path.exists(output_csv)
        csv_file = open(output_csv, "a", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["checkpoint", "config", "latent_dim", "k", "target",
                             "r2", "r2_shuffled", "delta", "mi_sum"])

        for mf in files:
            cfg = parse_model(Path(mf))
            is_cp = cfg["env"] == "CartPoleSim"
            print(f"\n[{Path(mf).name}]")
            model = load_model(mf, cfg, args.device)

            random_model = None
            if args.random_init:
                random_model = make_model(cfg["model_name"], state_dim=5 if is_cp else 3,
                    action_dim=1, hidden_dim=HIDDEN_DIM.get(cfg["model_name"], 64),
                    latent_dim=cfg["latent"])

            states, gravities, lengths = get_data(cfg)

            meta = {"checkpoint": Path(mf).name, "config": cfg["config"],
                    "latent": cfg["latent"], "k": cfg["k"], "policy": tag}

            train_z, val_z, train_t, val_t = stratified_probe_split(
                model, states, gravities, lengths, is_cp, regime=cfg["regime"])
            print("=== Trained ===")
            for tn in train_t:
                run_probe(train_z, val_z, train_t[tn], val_t[tn], tn, writer, meta, args.alpha)

            if random_model is not None:
                rz_tr, rz_val, rt_tr, rt_val = stratified_probe_split(
                    random_model, states, gravities, lengths, is_cp, regime=cfg["regime"])
                print("=== Random baseline ===")
                for tn in rt_tr:
                    run_probe(rz_tr, rz_val, rt_tr[tn], rt_val[tn], f"{tn}_random", writer, meta, args.alpha)

            csv_file.flush()
        csv_file.close()
        print(f"[{tag}] done -> {output_csv}")