import argparse, csv
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from models.simplenn import SimpleNN
from models.transfer import ProtocolAModel, ProtocolBModel
from analysis.common import parse_model, iter_model_groups, load_model, collect_for_config 


def make_regime_labels(actions, threshold=1e-3):
    if actions.ndim == 3:
        actions = actions.squeeze(-1)
    return (np.abs(actions) > threshold).astype(np.float32).reshape(-1)


def generate_latents_flat(model, states):
    model.eval()
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states).float()
    N, T, sd = states.shape
    states = states[:, :-1, :]
    with torch.inference_mode():
        z = model.encode_computational(states.reshape(-1, sd))
    return z.numpy()


ALL_CONFIGS = [(5.0,2.0),(5.0,10.0),(5.0,18.0),(9.8,2.0),(9.8,10.0),
               (9.8,18.0),(15.0,2.0),(15.0,10.0),(15.0,18.0)]

def build_protocol_model(config):
    latent_angular, latent_B, hidden = config["latent"], config["latent_B"], 64
    if config["protocol"] == "B":
        return ProtocolBModel(
            angular_encoder=SimpleNN(3, hidden, latent_angular),
            cartpole_encoder=SimpleNN(2, hidden, latent_B),
            latent_angular=latent_angular, latent_B=latent_B,
            action_dim=1, hidden_dim=hidden)
    else:
        return ProtocolAModel(
            unified_encoder=SimpleNN(5, hidden, latent_angular),
            latent_angular=latent_angular, action_dim=1, hidden_dim=hidden)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out_dir", required = True)
    ap.add_argument("--num_trajectories", type=int, default=50)
    ap.add_argument("--episode_time", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    for (env_tag, policy_tag, model_tag), files in iter_model_groups(args.models_dir).items():
        csv_path = Path(f"{args.out_dir}/regime_probe_{env_tag}_{policy_tag}_{model_tag}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        csv_file = open(csv_path, "a", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow([
                "checkpoint", "model_config", "eval_config", "latent_dim", "k",
                "env", "model_name", "beta", "accuracy", "auc", "frac_impulse", "frac_gap",
            ])

        for mf in files:
            cfg = parse_model(Path(mf))
            print(f"\n[{Path(mf).name}]")

            if cfg["protocol"] is not None:
                model = build_protocol_model(cfg)
                model.load_state_dict(torch.load(mf, map_location=args.device))
                model.eval()
            else:
                model = load_model(mf, cfg, args.device)

            eval_configs = ALL_CONFIGS if cfg["flag"] else [(cfg["g"], cfg["l"])]

            for g_eval, l_eval in eval_configs:
                states_t, actions_t = collect_for_config(
                    g_eval, l_eval, cfg["env"], impulse=True,
                    n_traj=args.num_trajectories, steps=args.episode_time)
                states, actions = states_t.numpy(), actions_t.numpy()

                labels = make_regime_labels(actions, args.threshold)
                z_flat = generate_latents_flat(model, states)
                if not np.isfinite(z_flat).all() or np.abs(z_flat).max() > 1e4:
                    print("  SKIPPED (latent explosion)")
                    continue

                frac_impulse = float(labels.mean())
                n = z_flat.shape[0]
                perm = np.random.permutation(n)
                z_flat, labels = z_flat[perm], labels[perm]
                ti = int(0.8 * n)
                train_z, val_z = z_flat[:ti], z_flat[ti:]
                train_l, val_l = labels[:ti], labels[ti:]

                probe = LogisticRegression(max_iter=1000, C=1.0)
                probe.fit(train_z, train_l)
                acc = round(float(accuracy_score(val_l, probe.predict(val_z))), 4)
                auc = round(float(roc_auc_score(val_l, probe.predict_proba(val_z)[:, 1])), 4)
                print(f"  g={g_eval} l={l_eval} | acc={acc:.4f} auc={auc:.4f} | impulse_frac={frac_impulse:.3f}")

                writer.writerow([
                    Path(mf).name, cfg["config"], f"g{g_eval}_l{l_eval}",
                    cfg["latent"], cfg["k"], cfg["env"], cfg["model_name"], cfg["beta"],
                    acc, auc, round(frac_impulse, 4), round(1.0 - frac_impulse, 4),
                ])
                csv_file.flush()
        csv_file.close()
        print(f"[{env_tag}_{policy_tag}_{model_tag}] done -> {csv_path}")