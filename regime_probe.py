import argparse
import numpy as np
import torch
import csv
import os
import glob
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from models import make_model
from models.simplenn import SimpleNN
from models.transfer import ProtocolAModel, ProtocolBModel
from collector.collect import collect_trajectories
from sim_envs.envs import make_env
from utils import parse_model


def collect_sparse_trajectories(g: float, l: float, env: str,
                                 num_trajectories: int, episode_time: int,
                                 policy_seed: int) -> tuple:
    if env == "CartPoleSim":
        environment = make_env("CartPoleSim",
            gravity=g, mass1=0.1, mass2=1.0,
            length=l, dt=0.02, max_action=10.0,
            damping=0.0, seed=42
        )
    else:
        environment = make_env("PendulumSim",
            gravity=g, mass1=1.0, mass2=0.0,
            length=l, dt=0.01, max_action=10.0,
            damping=0.0, seed=42
        )
    states, actions, _ = collect_trajectories(
        environment, num_trajectories, episode_time,
        policy_seed, False, True  
    )
    return states, actions


def make_regime_labels(actions: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    labels = (np.abs(actions) > threshold).astype(np.float32)
    return labels.reshape(-1)


def generate_latents_flat(model, states: np.ndarray) -> np.ndarray:
    model.eval()
    states_t = torch.from_numpy(states).float()
    N, T, state_dim = states_t.shape
    with torch.no_grad():
        flat = states_t.reshape(-1, state_dim)
        z = model.encode(flat)
    return z.numpy()


ALL_CONFIGS = [
    (5.0,  2.0),  (5.0,  10.0), (5.0,  18.0),
    (9.8,  2.0),  (9.8,  10.0), (9.8,  18.0),
    (15.0, 2.0),  (15.0, 10.0), (15.0, 18.0),
]


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--models_dir", type=Path, default=None)
    args.add_argument("--model_pt", type=Path, default=None)
    args.add_argument("--num_trajectories", type=int, default=50)
    args.add_argument("--episode_time", type=int, default=500)
    args.add_argument("--policy_seed", type=int, default=35)
    args.add_argument("--threshold", type=float, default=1e-3,
                      help="Action magnitude threshold for impulse label")
    parser = args.parse_args()

    pt_files: list[Path] = []
    if parser.models_dir:
        pt_files = [Path(p) for p in
                    glob.glob(str(parser.models_dir / "*.pt"))]
    if parser.model_pt:
        pt_files.append(parser.model_pt)

    csv_path = Path("probe_results/regime_probe.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "checkpoint", "model_config", "eval_config",
            "latent_dim", "k", "env",
            "accuracy", "auc",
            "frac_impulse", "frac_gap",
        ])

    for file in pt_files:
        config = parse_model(file)
        print(f"\n[{file.name}]")

        
        if config["protocol"] is not None:
            latent_angular = config["latent"]
            latent_B = config["latent_B"]
            hidden_dim = 64
            action_dim = 1
            if config["protocol"] == "B":
                angular_encoder = SimpleNN(3, hidden_dim, latent_angular)
                cartpole_encoder = SimpleNN(2, hidden_dim, latent_B)
                model = ProtocolBModel(
                    angular_encoder=angular_encoder,
                    cartpole_encoder=cartpole_encoder,
                    latent_angular=latent_angular,
                    latent_B=latent_B,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim
                )
            elif config["protocol"] == "A":
                unified_encoder = SimpleNN(5, hidden_dim, latent_angular)
                model = ProtocolAModel(
                    unified_encoder=unified_encoder,
                    latent_angular=latent_angular,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim
                )
        else:
            state_dim = 5 if config["env"] == "CartPoleSim" else 3
            model = make_model("WorldModel",
                state_dim=state_dim, action_dim=1,
                hidden_dim=64, latent_dim=config["latent"]
            )
        model.load_state_dict(torch.load(file))
        model.eval()

       
        eval_configs = ALL_CONFIGS if config["flag"] else [(config["g"], config["l"])]

        for g_eval, l_eval in eval_configs:
            states, actions = collect_sparse_trajectories(
                g_eval, l_eval, config["env"],
                parser.num_trajectories, parser.episode_time,
                parser.policy_seed
            )

            labels = make_regime_labels(actions, parser.threshold)
            z_flat = generate_latents_flat(model, states)

            
            frac_impulse = float(labels.mean())
            frac_gap = 1.0 - frac_impulse

            
            n = z_flat.shape[0]
            train_idx = int(0.8 * n)
            perm = np.random.permutation(n)
            z_flat = z_flat[perm]
            labels = labels[perm]

            train_z, val_z = z_flat[:train_idx], z_flat[train_idx:]
            train_l, val_l = labels[:train_idx], labels[train_idx:]

            
            probe = LogisticRegression(max_iter=1000, C=1.0)
            probe.fit(train_z, train_l)
            val_pred = probe.predict(val_z)
            val_prob = probe.predict_proba(val_z)[:, 1]

            accuracy = round(float(accuracy_score(val_l, val_pred)), 4)
            auc = round(float(roc_auc_score(val_l, val_prob)), 4)

            print(f"  g={g_eval} l={l_eval} | "
                  f"acc={accuracy:.4f} auc={auc:.4f} | "
                  f"impulse_frac={frac_impulse:.3f}")

            writer.writerow([
                file.name, config["config"], f"g{g_eval}_l{l_eval}",
                config["latent"], config["k"], config["env"],
                accuracy, auc,
                round(frac_impulse, 4), round(frac_gap, 4),
            ])
            csv_file.flush()

    csv_file.close()
    print(f"\nDone. Saved to {csv_path}")