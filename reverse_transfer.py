import argparse
import yaml
import numpy as np
import torch
import os
import csv
from pathlib import Path
from collector.collect import collect_trajectories
from sim_envs.envs import make_env
from models import make_model
from models.simplenn import SimpleNN
from models.transfer import ProtocolBModel, ProtocolAModel
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression


def load_protocol_model(yaml_out: dict, device: str):
    protocol = yaml_out["protocol"]
    transfer_config = yaml_out["transfer"]
    latent_angular = transfer_config["latent_angular"]
    latent_B = transfer_config["latent_B"]
    hidden_dim = transfer_config["hidden_dim"]
    action_dim = transfer_config["action_dim"]

    pendulum_model_config = yaml_out["pendulum_model"]
    pendulum_model_params = {k: v for k, v in pendulum_model_config.items()
                             if k not in ("name", "model_path")}
    pendulum_model = make_model(pendulum_model_config["name"], **pendulum_model_params)
    pendulum_model.load_state_dict(torch.load(pendulum_model_config["model_path"]))
    pendulum_model.eval()
    pendulum_model.requires_grad_(False)

    if protocol == "B":
        angular_encoder = pendulum_model.encoder
        angular_encoder.requires_grad_(False)
        cartpole_encoder = SimpleNN(2, hidden_dim, latent_B)
        model = ProtocolBModel(
            angular_encoder=angular_encoder,
            cartpole_encoder=cartpole_encoder,
            latent_angular=latent_angular,
            latent_B=latent_B,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
    elif protocol == "A":
        unified_encoder = SimpleNN(5, hidden_dim, latent_angular)
        model = ProtocolAModel(
            unified_encoder=unified_encoder,
            latent_angular=latent_angular,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    checkpoint_path = yaml_out["checkpointing"]["load_path"]
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model.requires_grad_(False)
    return model, pendulum_model, protocol


def pad_pendulum_states(states: np.ndarray, noise: bool = False, 
                         noise_scale: float = 1.0, seed: int = 42) -> np.ndarray:
    
    N, T, _ = states.shape
    if noise:
        rng = np.random.default_rng(seed)
        pad = rng.normal(0, noise_scale, size=(N, T, 2)).astype(np.float32)
    else:
        pad = np.zeros((N, T, 2), dtype=np.float32)
    return np.concatenate([states, pad], axis=-1)


def run_probe(train_z, val_z, train_target, val_target, alpha=10.0):
    probe = Ridge(alpha=alpha)
    probe.fit(train_z, train_target)
    r2 = r2_score(val_target, probe.predict(val_z))
    mi = float(mutual_info_regression(train_z, train_target, n_neighbors=3).sum())
    return r2, mi


def evaluate_predictions(model, states_padded: torch.Tensor, actions: torch.Tensor,
                          horizon: int = 50) -> dict:

    model.eval()
    with torch.no_grad():
        N, T, _ = states_padded.shape
        all_cos_err, all_sin_err, all_thetadot_err = [], [], []
        for i in range(N):
            s = states_padded[i, :T-1]       
            a = actions[i, :T-1].unsqueeze(-1) 
            z = model.encode(s)
            z_next = model.step(z, a)
            s_hat = model.decode(z_next)       
            true_next = states_padded[i, 1:T, :3]  
            pred_next = s_hat[:, :3]
            err = (pred_next - true_next).pow(2).mean(dim=0).sqrt()
            all_cos_err.append(err[0].item())
            all_sin_err.append(err[1].item())
            all_thetadot_err.append(err[2].item())
    return {
        "rmse_cos": float(np.mean(all_cos_err)),
        "rmse_sin": float(np.mean(all_sin_err)),
        "rmse_thetadot": float(np.mean(all_thetadot_err)),
    }


def evaluate_probe_retention(model, states_padded: torch.Tensor, 
                              pendulum_model, states_raw: torch.Tensor,
                              protocol: str, alpha: float = 10.0) -> dict:
    """Compare probe R² on angular latent before and after transfer."""
    model.eval()
    pendulum_model.eval()

    with torch.no_grad():
        N, T, _ = states_padded.shape
        flat_padded = states_padded.reshape(-1, 5)
        z_transfer = model.encode(flat_padded)

        flat_raw = states_raw.reshape(-1, 3)
        z_pendulum = pendulum_model.encode(flat_raw)

    theta = torch.atan2(flat_raw[:, 1], flat_raw[:, 0]).numpy()
    theta_dot = flat_raw[:, 2].numpy()

    train_idx = int(0.8 * flat_padded.shape[0])

    results = {}
    for target_name, target in [("theta", theta), ("theta_dot", theta_dot)]:

        z_pend_np = z_pendulum.numpy()
        r2_pend, mi_pend = run_probe(
            z_pend_np[:train_idx], z_pend_np[train_idx:],
            target[:train_idx], target[train_idx:], alpha
        )


        z_trans_np = z_transfer.numpy()
        r2_trans_full, mi_trans_full = run_probe(
            z_trans_np[:train_idx], z_trans_np[train_idx:],
            target[:train_idx], target[train_idx:], alpha
        )

        if protocol == "B":
            latent_angular = z_pendulum.shape[-1]
            z_ang_only = z_trans_np[:, :latent_angular]
            r2_trans_ang, mi_trans_ang = run_probe(
                z_ang_only[:train_idx], z_ang_only[train_idx:],
                target[:train_idx], target[train_idx:], alpha
            )
        else:
            r2_trans_ang, mi_trans_ang = float("nan"), float("nan")

        results[target_name] = {
            "r2_pendulum_baseline": round(r2_pend, 4),
            "mi_pendulum_baseline": round(mi_pend, 4),
            "r2_transfer_full": round(r2_trans_full, 4),
            "mi_transfer_full": round(mi_trans_full, 4),
            "r2_transfer_angular_only": round(r2_trans_ang, 4),
            "mi_transfer_angular_only": round(mi_trans_ang, 4),
        }

    return results


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--yaml", type=str, required=True)
    args.add_argument("--noise", action="store_true", 
                      help="Append random noise instead of zeros for translational dims")
    args.add_argument("--noise_scale", type=float, default=1.0)
    args.add_argument("--num_trajectories", type=int, default=50)
    args.add_argument("--horizon", type=int, default=50)
    parser = args.parse_args()

    with open(parser.yaml, 'r') as f:
        yaml_out = yaml.safe_load(f)

    device = yaml_out["settings"]["device"]
    protocol = yaml_out["protocol"]

    model, pendulum_model, protocol = load_protocol_model(yaml_out, device)

    pend_env_config = yaml_out["pendulum_eval_env"]
    pend_env = make_env("PendulumSim", **{k: v for k, v in pend_env_config.items() if k != "name"})
    states_raw, actions, _ = collect_trajectories(
        pend_env,
        parser.num_trajectories,
        yaml_out["collector"]["episode_time"],
        yaml_out["collector"]["policy_seed"] + 99,  
        False,
        False  
    )


    states_padded = pad_pendulum_states(states_raw, noise=parser.noise,
                                        noise_scale=parser.noise_scale)
    states_padded_t = torch.from_numpy(states_padded).float()
    states_raw_t = torch.from_numpy(states_raw).float()
    actions_t = torch.from_numpy(actions).float()


    print(f"\n=== Prediction Accuracy (noise={parser.noise}) ===")
    pred_results = evaluate_predictions(model, states_padded_t, actions_t, parser.horizon)
    for k, v in pred_results.items():
        print(f"  {k}: {v:.4f}")


    print(f"\n=== Probe Retention (noise={parser.noise}) ===")
    probe_results = evaluate_probe_retention(
        model, states_padded_t, pendulum_model, states_raw_t, protocol
    )
    for target_name, metrics in probe_results.items():
        print(f"\n  {target_name}:")
        for metric_name, val in metrics.items():
            print(f"    {metric_name}: {val}")


    condition = "noise" if parser.noise else "zeros"
    out_dir = "probe_results/reverse_transfer"
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_name = Path(yaml_out["checkpointing"]["load_path"]).stem
    out_path = os.path.join(out_dir, f"{checkpoint_name}_protocol{protocol}_{condition}.csv")

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "protocol", "condition", "target",
                "r2_pendulum_baseline", "mi_pendulum_baseline",
                "r2_transfer_full", "mi_transfer_full",
                "r2_transfer_angular_only", "mi_transfer_angular_only",
                "rmse_cos", "rmse_sin", "rmse_thetadot"
            ])
        for target_name, metrics in probe_results.items():
            writer.writerow([
                protocol, condition, target_name,
                metrics["r2_pendulum_baseline"],
                metrics["mi_pendulum_baseline"],
                metrics["r2_transfer_full"],
                metrics["mi_transfer_full"],
                metrics["r2_transfer_angular_only"],
                metrics["mi_transfer_angular_only"],
                pred_results["rmse_cos"],
                pred_results["rmse_sin"],
                pred_results["rmse_thetadot"],
            ])

    print(f"\n[SAVED] {out_path}")