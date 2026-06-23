from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import torch
import numpy as np
import os
import glob
import csv
import yaml
import argparse
from pathlib import Path
from models import make_model
from collector.collect import collect_trajectories
from models.transfer import ProtocolAModel, ProtocolBModel
from models.wmodel import WorldModel
from sim_envs.envs import make_env
from utils import parse_model
from models.simplenn import SimpleNN

def generate_latents(model, states: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        N, T, state_dim = states.shape
        flat = states.reshape(-1, state_dim)
        z = model.encode(flat)
        return z.reshape(N, T, -1)


def train_val_split(states: torch.Tensor, z_states: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trajs, _, state_dims = states.shape
    train_idx = int(0.8 * trajs)
    train_states = states[:train_idx].reshape(-1, state_dims)
    val_states = states[train_idx:].reshape(-1, state_dims)
    if z_states is None:
        train_zstates, val_zstates = torch.zeros(1), torch.zeros(1)
    else:
        _, _, latent_dim = z_states.shape
        train_zstates = z_states[:train_idx].reshape(-1, latent_dim)
        val_zstates = z_states[train_idx:].reshape(-1, latent_dim)
    return train_states, val_states, train_zstates, val_zstates


def run_probe(train_z: np.ndarray, val_z: np.ndarray,
              train_target: np.ndarray, val_target: np.ndarray,
              label: str, writer, meta: dict, alpha: float, probe_path: str) -> tuple[float, float, float]:

    if not np.isfinite(train_z).all() or not np.isfinite(val_z).all():
        print(f"{label}: SKIPPED (NaN/inf in latents)")
        return float('nan'), float('nan'), float('nan')

    if np.abs(train_z).max() > 1e4:
        print(f"{label}: SKIPPED (latent explosion > 1e4)")
        return float('nan'), float('nan'), float('nan')
    
    probe = Ridge(alpha=alpha)
    probe.fit(train_z, train_target)
    if not label.endswith("_random"):
        coef_path = f"{probe_path}/{meta['checkpoint']}_{label}_{meta['policy']}.npy"
        os.makedirs(os.path.dirname(coef_path), exist_ok=True)
        np.save(coef_path, probe.coef_)
    r2 = r2_score(val_target, probe.predict(val_z))

    mi = float(mutual_info_regression(train_z, train_target, n_neighbors=3).sum())

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



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--yaml", type=str, required=True)
    parser = args.parse_args()

    with open(parser.yaml, 'r') as f:
        yaml_out = yaml.safe_load(f)

    impulse_policy = yaml_out["collector"]["impulse_policy"]

    output_csv = yaml_out["probe"]["output_csv"]
    
    tag = "noise"
    if impulse_policy:
        tag = "sparse"
    output_csv = output_csv[:-4]+f"_{tag}.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not os.path.exists(output_csv)
    csv_file = open(output_csv, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["checkpoint", "config", "latent_dim", "k", "target", "r2", "r2_shuffled", "delta", "mi_sum"])

    
    
    env_name = yaml_out["environment"]["name"]
    is_cartpole = (env_name == "CartPoleSim")
    
    if is_cartpole:
        probe_path = f"probe_results/coefs_cartpole_{tag}"
        
    else:
        probe_path = f"probe_results/coefs_pendulum_{tag}"
    os.makedirs(probe_path, exist_ok=True)
    model_paths = glob.glob(yaml_out["checkpointing"]["load_paths"])
    all_states, gravities, lengths = [], [], []
    
    model_config = yaml_out["model"]
    model_params = {k: v for k, v in model_config.items()
            if k not in ("name")}
    for path in model_paths:
        current_config = parse_model(Path(path))
        if current_config["flag"]:
            all_states, gravities, lengths = [], [], []
            dataset_glob = yaml_out["datasets"]["paths"][0]
            for data_path in glob.glob(dataset_glob):
                data = np.load(data_path)
                all_states.append(data["states"][:yaml_out["collector"]["num_trajectories"]])
                gravities.append(np.full((all_states[-1].shape[0], all_states[-1].shape[1]), float(data["gravity"])))
                lengths.append(np.full((all_states[-1].shape[0], all_states[-1].shape[1]), float(data["length"])))
            probe_states = torch.from_numpy(np.concatenate(all_states, axis=0)).float()
            gravity_tensor = torch.from_numpy(np.concatenate(gravities, axis=0)).float()
            length_tensor = torch.from_numpy(np.concatenate(lengths, axis=0)).float()
            if is_cartpole:
                targets = {
                    "theta": torch.atan2(probe_states[:,:,1], probe_states[:,:,0]),
                    "theta_dot": probe_states[:,:,2],
                    "x": probe_states[:,:,3],
                    "x_dot": probe_states[:,:,4],
                    "gravity": gravity_tensor,
                    "length": length_tensor,
                }
            else:
                targets = {
                    "theta": torch.atan2(probe_states[:,:,1], probe_states[:,:,0]),
                    "theta_dot": probe_states[:,:,2],
                    "gravity": gravity_tensor,
                    "length": length_tensor,
                }
            

        else:
            collector_config = yaml_out["collector"]
            env_config = yaml_out["environment"]
            env_config["gravity"] = current_config["g"]
            env_config["length"] = current_config["l"]
            env_params = {k: v for k, v in env_config.items() if k not in ("name",)}
            environment = make_env(env_config["name"], **env_params)
            raw_states, _, _ = collect_trajectories(environment, 
                                            collector_config["num_trajectories"],
                                            collector_config["episode_time"],
                                            collector_config["policy_seed"],
                                            collector_config["save"],
                                            collector_config["impulse_policy"])
            probe_states = torch.from_numpy(raw_states).float()
            if is_cartpole:
                targets = {
                    "theta": torch.atan2(probe_states[:,:,1], probe_states[:,:,0]),
                    "theta_dot": probe_states[:,:,2],
                    "x": probe_states[:,:,3],
                    "x_dot": probe_states[:,:,4],
                    "gravity": torch.full((probe_states.shape[0], probe_states.shape[1]), current_config["g"]),
                    "length": torch.full((probe_states.shape[0], probe_states.shape[1]), current_config["l"]),
                }
            else:
                targets = {
                    "theta": torch.atan2(probe_states[:,:,1], probe_states[:,:,0]),
                    "theta_dot": probe_states[:,:,2],
                    "gravity": torch.full((probe_states.shape[0], probe_states.shape[1]), current_config["g"]),
                    "length": torch.full((probe_states.shape[0], probe_states.shape[1]), current_config["l"]),
                }
        model_params["latent_dim"] = current_config["latent"]
        if current_config["protocol"] is not None:
            
            latent_angular = current_config["latent"]
            latent_B = current_config["latent_B"]
            hidden_dim = model_params.get("hidden_dim", 64)
            action_dim = model_params.get("action_dim", 1)
            
            if current_config["protocol"] == "B":
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
            elif current_config["protocol"] == "A":
                unified_encoder = SimpleNN(5, hidden_dim, latent_angular)
                model = ProtocolAModel(
                    unified_encoder=unified_encoder,
                    latent_angular=latent_angular,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim
                )
            model.load_state_dict(torch.load(path))
            random_model = None 
        else:
            model_params["latent_dim"] = current_config["latent"]
            model = make_model(model_config["name"], **model_params)
            model.load_state_dict(torch.load(path))
            random_model = make_model(model_config["name"], **model_params)

        train_z_rand_np = None
        val_z_rand_np = None
        if random_model is not None:
            z_random = generate_latents(random_model, probe_states)
            _, _, train_z_rand, val_z_rand = train_val_split(probe_states, z_random)
            train_z_rand_np = train_z_rand.numpy()
            val_z_rand_np = val_z_rand.numpy()
                

        z_states = generate_latents(model, probe_states)
        train_states, val_states, train_z, val_z = train_val_split(probe_states, z_states)
        train_z_np = train_z.numpy()
        val_z_np = val_z.numpy()
        

        

        meta = {
            "checkpoint": os.path.basename(path),
            "config": current_config["config"],
            "latent": current_config["latent"],
            "k": current_config["k"],
            "policy": tag
        }

        print(f"\n[{meta['checkpoint']}]")
        print("=== Trained Model ===")
        for target_name, target in targets.items():
            train_target, val_target, _, _ = train_val_split(target.unsqueeze(-1), None)
            train_target_np = train_target.numpy().ravel()
            val_target_np = val_target.numpy().ravel()
            run_probe(train_z_np, val_z_np, train_target_np, val_target_np, target_name, writer, meta, yaml_out["probe"]["alpha"], probe_path)

        if random_model is not None:
            print("=== Random Model Baseline ===")
            for target_name, target in targets.items():
                train_target, val_target, _, _ = train_val_split(target.unsqueeze(-1), None)
                train_target_np = train_target.numpy().ravel()
                val_target_np = val_target.numpy().ravel()
                run_probe(train_z_rand_np, val_z_rand_np, train_target_np, val_target_np,
                        f"{target_name}_random", writer, meta, yaml_out["probe"]["alpha"], probe_path)

        csv_file.flush()
    csv_file.close()
    print(f"\nDone. Results saved to {output_csv}")