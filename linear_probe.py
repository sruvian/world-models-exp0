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
from models import make_model
from collector.collect import collect_trajectories
from models.wmodel import WorldModel
from sim_envs.envs import make_env


def generate_latents(model: WorldModel, states: torch.Tensor) -> torch.Tensor:
    z_states = []
    model.eval()
    with torch.no_grad():
        for state in states:
            latent = model.encode(state)
            z_states.append(latent)
    return torch.stack(z_states)


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
              label: str, writer, meta: dict, alpha: float) -> tuple[float, float, float]:
    
    probe = Ridge(alpha=alpha)
    probe.fit(train_z, train_target)
    if not label.endswith("_random"):
        coef_path = f"probe_results/coefs/{meta['checkpoint']}_{label}.npy"
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

def parse_model(model_path: str)-> dict:
    flag = False
    g, l = 0.0, 0.0
    latent = 0
    k = 0

    model_name = os.path.basename(model_path)
    model_name = model_name.replace(".pt", "")
    
    model_components = model_name.split("_")[2:]
    if "combined" in model_components or "combinedstratified" in model_components:
        flag = True
        k = int(model_components[1][1:])
        
    else:
        g = float(model_components[0][1:])
        l = float(model_components[1][1:])
        k = int(model_components[2][1:])

    latent = int(model_components[-1][6:])
    config = "combined" if flag else f"g{g}_l{l}"
    return {"flag": flag, "g": g, "l": l, "latent": latent, "k": k, "config": config}


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--yaml", type=str, required=True)
    parser = args.parse_args()

    with open(parser.yaml, 'r') as f:
        yaml_out = yaml.safe_load(f)

    output_csv = yaml_out["probe"]["output_csv"]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not os.path.exists(output_csv)
    csv_file = open(output_csv, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["checkpoint", "config", "latent_dim", "k", "target", "r2", "r2_shuffled", "delta", "mi_sum"])

    os.makedirs("probe_results/coefs", exist_ok=True)
    
    model_paths = glob.glob(yaml_out["checkpointing"]["load_paths"])
    all_states, gravities, lengths = [], [], []
    for data_path in glob.glob("datasets/*.npz"):
        data = np.load(data_path)
        states = data["states"][:yaml_out["collector"]["num_trajectories"]]
        gravity = float(data["gravity"])
        length = float(data["length"])
        all_states.append(states)
        gravities.append(np.full((states.shape[0], states.shape[1]), gravity))
        lengths.append(np.full((states.shape[0], states.shape[1]), length))

    combined_states = torch.from_numpy(np.concatenate(all_states, axis=0)).float()
    gravity_tensor = torch.from_numpy(np.concatenate(gravities, axis=0)).float()
    length_tensor = torch.from_numpy(np.concatenate(lengths, axis=0)).float()
    
    model_config = yaml_out["model"]
    model_params = {k: v for k, v in model_config.items()
            if k not in ("name")}
    for path in model_paths:
        current_config = parse_model(path)
        if current_config["flag"]:
            probe_states = combined_states  # just reference, no reassignment
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
                                            collector_config["save"])
            probe_states = torch.from_numpy(raw_states).float()
            targets = {
                "theta": torch.atan2(probe_states[:,:,1], probe_states[:,:,0]),
                "theta_dot": probe_states[:,:,2],
                "gravity": torch.full((probe_states.shape[0], probe_states.shape[1]), current_config["g"]),
                "length": torch.full((probe_states.shape[0], probe_states.shape[1]), current_config["l"]),
            }
        model_params["latent_dim"] = current_config["latent"]
        model = make_model(model_config["name"], **model_params)
        model.load_state_dict(torch.load(path))
        random_model = make_model(model_config["name"], **model_params)
        

        z_states = generate_latents(model, probe_states)
        z_random = generate_latents(random_model, probe_states)

        train_states, val_states, train_z, val_z = train_val_split(probe_states, z_states)
        _, _, train_z_rand, val_z_rand = train_val_split(probe_states, z_random)

        train_states_np = train_states.numpy()
        val_states_np = val_states.numpy()
        train_z_np = train_z.numpy()
        val_z_np = val_z.numpy()
        train_z_rand_np = train_z_rand.numpy()
        val_z_rand_np = val_z_rand.numpy()

        meta = {
            "checkpoint": os.path.basename(path),
            "config": current_config["config"],
            "latent": current_config["latent"],
            "k": current_config["k"]
        }

        print(f"\n[{meta['checkpoint']}]")
        print("=== Trained Model ===")
        for target_name, target in targets.items():
            train_target, val_target, _, _ = train_val_split(target.unsqueeze(-1), None)
            train_target_np = train_target.numpy().ravel()
            val_target_np = val_target.numpy().ravel()
            run_probe(train_z_np, val_z_np, train_target_np, val_target_np, target_name, writer, meta, yaml_out["probe"]["alpha"])

        print("=== Random Model Baseline ===")
        for target_name, target in targets.items():
            train_target, val_target, _, _ = train_val_split(target.unsqueeze(-1), None)
            train_target_np = train_target.numpy().ravel()
            val_target_np = val_target.numpy().ravel()
            run_probe(train_z_rand_np, val_z_rand_np, train_target_np, val_target_np, 
                    f"{target_name}_random", writer, meta, yaml_out["probe"]["alpha"])

        csv_file.flush()
    csv_file.close()
    print(f"\nDone. Results saved to {output_csv}")