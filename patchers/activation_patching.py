import numpy as np
import os
import torch
import argparse
from pathlib import Path
import glob
from models import make_model
from utils import parse_model
from sim_envs import make_env
from collector import collect_trajectories
import csv

from models import WorldModel, ProtocolAModel, ProtocolBModel
ALL_CONFIGS = [
    (5.0, 2.0), (5.0, 10.0), (5.0, 18.0),
    (9.8, 2.0), (9.8, 10.0), (9.8, 18.0),
    (15.0, 2.0), (15.0, 10.0), (15.0, 18.0),
]
def get_angular_dims(probe_coef: Path, top_k: int = 3)-> np.ndarray:
    coef = np.load(probe_coef)
    return np.argsort(np.abs(coef))[-top_k:]

def patch_trajectories(model: WorldModel| ProtocolAModel| ProtocolBModel, source_traj: torch.Tensor, target_traj: torch.Tensor, 
                       angular_dims, random_patch: bool = False):

    z_source = model.encode(source_traj)
    z_target = model.encode(target_traj)
    batch= source_traj.shape[0]
    action = torch.zeros(batch, 1)
    z_patched = z_target.clone()

    if random_patch:
        z_patched[:, angular_dims] = torch.randn_like(z_source[:, angular_dims])
    else:
        z_patched[:, angular_dims] = z_source[:, angular_dims]

    z_source_step = model.step(z_source, action)
    z_target_step = model.step(z_target, action)
    z_patched_step = model.step(z_patched, action)


    s_hat_baseline = model.decode(z_target_step)
    s_hat_patched  = model.decode(z_patched_step)
    s_hat_source   = model.decode(z_source_step)
    

    baseline_err       = ((s_hat_baseline[:, :2] - s_hat_source[:, :2])**2).mean().sqrt().item()
    patched_err        = ((s_hat_patched[:, :2]  - s_hat_source[:, :2])**2).mean().sqrt().item()
    baseline_err_source = ((s_hat_baseline[:, :2] - source_traj[:, :2])**2).mean().sqrt().item()
    patched_err_source  = ((s_hat_patched[:, :2]  - source_traj[:, :2])**2).mean().sqrt().item()

    shift        = baseline_err - patched_err
    shift_source = baseline_err_source - patched_err_source

    return shift, shift_source, baseline_err, patched_err, baseline_err_source, patched_err_source

def make_env_for_config(env_name: str, g: float, l: float, seed: int):
    if env_name == "CartPoleSim":
        return make_env("CartPoleSim",
            gravity=g, mass1=0.1, mass2=1.0,
            length=l, dt=0.01,
            max_action=10.0, damping=0.0, seed=seed)
    else:
        return make_env("PendulumSim",
            gravity=g, mass1=1.0, mass2=0.0,
            length=l, dt=0.01,
            max_action=10.0, damping=0.0, seed=seed)
    
COLLECTOR = {
    "num_trajectories" : 50,
    "episode_time" : 100,
    "policy_seed" : 35,
    "save" : False,
    "impulse_policy": False
}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--probes_dir", type=Path, required=True)
    args.add_argument("--models_dir", type=Path, required=True)
    args.add_argument("--device", type=str, default="cpu")
    parser = args.parse_args()

    device = parser.device
    pt_files_sample = glob.glob(str(parser.probes_dir / "*.npy"))
    if not pt_files_sample:
        raise ValueError("No probe files found")

    first_parts = Path(pt_files_sample[0]).name.split(".pt_")
    first_model = parser.models_dir / (first_parts[0] + ".pt")
    first_config = parse_model(first_model)
    env_tag = "cartpole" if first_config["env"] == "CartPoleSim" else "pendulum"

    policy_tag = "sparse" if "sparse" in str(parser.probes_dir).lower() else "noise"
    csv_path = Path(f"probe_results/activation_patch_{env_tag}_{policy_tag}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "checkpoint", "target_var", "policy_tag", "eval_config",
            "latent_dim", "k", "top_k", "is_ood", "random_patch",
            "shift", "shift_source",
            "baseline_err", "patched_err",
            "baseline_err_source", "patched_err_source"
        ])

    probe_files = glob.glob(str(parser.probes_dir / "*.npy"))
    model_cache = {}

    for probe_coef in probe_files:
        parts = Path(probe_coef).name.split(".pt_")
        if len(parts) != 2:
            print(f"Skipping malformed filename: {probe_coef}")
            continue

        checkpoint_name = parts[0] + ".pt"
        remainder = parts[1]
        remainder_no_ext = remainder.replace(".npy", "")
        split = remainder_no_ext.split("_")
        policy_tag = split[-1]
        target_var = "_".join(split[:-1])

        if target_var.endswith("_random"):
            continue

        model_file = parser.models_dir / checkpoint_name
        if not model_file.exists():
            print(f"Model not found: {model_file}")
            continue

        config = parse_model(model_file)
        print(f"\n[{checkpoint_name}] target={target_var} policy={policy_tag}")

        if checkpoint_name not in model_cache:
            state_dim = 5 if config["env"] == "CartPoleSim" else 3
            model_params = {
                "state_dim": state_dim, "action_dim": 1,
                "hidden_dim": 64, "latent_dim": config["latent"]
            }
            model = make_model("WorldModel", **model_params)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            model_cache[checkpoint_name] = model
        else:
            model = model_cache[checkpoint_name]

        if config["flag"]:
            eval_configs = ALL_CONFIGS
        else:
            eval_configs = [(config["g"], config["l"])]

        for g_eval, l_eval in eval_configs:
            is_ood = (g_eval, l_eval) not in ALL_CONFIGS

            source_env = make_env_for_config(config["env"], g_eval, l_eval, seed=42)
            target_env = make_env_for_config(config["env"], g_eval, l_eval, seed=100)

            source_states, _, _ = collect_trajectories(
                source_env, COLLECTOR["num_trajectories"], COLLECTOR["episode_time"],
                COLLECTOR["policy_seed"], COLLECTOR["save"], config["impulse"]
            )
            target_states, _, _ = collect_trajectories(
                target_env, COLLECTOR["num_trajectories"], COLLECTOR["episode_time"],
                COLLECTOR["policy_seed"], COLLECTOR["save"], config["impulse"]
            )

            source_states = torch.from_numpy(source_states).to(device)
            target_states = torch.from_numpy(target_states).to(device)
            N, T, D = source_states.shape
            source_flat = source_states.float().reshape(-1, D)
            target_flat = target_states.float().reshape(-1, D)

            eval_tag = f"g{g_eval}_l{l_eval}"
            print(f"  eval={eval_tag}")

            for top_k in [1, 2, 3, 4, 5]:
                angular_dims = get_angular_dims(Path(probe_coef), top_k=top_k)

                for random_patch in [False, True]:
                    shift, shift_source, bl_err, pt_err, bl_err_src, pt_err_src = patch_trajectories(
                        model, source_flat, target_flat, angular_dims, random_patch=random_patch
                    )
                    print(f"    top_k={top_k} random={random_patch} | "
                            f"shift={shift:.4f} shift_src={shift_source:.4f}")

                    writer.writerow([
                        checkpoint_name, target_var, policy_tag, eval_tag,
                        config["latent"], config["k"], top_k, is_ood, random_patch,
                        round(shift, 6), round(shift_source, 6),
                        round(bl_err, 6), round(pt_err, 6),
                        round(bl_err_src, 6), round(pt_err_src, 6)
                    ])
                    csv_file.flush()

    csv_file.close()
    print(f"\nDone. Saved to {csv_path}")