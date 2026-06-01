from pathlib import Path
import os
import csv
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import make_model
from rolloutEngine import RolloutEngine
from sim_envs.envs import make_env
from collector import collect_trajectories
from utils import parse_model



def compute_jacobian(model, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    z = z.detach().requires_grad_(True)
    z_next = model.step(z.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
    J = torch.zeros(z_next.shape[0], z.shape[0])
    for i in range(z_next.shape[0]):
        grad = torch.autograd.grad(z_next[i], z, retain_graph=True)[0]
        J[i] = grad.detach()
    return J

def action_jacobian(model, z, a):
    a = a.detach().requires_grad_(True)
    z_next = model.step(z.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
    J_a = torch.zeros(z_next.shape[0])
    for i in range(z_next.shape[0]):
        grad = torch.autograd.grad(z_next[i], a, retain_graph=True)[0]
        J_a[i] = grad.detach()
    return J_a

def jacobian_stats(J: torch.Tensor, g: float, l: float, dt: float = 0.01, env = "PendulumSim") -> dict:
    eigenvalues = torch.linalg.eigvals(J)
    magnitudes = eigenvalues.abs()
    phases = torch.angle(eigenvalues)

    _, S, _ = torch.linalg.svd(J)

    if env == "CartPoleSim":
        expected_phase = np.sqrt((1.1 *g) / (1.0 * l)) * dt
        phase_error = round(float(abs(phases.abs().mean().item() - expected_phase)), 6)
    
    else:
        expected_phase = float(np.sqrt(g / l)  * dt)
        phase_error = float(abs(phases.abs().mean().item() - expected_phase))

    return {
        "max_eig":          round(float(magnitudes.max()), 6),

        "min_eig":          round(float(magnitudes.min()), 6),

        "mean_eig":         round(float(magnitudes.mean()), 6),

        "unit_circle_frac": round(float(((magnitudes > 0.95) & (magnitudes < 1.05)).float().mean()), 6),

        "contracting_frac": round(float((magnitudes < 0.95).float().mean()), 6),

        "expanding_frac":   round(float((magnitudes > 1.05).float().mean()), 6),

        "spectral_radius":  round(float(S.max()), 6),

        "min_singular":     round(float(S.min()), 6),

        "condition_number": round(float(S.max() / (S.min() + 1e-8)), 6),

        "mean_eig_phase":   round(float(phases.abs().mean()), 6),

        "expected_phase":   round(expected_phase, 6),

        "phase_error":      phase_error, 

        "g_over_l":         round(g / l, 6),
    }


ALL_CONFIGS = [
    (5.0,  2.0),  (5.0,  10.0), (5.0,  18.0),
    (9.8,  2.0),  (9.8,  10.0), (9.8,  18.0),
    (15.0, 2.0),  (15.0, 10.0), (15.0, 18.0),
]
OOD_CONFIGS = [
    (7.5, 6.0), (12.0, 14.0), (9.8, 6.0), (7.5, 10.0),
    (2.0, 2.0), (20.0, 2.0), (5.0, 25.0),
]
COLLECTOR = {
        "num_trajectories": 5,
        "episode_time": 1000,
        "policy_seed": 35,
        "save": False,
        "impulse_policy": False
    }


def collect_for_config(g: float, l: float, env: str) -> tuple:
    if env == "CartPoleSim":
        environment = make_env("CartPoleSim",
            gravity=g, mass1=0.1, mass2=1.0,
            length=l, dt=0.02,
            max_action=10.0, damping=0.0, seed=100
        )
    else:
        environment = make_env("PendulumSim",
            gravity=g, mass1=1.0, mass2=0.0,
            length=l, dt=0.01,
            max_action=10.0, damping=0.0, seed=100
        )
    states, actions, _ = collect_trajectories(
        environment,
        COLLECTOR["num_trajectories"],
        COLLECTOR["episode_time"],
        COLLECTOR["policy_seed"],
        COLLECTOR["save"],
        COLLECTOR["impulse_policy"]
    )
    return torch.from_numpy(states).float(), torch.from_numpy(actions).float()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--models_dir", type=Path, default=None)
    args.add_argument("--model_pt",   type=Path, default=None)
    args.add_argument("--num_points", type=int,  default=50)
    args.add_argument("--impulse_policy", action="store_true")
    parser = args.parse_args()

    pt_files: list[Path] = []
    if parser.models_dir:
        pt_files = [Path(p) for p in
                    glob.glob(str(parser.models_dir / "*.pt"))]
    if parser.model_pt:
        pt_files.append(parser.model_pt)

    COLLECTOR["impulse_policy"] = parser.impulse_policy

    tag = "noise"
    if COLLECTOR["impulse_policy"]:
        tag = "sparse"
    env_tag = "cartpole" if "cartpole" in str(parser.models_dir).lower() else "pendulum"
    csv_path = Path(f"probe_results/jacobian_results_{env_tag}_{parser.num_points}_{tag}.csv")
    csv_path.parent.mkdir(parents = True, exist_ok = True)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline = "")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "checkpoint", "model_config", "eval_config",
            "latent_dim", "k",
            "max_eig", "min_eig", "mean_eig",
            "unit_circle_frac", "contracting_frac", "expanding_frac",
            "spectral_radius", "min_singular", "condition_number",
            "mean_eig_phase", "expected_phase", "phase_error",
            "std_unit_circle", "std_spectral_radius", "std_phase", "std_phase_error",
            "g_over_l",
            "j_a_max", "j_a_mean", "j_a_norm",
            "is_ood",
        ])


    for file in pt_files:
        config = parse_model(file)
        print(f"\n[{file.name}]")
        state_dim = 5 if config["env"] == "CartPoleSim" else 3

        model_params = {
            "state_dim": state_dim, "action_dim": 1,
            "hidden_dim": 64, "latent_dim": config["latent"]
        }
        model = make_model("WorldModel", **model_params)
        model.load_state_dict(torch.load(file))
        model.eval()

        if config["flag"]:
            eval_configs = ALL_CONFIGS + OOD_CONFIGS
        else:
            eval_configs = [(config["g"], config["l"])]
        
        for g_eval, l_eval in eval_configs:
            states_t, actions_t = collect_for_config(g_eval, l_eval, config["env"])
            is_ood = (g_eval, l_eval) not in ALL_CONFIGS
            all_stats = []
            for _ in range(parser.num_points):
                traj_idx = np.random.randint(0, states_t.shape[0])
                time_idx = np.random.randint(0, actions_t.shape[1])
                s = states_t[traj_idx, time_idx]
                a = actions_t[traj_idx, time_idx].unsqueeze(0)

                with torch.enable_grad():
                    z = model.encode(s.unsqueeze(0)).squeeze(0)
                    J_s = compute_jacobian(model, z, a)
                    J_a = action_jacobian(model, z, a)

                stats = jacobian_stats( J_s, g_eval, l_eval, env = config["env"])
                stats["j_a_max"] = round(float(J_a.abs().max()), 6)
                stats["j_a_mean"] = round(float(J_a.abs().mean()), 6)
                stats["j_a_norm"] = round(float(torch.norm(J_a)), 6)
                
                all_stats.append(stats)

            avg = {k: round(float(np.mean([s[k] for s in all_stats])), 6)
                   for k in all_stats[0]}
            std = {k: round(float(np.std([s[k] for s in all_stats])), 6)
                    for k in all_stats[0]}

            print(f"  eval=g{g_eval}_l{l_eval} \n "
                  f"spectral_radius = {avg['spectral_radius']:.4f} \n "
                  f"unit_circle = {avg['unit_circle_frac']:.3f} \n "
                  f"phase = {avg['mean_eig_phase']:.4f} \n "
                  f"expected = {avg['expected_phase']:.4f} \n "
                  f"phase_err = {avg['phase_error']:.4f} \n"
                  f"std_spectral_radius = {std['spectral_radius']:.4f} \n"
                  f"std_phase_error = {std['phase_error']:.4f} \n" 
                  f"std_phase = {std['mean_eig_phase']:.4f} \n"
                  f"std_unit_circle = {std['unit_circle_frac']:.4f}"
                  )
            
            writer.writerow([
                file.name, config["config"], f"g{g_eval}_l{l_eval}",
                config["latent"], config["k"],
                avg["max_eig"], avg["min_eig"], avg["mean_eig"],
                avg["unit_circle_frac"], avg["contracting_frac"], avg["expanding_frac"],
                avg["spectral_radius"], avg["min_singular"], avg["condition_number"],
                avg["mean_eig_phase"], avg["expected_phase"], avg["phase_error"],
                std["unit_circle_frac"], std["spectral_radius"], std["mean_eig_phase"], std["phase_error"],
                avg["g_over_l"],
                avg["j_a_max"], avg["j_a_mean"], avg["j_a_norm"],
                is_ood,
            ])
            csv_file.flush()

    csv_file.close()
    print(f"\nDone. Results saved to {csv_path}")