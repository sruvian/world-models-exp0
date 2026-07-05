from pathlib import Path
import csv
import argparse
import numpy as np
import torch
from analysis.common import parse_model, iter_model_groups, load_model, collect_for_config

def compute_jacobian(model, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor | None:
    z = z.detach().requires_grad_(True)
    z_next = model.step_computational(z.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
    J = torch.zeros(z_next.shape[0], z.shape[0])
    for i in range(z_next.shape[0]):
        grad = torch.autograd.grad(z_next[i], z, retain_graph=True)[0]
        J[i] = grad.detach()
    if not torch.isfinite(J).all() or J.abs().max() > 1e8:
        return None
    return J

def action_jacobian(model, z, a):
    a = a.detach().requires_grad_(True)
    z_next = model.step_computational(z.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
    J_a = torch.zeros(z_next.shape[0])
    for i in range(z_next.shape[0]):
        grad = torch.autograd.grad(z_next[i], a, retain_graph=True)[0]
        J_a[i] = grad.detach()
    return J_a

def jacobian_stats(J: torch.Tensor, g: float, l: float, dt: float = 0.01, env = "PendulumSim") -> dict:

    if J is None:
        nan = float('nan')
        return {
            "max_eig": nan, "min_eig": nan, "mean_eig": nan,
            "unit_circle_frac": nan, "contracting_frac": nan, "expanding_frac": nan,
            "spectral_radius": nan, "min_singular": nan, "condition_number": nan,
            "mean_eig_phase": nan, "expected_phase": nan, "phase_error": nan,
            "g_over_l": round(g / l, 6),
            "j_a_max": nan, "j_a_mean": nan, "j_a_norm": nan,
        }
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--num_points", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    for (env_tag, policy_tag, model_tag), files in iter_model_groups(args.models_dir).items():
        csv_path = Path(f"jacobian_results/jacobian_{env_tag}_{args.num_points}_{policy_tag}_{model_tag}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        f = open(csv_path, "a", newline="")
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "checkpoint", "model_config", "eval_config", "latent_dim", "k",
                "max_eig", "min_eig", "mean_eig", "unit_circle_frac", "contracting_frac",
                "expanding_frac", "spectral_radius", "min_singular", "condition_number",
                "mean_eig_phase", "expected_phase", "phase_error",
                "std_spectral_radius", "std_phase_error", "g_over_l", "is_ood",
            ])

        for mf in files:
            cfg = parse_model(Path(mf))
            print(f"\n[{Path(mf).name}]")
            model = load_model(mf, cfg, args.device)

            eval_configs = (ALL_CONFIGS + OOD_CONFIGS) if cfg["flag"] else [(cfg["g"], cfg["l"])]
            impulse = (policy_tag == "sparse")

            for g_eval, l_eval in eval_configs:
                states_t, actions_t = collect_for_config(g_eval, l_eval, cfg["env"], impulse)
                is_ood = (g_eval, l_eval) not in ALL_CONFIGS
                all_stats = []
                for _ in range(args.num_points):
                    ti = np.random.randint(0, states_t.shape[0])
                    tt = np.random.randint(0, actions_t.shape[1])
                    s = states_t[ti, tt]
                    a = actions_t[ti, tt].unsqueeze(0)
                    with torch.enable_grad():
                        comp = model.encode_computational(s.unsqueeze(0)).squeeze(0)
                        J_s = compute_jacobian(model, comp, a)
                        J_a = action_jacobian(model, comp, a)
                    stats = jacobian_stats(J_s, g_eval, l_eval, env=cfg["env"])
                    if J_s is not None:
                        stats["j_a_max"] = round(float(J_a.abs().max()), 6)
                        stats["j_a_mean"] = round(float(J_a.abs().mean()), 6)
                        stats["j_a_norm"] = round(float(torch.norm(J_a)), 6)
                    all_stats.append(stats)

                avg = {k: round(float(np.mean([s[k] for s in all_stats])), 6) for k in all_stats[0]}
                std = {k: round(float(np.std([s[k] for s in all_stats])), 6) for k in all_stats[0]}
                writer.writerow([
                    Path(mf).name, cfg["config"], f"g{g_eval}_l{l_eval}",
                    cfg["latent"], cfg["k"],
                    avg["max_eig"], avg["min_eig"], avg["mean_eig"],
                    avg["unit_circle_frac"], avg["contracting_frac"], avg["expanding_frac"],
                    avg["spectral_radius"], avg["min_singular"], avg["condition_number"],
                    avg["mean_eig_phase"], avg["expected_phase"], avg["phase_error"],
                    std["spectral_radius"], std["phase_error"], avg["g_over_l"], is_ood,
                ])
                f.flush()
        f.close()
        print(f"[{env_tag}_{policy_tag}_{model_tag}] done -> {csv_path}")