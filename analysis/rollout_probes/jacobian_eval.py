from pathlib import Path
import csv
import argparse
import numpy as np
import torch
from analysis.common import parse_model, iter_model_groups, load_model, collect_for_config



def compute_jacobian_rssm(model, h, a):
    h = h.detach().requires_grad_(True)          # (hidden,)
    hb = h.unsqueeze(0)                           # (1, hidden)
    z = model.probe_state(hb).view(1, -1)         # (1, latent)
    ab = a.view(1, -1)                            # (1, action)
    h_next = model.gru(torch.cat([z, ab], dim=-1), hb).squeeze(0)   # (hidden,)
    J = torch.zeros(h_next.shape[0], h.shape[0])
    for i in range(h_next.shape[0]):
        grad = torch.autograd.grad(h_next[i], h, retain_graph=True)[0]
        J[i] = grad.detach()
    if not torch.isfinite(J).all() or J.abs().max() > 1e8:
        return None
    return J


def action_jacobian_rssm(model, h, a):
    a = a.view(1, -1).detach().requires_grad_(True)    # (1, action)
    hb = h.detach().view(1, -1)                        # (1, hidden)
    z = model.probe_state(hb).view(1, -1)              # (1, latent)
    h_next = model.gru(torch.cat([z, a], dim=-1), hb).squeeze(0)
    J_a = torch.zeros(h_next.shape[0])
    for i in range(h_next.shape[0]):
        grad = torch.autograd.grad(h_next[i], a, retain_graph=True)[0]
        J_a[i] = grad.detach().norm()   # scalar per output dim (works for action_dim>=1)
    return J_a

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

def jacobian_stats(J, g, l, dt=0.01, env="PendulumSim", recurrent=False) -> dict:

    PyTorchLinAlgError = getattr(torch._C, "_LinAlgError", RuntimeError)
    if J is None:
        return _nan_stats(g, l)
    try:
        eigenvalues = torch.linalg.eigvals(J)
        magnitudes = eigenvalues.abs()
        phases = torch.angle(eigenvalues)
    except PyTorchLinAlgError:
        return _nan_stats(g, l)
    

    try:
        _, S, _ = torch.linalg.svd(J)
    except PyTorchLinAlgError:
        # eigen worked but SVD didn't — record eigen stats, NaN the SVD ones
        S = None
    if recurrent:
        expected_phase = float('nan')
        phase_error = float('nan')
    else:    
        if env == "CartPoleSim":
                expected_phase = np.sqrt((1.1 * g) / (1.0 * l)) * dt
        else:
            expected_phase = float(np.sqrt(g / l) * dt)
        phase_error = float(abs(phases.abs().mean().item() - expected_phase))

    stats = {
        "max_eig":          round(float(magnitudes.max()), 6),
        "min_eig":          round(float(magnitudes.min()), 6),
        "mean_eig":         round(float(magnitudes.mean()), 6),
        "unit_circle_frac": round(float(((magnitudes > 0.95) & (magnitudes < 1.05)).float().mean()), 6),
        "contracting_frac": round(float((magnitudes < 0.95).float().mean()), 6),
        "expanding_frac":   round(float((magnitudes > 1.05).float().mean()), 6),
        "mean_eig_phase":   round(float(phases.abs().mean()), 6),
        "expected_phase":   round(expected_phase, 6),
        "phase_error":      phase_error,
        "g_over_l":         round(g / l, 6),
    }
    # SVD-derived stats (NaN if SVD failed)
    if S is not None:
        stats["spectral_radius"]  = round(float(S.max()), 6)
        stats["min_singular"]     = round(float(S.min()), 6)
        stats["condition_number"] = round(float(S.max() / (S.min() + 1e-8)), 6)
    else:
        stats["spectral_radius"]  = float('nan')
        stats["min_singular"]     = float('nan')
        stats["condition_number"] = float('nan')
    return stats
    

ALL_CONFIGS = [
    (5.0,  2.0),  (5.0,  10.0), (5.0,  18.0),
    (9.8,  2.0),  (9.8,  10.0), (9.8,  18.0),
    (15.0, 2.0),  (15.0, 10.0), (15.0, 18.0),
]
OOD_CONFIGS = [
    (7.5, 6.0), (12.0, 14.0), (9.8, 6.0), (7.5, 10.0),
    (2.0, 2.0), (20.0, 2.0), (5.0, 25.0),
]
DRIVEN_CONFIGS = [(15.0, 0.5), (9.8, 1.0), (15.0, 2.0), (9.8, 2.0), (5.0, 2.0)]
# DRIVEN_CONFIGS = [(15.0, 10.0)]
HELDOUT_CONFIGS = [(15, 10), (15, 18)]
COLLECTOR = {
        "num_trajectories": 5,
        "episode_time": 1000,
        "policy_seed": 35,
        "save": False,
        "impulse_policy": False
    }
def _nan_stats(g, l):
    nan = float('nan')
    return {
        "max_eig": nan, "min_eig": nan, "mean_eig": nan,
        "unit_circle_frac": nan, "contracting_frac": nan, "expanding_frac": nan,
        "spectral_radius": nan, "min_singular": nan, "condition_number": nan,
        "mean_eig_phase": nan, "expected_phase": nan, "phase_error": nan,
        "g_over_l": round(g / l, 6),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_points", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_traj", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--action", default = 0, type = float)
    args = ap.parse_args()

    for (env_tag, policy_tag, model_tag), files in iter_model_groups(args.models_dir).items():
        csv_path = Path(f"{args.out_dir}/jacobian_{env_tag}_{args.num_points}_{policy_tag}_{model_tag}.csv")
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
                "std_spectral_radius", "std_phase_error",  "j_a_max", "j_a_mean", "j_a_norm", "g_over_l", "is_ood",
            ])

        for mf in files:
            cfg = parse_model(Path(mf))
            print(f"\n[{Path(mf).name}]")
            model = load_model(mf, cfg, args.device)
            env_name = cfg["env"]
            if env_name != "DrivenPendulumSim":
                eval_configs = (ALL_CONFIGS + OOD_CONFIGS) if cfg["flag"] else [(cfg["g"], cfg["l"])]
            else:
                eval_configs = DRIVEN_CONFIGS + HELDOUT_CONFIGS
            impulse = (policy_tag == "sparse")

            for g_eval, l_eval in eval_configs:
                params = {"gravity": g_eval, "length": l_eval, "mass1": 1.0, "dt": 0.01, "damping": 0.0}
                if env_name == "CartPoleSim":
                    params.update(mass1=0.1, mass2=1.0)
                if env_name == "DrivenPendulumSim":
                    params.update(drive_omega= (g_eval/l_eval)**2, drive_amp = 0.2)
                states_t, actions_t = collect_for_config(
                    cfg["env"], params, impulse=cfg["impulse"],
                    seed=4200, n_traj=args.n_traj, steps=args.steps, max_action=args.action)
                if env_name != "DrivenPendulumSim":
                    is_ood = (g_eval, l_eval) not in ALL_CONFIGS
                else:
                    is_ood = (g_eval, l_eval) not in DRIVEN_CONFIGS
                all_stats = []
                for _ in range(args.num_points):
                    ti = np.random.randint(0, states_t.shape[0])
                    tt = np.random.randint(0, actions_t.shape[1])
                    s = states_t[ti, tt]
                    a = actions_t[ti, tt].unsqueeze(0)
                    with torch.enable_grad():
                        comp = model.encode_computational(s.unsqueeze(0))
                        if isinstance(comp, tuple):   # RSSM: roll to accumulate carry
                            h, z = model.initial(1, s.device)
                            h = h.view(1, -1)                          # (1, hidden)
                            for tau in range(tt):
                                s_tau = states_t[ti, tau].view(1, -1)  # (1, state)
                                a_tau = actions_t[ti, tau].view(1, -1) # (1, action)
                                z_post, _, _ = model.posterior(h, s_tau)   # (1, latent)
                                z_post = z_post.view(1, -1)
                                h = model.gru(torch.cat([z_post, a_tau], dim=-1), h)  # (1, hidden)
                                h = h.view(1, -1)
                            h = h.squeeze(0).detach()                  # (hidden,) for compute_jacobian_rssm
                            J_s = compute_jacobian_rssm(model, h, a)
                            J_a = action_jacobian_rssm(model, h, a)
                        else:                                            # MLP / DMD / GRU: tensor
                            comp = comp.squeeze(0)
                            with torch.enable_grad():
                                J_s = compute_jacobian(model, comp, a)
                                J_a = action_jacobian(model, comp, a)

                    stats = jacobian_stats(J_s, g_eval, l_eval, env=cfg["env"])
                    if J_s is not None and J_a is not None:
                        stats["j_a_max"]  = round(float(J_a.abs().max()), 6)
                        stats["j_a_mean"] = round(float(J_a.abs().mean()), 6)
                        stats["j_a_norm"] = round(float(torch.norm(J_a)), 6)
                    else:
                        stats["j_a_max"] = stats["j_a_mean"] = stats["j_a_norm"] = float('nan')
                    all_stats.append(stats)

                avg = {k: round(float(np.nanmean([s[k] for s in all_stats])), 6) for k in all_stats[0]}
                std = {k: round(float(np.nanstd([s[k] for s in all_stats])), 6) for k in all_stats[0]}
                writer.writerow([
                    Path(mf).name, cfg["config"], f"g{g_eval}_l{l_eval}",
                    cfg["latent"], cfg["k"],
                    avg["max_eig"], avg["min_eig"], avg["mean_eig"],
                    avg["unit_circle_frac"], avg["contracting_frac"], avg["expanding_frac"],
                    avg["spectral_radius"], avg["min_singular"], avg["condition_number"],
                    avg["mean_eig_phase"], avg["expected_phase"], avg["phase_error"],
                    std["spectral_radius"], std["phase_error"],
                    avg["j_a_max"], avg["j_a_mean"], avg["j_a_norm"],
                    avg["g_over_l"], is_ood,
                ])
                f.flush()
        f.close()
        print(f"[{env_tag}_{policy_tag}_{model_tag}] done -> {csv_path}")