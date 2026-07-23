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
from analysis.common import parse_model, load_model, iter_model_groups, get_data, HIDDEN_DIM, probeable_vars

def generate_latents(model, states: torch.Tensor, full: bool = False) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        N, T, state_dim = states.shape
        flat = states.reshape(-1, state_dim)
        comp = model.encode_computational(flat)
        if full and isinstance(comp, tuple):
            z = torch.cat(comp, dim=-1)
        else:
            z = model.probe_representation(comp)
        return z.reshape(N, T, -1)
    
def generate_latents_rollout(model, states: torch.Tensor,
                             full: bool = True, max_steps: int | None = None, probe_target = "full") -> torch.Tensor:
    model.eval()
    actions = torch.zeros(states.shape[0], states.shape[1], 1)
    with torch.inference_mode():
        N, T, _ = states.shape
        T_roll = T - 1 if max_steps is None else min(max_steps, T - 1)

        comp = model.encode_computational(states[:, 0, :])
        outs = []
        for t in range(T_roll):
            a = actions[:, t]
            if a.dim() == 1:
                a = a.unsqueeze(-1)

            comp = model.step_computational(comp, a)
            obs_comp = model.encode_computational(states[:, t + 1, :])
            if isinstance(comp, tuple):
                h_acc, _ = comp
                _, z_obs = obs_comp if isinstance(obs_comp, tuple) else (None, obs_comp)
                comp = (h_acc, z_obs)
                if probe_target == "full":
                    outs.append(torch.cat(comp, dim=-1) if full else z_obs)
                elif probe_target == 'h':
                    outs.append(h_acc)
                elif probe_target == 'z':
                    outs.append(z_obs)
            else:
                outs.append(model.probe_representation(comp) if not full else comp)

        return torch.stack(outs, dim=1)
    
def compute_mi(train_z, train_target, neighbours=5, n_sub=10000, n_perm=2, seed=0):
    rng = np.random.default_rng(seed)
    n = train_z.shape[0]
    if n > n_sub:
        idx = rng.choice(n, n_sub, replace=False)
        z_sub, t_sub = train_z[idx], train_target[idx]
    else:
        z_sub, t_sub = train_z, train_target
    mi = mutual_info_regression(z_sub, t_sub, n_neighbors=neighbours)
    mi_max = float(mi.max())
    mi_mean = float(mi.mean())

    if n_perm:
        perm_mat = np.stack([
        mutual_info_regression(z_sub, rng.permutation(t_sub), n_neighbors=neighbours)
        for _ in range(n_perm)
        ])
        perm      = float(perm_mat.mean())
        perm_max  = float(perm_mat.max(axis=1).mean())
    else:
        perm = np.nan
        perm_max = np.nan
    return mi_mean, mi_max, float(perm), float(perm_max)

def run_probe(train_z: np.ndarray, val_z: np.ndarray,
              train_target: np.ndarray, val_target: np.ndarray,
              label: str, writer, meta: dict, alpha: float, neighbours) -> tuple[float, float, float]:

    if not np.isfinite(train_z).all() or not np.isfinite(val_z).all():
        print(f"{label}: SKIPPED (NaN/inf in latents)")
        return float('nan'), float('nan'), float('nan')

    if np.abs(train_z).max() > 1e4:
        print(f"{label}: SKIPPED (latent explosion > 1e4)")
        return float('nan'), float('nan'), float('nan')
    probe = Ridge(alpha=alpha)
    probe.fit(train_z, train_target)
    r2 = r2_score(val_target, probe.predict(val_z))
    mi_mean, mi_max, perm, perm_max = compute_mi(train_z, train_target, neighbours)

    shuffled = train_target.copy()
    np.random.shuffle(shuffled)
    probe_shuffled = Ridge(alpha=alpha)
    probe_shuffled.fit(train_z, shuffled)
    r2_shuffled = r2_score(val_target, probe_shuffled.predict(val_z))
    delta = r2 - r2_shuffled
    shuf_mi = mi_max/perm_max
    print(f"{label}: R2={r2:.4f} | shuffled={r2_shuffled:.4f} | delta={delta:.4f} | MI={mi_mean:.4f}|MI max={mi_max:.4f}| Perm={perm:.4f}| Perm Max={perm_max:.4f}| Shuff={shuf_mi:.4f}")
    
    writer.writerow([
        meta["checkpoint"], meta["config"], meta["latent"], meta["k"],
        label, round(r2, 4), round(r2_shuffled, 4), round(delta, 4), round(mi_mean, 4), round(mi_max, 4), round(perm, 4), round(perm_max, 4), round(shuf_mi, 4)
    ])

    return r2, r2_shuffled, mi_mean


def stratified_probe_split(model, all_states, gravities, lengths,
                           is_cartpole, train_frac=0.8, regime="combined", representation="latent", roll = 200, probe_target = "full"):
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
    if representation == "latent":
        train_z = generate_latents(model, train_states)
        val_z   = generate_latents(model, val_states)
        latent_dim = train_z.shape[-1]
        train_z_flat = train_z.reshape(-1, latent_dim).numpy()
        val_z_flat   = val_z.reshape(-1, latent_dim).numpy()
    elif representation == "latent_full":
        train_z = generate_latents_rollout(model, train_states, full=True,max_steps=roll, probe_target=probe_target)
        val_z   = generate_latents_rollout(model, val_states, full=True, max_steps=roll, probe_target=probe_target)
        latent_dim = train_z.shape[-1]
        train_z_flat = train_z.reshape(-1, latent_dim).numpy()
        val_z_flat   = val_z.reshape(-1, latent_dim).numpy()
    elif representation == "state":
        latent_dim = train_states.shape[-1]
        train_z_flat = train_states.reshape(-1, latent_dim).numpy()
        val_z_flat = val_states.reshape(-1, latent_dim).numpy()

    def build_targets(states, g_t, l_t, regime, repr="latent", T_roll = 1):
        vars_here = probeable_vars(regime, is_cartpole)
        t = {}
        if repr == 'latent_full':
            s = states[:, 1:T_roll + 1, :]
            g = g_t[:, 1:T_roll + 1]
            l = l_t[:, 1:T_roll + 1]
            # print(states.shape, g.shape, l.shape)
            if "cos_theta" in vars_here:      t["cos_theta"] =  s[:, :, 0].reshape(-1).numpy()
            if "sin_theta" in vars_here:      t["sin_theta"] =  s[:, :, 1].reshape(-1).numpy()
            if "theta_dot" in vars_here:  t["theta_dot"] = s[:, :, 2].reshape(-1).numpy()
            if "x" in vars_here:          t["x"] = s[:, :, 3].reshape(-1).numpy()
            if "x_dot" in vars_here:      t["x_dot"] = s[:, :, 4].reshape(-1).numpy()
            if "gravity" in vars_here:    t["gravity"] = g.reshape(-1).numpy()
            if "length" in vars_here:     t["length"] = l.reshape(-1).numpy()
            if "g_over_l" in vars_here:   t["g_over_l"] = (g / l).reshape(-1).numpy()
            if "sqrt_l_over_g" in vars_here: t["sqrt_l_over_g"] = torch.sqrt(l / g).reshape(-1).numpy()
        
        if repr == "latent":
            if "cos_theta" in vars_here:      t["cos_theta"] =  states[:, :, 0].reshape(-1).numpy()
            if "sin_theta" in vars_here:      t["sin_theta"] =  states[:, :, 1].reshape(-1).numpy()
            if "theta_dot" in vars_here:  t["theta_dot"] = states[:, :, 2].reshape(-1).numpy()
            if "x" in vars_here:          t["x"] = states[:, :, 3].reshape(-1).numpy()
            if "x_dot" in vars_here:      t["x_dot"] = states[:, :, 4].reshape(-1).numpy()
            if "gravity" in vars_here:    t["gravity"] = g_t.reshape(-1).numpy()
            if "length" in vars_here:     t["length"] = l_t.reshape(-1).numpy()
            if "g_over_l" in vars_here:   t["g_over_l"] = (g_t / l_t).reshape(-1).numpy()
            if "sqrt_l_over_g" in vars_here: t["sqrt_l_over_g"] = torch.sqrt(l_t / g_t).reshape(-1).numpy()
        elif repr == 'state':
            if "gravity" in vars_here:    t["gravity"] = g_t.reshape(-1).numpy()
            if "length" in vars_here:     t["length"] = l_t.reshape(-1).numpy()
            if "g_over_l" in vars_here:   t["g_over_l"] = (g_t / l_t).reshape(-1).numpy()
            if "sqrt_l_over_g" in vars_here: t["sqrt_l_over_g"] = torch.sqrt(l_t / g_t).reshape(-1).numpy()
        return t
    T_roll = train_z.shape[1]
    return (train_z_flat, val_z_flat,
            build_targets(train_states, train_gt, train_lt, regime, representation, T_roll),
            build_targets(val_states, val_gt, val_lt, regime, representation, T_roll))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--save_dir", required = True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--state_type", default='latent')
    ap.add_argument("--neighbours", default = 5, type = int)
    ap.add_argument("--roll", type = int, default = 1200)
    ap.add_argument("--probe_target", default = 'full')
    args = ap.parse_args()
    if args.state_type == "state":
        args.random_init = False
    for (env_tag, policy_tag, model_tag), files in iter_model_groups(args.models_dir).items():
        tag = f"{env_tag}_{policy_tag}_{model_tag}"
        output_csv = f"{args.save_dir}/linear_probe_{tag}.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        write_header = not os.path.exists(output_csv)
        csv_file = open(output_csv, "a", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["checkpoint", "config", "latent_dim", "k", "target",
                             "r2", "r2_shuffled", "delta", "mi_mean", "mi_max", "perm", "perm_max", "mi_shuf_ratio"])

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
                model, states, gravities, lengths, is_cp, regime=cfg["regime"], representation=args.state_type, roll=args.roll, probe_target = args.probe_target)
            print("=== Trained ===")
            for tn in train_t:
                run_probe(train_z, val_z, train_t[tn], val_t[tn], tn, writer, meta, args.alpha, args.neighbours,)

            if random_model is not None:
                rz_tr, rz_val, rt_tr, rt_val = stratified_probe_split(
                    random_model, states, gravities, lengths, is_cp, regime=cfg["regime"], representation= args.state_type)
                print("=== Random baseline ===")
                for tn in rt_tr:
                    run_probe(rz_tr, rz_val, rt_tr[tn], rt_val[tn], f"{tn}_random", writer, meta, args.alpha, args.neighbours)

            csv_file.flush()
        csv_file.close()
        print(f"[{tag}] done -> {output_csv}")