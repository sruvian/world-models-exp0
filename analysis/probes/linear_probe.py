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
from analysis.common import parse_model, load_model, iter_model_groups, get_data, HIDDEN_DIM, probeable_vars, STATE_CHANNELS, DERIVED_TARGETS

def generate_latents(model, states, full=False, batch=8192):
    model.eval()
    with torch.inference_mode():
        N, T, D = states.shape
        if N == 0 or T == 0:
            raise ValueError(f"generate_latents got empty states: shape {states.shape}")
        flat = states.reshape(-1, D)
        outs = []
        for i in range(0, flat.shape[0], batch):
            comp = model.encode_computational(flat[i:i+batch])
            z = torch.cat(comp, dim=-1) if (full and isinstance(comp, tuple)) \
                else model.probe_representation(comp)
            outs.append(z)
        return torch.cat(outs, dim=0).reshape(N, T, -1)
        
def generate_latents_rollout(model, states, full=True, max_steps=None,
                             probe_target="full", traj_batch=512):
    model.eval()
    with torch.inference_mode():
        N, T, D = states.shape
        T_roll = T - 1 if max_steps is None else min(max_steps, T - 1)

        chunks = []
        for start in range(0, N, traj_batch):
            s = states[start:start + traj_batch]
            b = s.shape[0]
            actions = torch.zeros(b, T, 1)

            comp = model.encode_computational(s[:, 0, :])
            outs = []
            for t in range(T_roll):
                a = actions[:, t]
                if a.dim() == 1:
                    a = a.unsqueeze(-1)
                comp = model.step_computational(comp, a)
                obs_comp = model.encode_computational(s[:, t + 1, :])
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

            chunks.append(torch.stack(outs, dim=1))
        return torch.cat(chunks, dim=0)
    
def compute_mi(train_z, train_target, neighbours=5, n_sub=5000, n_perm=5, seed=0):
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
              label: str, writer, meta: dict, alpha: float, neighbours: int, perm: int) -> tuple[float, float, 
                                                                                                #  float
                                                                                                 ]:

    if not np.isfinite(train_z).all() or not np.isfinite(val_z).all():
        print(f"{label}: SKIPPED (NaN/inf in latents)")
        return float('nan'), float('nan')#, float('nan')

    if np.abs(train_z).max() > 1e4:
        print(f"{label}: SKIPPED (latent explosion > 1e4)")
        return float('nan'), float('nan')#, float('nan')
    probe = Ridge(alpha=alpha)
    probe.fit(train_z, train_target)
    r2 = r2_score(val_target, probe.predict(val_z))
    # mi_mean, mi_max, perm, perm_max = compute_mi(train_z, train_target, neighbours, n_perm=perm)

    shuffled = train_target.copy()
    np.random.shuffle(shuffled)
    probe_shuffled = Ridge(alpha=alpha)
    probe_shuffled.fit(train_z, shuffled)
    r2_shuffled = r2_score(val_target, probe_shuffled.predict(val_z))
    delta = r2 - r2_shuffled
    # shuf_mi = mi_max/perm_max
    # print(f"{label}: R2={r2:.4f} | shuffled={r2_shuffled:.4f} | delta={delta:.4f} | MI={mi_mean:.4f}|MI max={mi_max:.4f}| Perm={perm:.4f}| Perm Max={perm_max:.4f}| Shuff={shuf_mi:.4f}")
    
    writer.writerow([
        meta["checkpoint"], meta["config"], meta["latent"], meta["k"],
        label, round(r2, 4), round(r2_shuffled, 4), round(delta, 4), #round(mi_mean, 4), round(mi_max, 4), round(perm, 4), round(perm_max, 4), round(shuf_mi, 4)
    ])

    return r2, r2_shuffled#, mi_mean


def _build_probe_targets(states, params_arr, env_name, probe_targets,
                         regime, hold_param, repr, T_roll=1):
    channels = STATE_CHANNELS.get(env_name, {})
    vars_here = probeable_vars(probe_targets, env_name, regime, hold_param)
    t = {}

    if repr == "latent_full":
        s = states[:, 1:T_roll + 1, :]
        P = {k: v[:, 1:T_roll + 1] for k, v in params_arr.items()}
    else:
        s = states
        P = params_arr

    for name in vars_here:
        if name in channels:
            if repr == "state":
                continue
            t[name] = s[:, :, channels[name]].reshape(-1).numpy()
        elif name in DERIVED_TARGETS:
            val = DERIVED_TARGETS[name](P)
            t[name] = np.asarray(val).reshape(-1)
    return t

def stratified_probe_split(model, all_states, params_list, env_name,
                           probe_targets, train_frac=0.8, regime="combined",
                           hold_param=None, representation="latent",
                           roll=200, probe_target="full", max_traj_per_config=50, subsample_T=None):

    param_keys = sorted({k for p in params_list for k in p})
    train_s, val_s = [], []
    train_P = {k: [] for k in param_keys}
    val_P   = {k: [] for k in param_keys}

    for states_c, params in zip(all_states, params_list):
        if states_c.shape[0] == 0:
            print(f"WARNING: config {params} has 0 trajectories, skipping")
            continue
        if max_traj_per_config and states_c.shape[0] > max_traj_per_config:
            rng = np.random.default_rng(125)
            idx = rng.choice(states_c.shape[0], max_traj_per_config, replace=False)
            states_c = states_c[idx]
        N, T = states_c.shape[0], states_c.shape[1]
        ti = int(train_frac * N)
        ti = max(1, min(ti, N - 1))
        train_s.append(states_c[:ti]); val_s.append(states_c[ti:])
        for k in param_keys:
            arr = np.full((N, T), float(params.get(k, np.nan)), dtype=np.float32)
            train_P[k].append(arr[:ti]); val_P[k].append(arr[ti:])

    train_states = torch.from_numpy(np.concatenate(train_s, 0)).float()
    val_states   = torch.from_numpy(np.concatenate(val_s, 0)).float()
    train_params = {k: np.concatenate(v, 0) for k, v in train_P.items()}
    val_params   = {k: np.concatenate(v, 0) for k, v in val_P.items()}

    if representation == "latent":
        train_z = generate_latents(model, train_states)
        val_z   = generate_latents(model, val_states)
        latent_dim = train_z.shape[-1]
        train_z_flat = train_z.reshape(-1, latent_dim).numpy()
        val_z_flat   = val_z.reshape(-1, latent_dim).numpy()
    elif representation == "latent_full":
        train_z = generate_latents_rollout(model, train_states, full=True, max_steps=roll, probe_target=probe_target)
        val_z   = generate_latents_rollout(model, val_states,   full=True, max_steps=roll, probe_target=probe_target)
        latent_dim = train_z.shape[-1]
        train_z_flat = train_z.reshape(-1, latent_dim).numpy()
        val_z_flat   = val_z.reshape(-1, latent_dim).numpy()
    elif representation == "state":
        latent_dim = train_states.shape[-1]
        train_z_flat = train_states.reshape(-1, latent_dim).numpy()
        val_z_flat   = val_states.reshape(-1, latent_dim).numpy()
    else:
        raise ValueError(f"unknown representation: {representation}")

    T_roll = train_z.shape[1] if representation != "state" else train_states.shape[1]

    return (train_z_flat, val_z_flat,
            _build_probe_targets(train_states, train_params, env_name, probe_targets,
                                 regime, hold_param, representation, T_roll),
            _build_probe_targets(val_states, val_params, env_name, probe_targets,
                                 regime, hold_param, representation, T_roll))

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
    ap.add_argument("--perm", default = 2, type = int)
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
                             "r2", "r2_shuffled", "delta"#, "mi_mean", "mi_max", "perm", "perm_max", "mi_shuf_ratio"
                             ])

        for mf in files:
            cfg = parse_model(Path(mf))
            print(f"\n[{Path(mf).name}]")

            regime = cfg["regime"]
            hold_param, hold_value = None, None
            if regime == "holdg":
                regime, hold_param, hold_value = "hold", "gravity", 9.8
            elif regime == "holdl":
                regime, hold_param, hold_value = "hold", "length", 10.0
            cfg["regime"], cfg["hold_param"], cfg["hold_value"] = regime, hold_param, hold_value

            model = load_model(mf, cfg, args.device)

            random_model = None
            if args.random_init:
                state_dim = len(STATE_CHANNELS[cfg["env"]])
                random_model = make_model(cfg["model_name"], state_dim=state_dim, action_dim=1,
                    hidden_dim=HIDDEN_DIM.get(cfg["model_name"], 64), latent_dim=cfg["latent"])
                random_model.eval()

            states, params_list, probe_targets = get_data(cfg)
            env_name = cfg["env"]

            meta = {"checkpoint": Path(mf).name, "config": cfg["config"],
                    "latent": cfg["latent"], "k": cfg["k"], "policy": tag}

            train_z, val_z, train_t, val_t = stratified_probe_split(
                model, states, params_list, env_name, probe_targets,
                regime=cfg["regime"], hold_param=cfg.get("hold_param"),
                representation=args.state_type, roll=args.roll, probe_target=args.probe_target)

            print("=== Trained ===")
            for tn in train_t:
                run_probe(train_z, val_z, train_t[tn], val_t[tn], tn, writer, meta,
                        args.alpha, args.neighbours, args.perm)

            if random_model is not None:
                rz_tr, rz_val, rt_tr, rt_val = stratified_probe_split(
                    random_model, states, params_list, env_name, probe_targets,
                    regime=cfg["regime"], hold_param=cfg.get("hold_param"),
                    representation=args.state_type)
                print("=== Random baseline ===")
                for tn in rt_tr:
                    run_probe(rz_tr, rz_val, rt_tr[tn], rt_val[tn], f"{tn}_random", writer, meta,
                            args.alpha, args.neighbours, args.perm)

            csv_file.flush()
        csv_file.close()
        print(f"[{tag}] done -> {output_csv}")