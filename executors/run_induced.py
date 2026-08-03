import argparse, glob, csv
from pathlib import Path
import numpy as np
import torch
from analysis.common.utils import parse_model, load_model, TAGS
from analysis.common.data_provider import collect_for_config
from sim_envs.envs import make_env

EPS = 1e-3
N_STATES = 100
STATE_CHANNEL = {"theta_dot": 2}
def simulator_next_state(env_name, mass1, mass2, g, l, s, action, dt=0.01):

    cfg = dict(gravity=g, mass1=mass1, mass2=mass2, length=l, dt=dt, max_action=10.0, damping=0.0, seed=42)
    env = make_env(env_name, **cfg)
    env.reset()
    if env_name == "PendulumSim":
        theta = np.arctan2(s[1], s[0])
        env.theta = theta
        env.theta_dot = float(s[2])

    elif env_name == "CartPoleSim":
        theta = np.arctan2(s[1], s[0])
        env.theta = theta
        env.theta_dot = float(s[2])
        env.x = float(s[3])
        env.x_dot = float(s[4])
    nxt = env.step(float(action))
    return np.asarray(nxt, dtype=np.float64)[:len(s)]


def compute_Sx(env_name, mass1, mass2, g, l, s, action):
    dg_p = simulator_next_state(env_name, mass1, mass2, g + EPS, l, s, action)
    dg_m = simulator_next_state(env_name, mass1, mass2, g - EPS, l, s, action)
    dl_p = simulator_next_state(env_name, mass1, mass2, g, l + EPS, s, action)
    dl_m = simulator_next_state(env_name, mass1, mass2, g, l - EPS, s, action)
    d_dg = (dg_p - dg_m) / (2 * EPS)
    d_dl = (dl_p - dl_m) / (2 * EPS)
    return np.stack([d_dg, d_dl], axis=1)




def encoder_jacobian(model, s):
    s_t = torch.as_tensor(s, dtype=torch.float32).reshape(1, -1).requires_grad_(True)

    def enc_z(x):
        e = model.encode(x)
        if isinstance(e, tuple):
            e = e[1]
        return e.reshape(-1)

    J = torch.autograd.functional.jacobian(enc_z, s_t)
    return J.reshape(J.shape[0], -1).detach().numpy()


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--action", type=float, default=0.0)
    ap.add_argument("--eval_g", type=float, default=9.8)
    ap.add_argument("--eval_l", type=float, default=10.0)
    args = ap.parse_args()

    csv_handles, model_cache = {}, {}

    for rf in glob.glob(str(Path(args.readouts_dir) / "**" / "*.npz"), recursive=True):
        d = np.load(rf, allow_pickle=True)
        w_probe = np.asarray(d["direction"], dtype=np.float64)
        meta = {k: d[k].item() for k in d.files if k != "direction"}
        variable = meta["variable"]
        ckpt = meta["checkpoint"]

        cfg = parse_model(Path(ckpt))
        if not Path(ckpt).exists():
            print(f"[skip] model not found: {ckpt}"); continue

        if ckpt not in model_cache:
            model_cache[ckpt] = load_model(ckpt, cfg, args.device)
        model = model_cache[ckpt]

        states, actions = collect_for_config(args.eval_g, args.eval_l, cfg["env"],
                                              cfg["impulse"], seed=4200, n_traj=5, steps=200, max_action=args.action)
        states = states[:, :-1]
        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1)
        states = states.numpy() if torch.is_tensor(states) else np.asarray(states)
        actions = actions.numpy() if torch.is_tensor(actions) else np.asarray(actions)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(states), min(N_STATES, len(states)), replace=False)

        is_param = variable in ("gravity", "length")
        chan = STATE_CHANNEL.get(variable)

        cosines = []
        mass1, mass2 = 0, 0
        if cfg['env'] == 'CartPoleSim':
            mass1, mass2 = 0.1, 1.0
        elif cfg['env'] == 'PendulumSim':
            mass1, mass2 = 1.0, 0.0
        else:
            raise ValueError("Fix parse models")
        v = None
        if is_param:
            F = np.zeros((2, 2), dtype=np.float64)
            for s, a in zip(states, actions):
                Sx = compute_Sx(
                    cfg["env"],
                    mass1,
                    mass2,
                    args.eval_g,
                    args.eval_l,
                    s.astype(np.float64),
                    float(a),
                )
                F += Sx.T @ Sx

            eigvals, eigvecs = np.linalg.eigh(F)
            v = eigvecs[:, -1]
            v /= np.linalg.norm(v) + 1e-12

        cosines = []

        for i in idx:

            s = states[i].astype(np.float64)
            a = float(actions[i])

            J_E = encoder_jacobian(model, s)

            if is_param:

                Sx = compute_Sx(cfg["env"], mass1, mass2, args.eval_g, args.eval_l, s, a,)

                dz_psi = J_E @ (Sx @ v)
            elif chan is not None:
                dz_psi = J_E[:, chan]
            else:
                continue

            cosines.append(cos(w_probe, dz_psi))
        cosines = np.asarray(cosines, dtype=np.float64)

        if len(cosines) == 0:
            continue

        abs_cos = np.abs(cosines)

        policy_tag = "sparse" if cfg["impulse"] else "noise"
        env_tag = cfg["env"].replace("Sim", "").lower()
        group = f"{env_tag}_{policy_tag}_{TAGS[cfg['model_name']]}"
        if group not in csv_handles:
            path = Path(f"{args.save_dir}/induced_{group}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            hdr = not path.exists()
            fh = open(path, "a", newline="")
            w = csv.writer(fh)
            if hdr:
                w.writerow(["checkpoint", "variable", "latent_dim", "k", "regime",
                            "cos_mean", "cos_std", "abscos_mean", "n_states"])
            csv_handles[group] = (fh, w)
        fh, w = csv_handles[group]
        w.writerow([Path(ckpt).name, variable, cfg["latent"], cfg["k"], cfg["regime"],
                    round(float(cosines.mean()), 6), round(float(cosines.std()), 6),
                    round(float(abs_cos.mean()), 6), len(cosines)])
        fh.flush()
        print(f"[{Path(ckpt).name}] {variable}: cos={cosines.mean():+.4f}+/-{cosines.std():.4f} "
              f"|cos|={abs_cos.mean():.4f} (n={len(cosines)})")

    for fh, _ in csv_handles.values():
        fh.close()
    print("Done.")