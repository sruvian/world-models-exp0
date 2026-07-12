import argparse, glob, csv
from pathlib import Path
import numpy as np
import torch
from analysis.common.utils import parse_model, load_model, TAGS
from analysis.common.data_provider import collect_for_config
from analysis.do_checker.checker import DoChecker
from analysis.common.utils import make_model, HIDDEN_DIM
from sklearn.linear_model import LinearRegression
TRAIN_G = [5.0, 9.8, 15.0]
TRAIN_L = [2.0, 10.0, 18.0]

TRAIN_CONFIGS = [(g, l) for g in TRAIN_G for l in TRAIN_L]
HELDOUT_CONFIGS = [
    (7.0, 6.0),
    (12.0, 14.0),
    (7.0, 20.0),
    (17.0, 10.0),
    (17.0, 20.0),
    (3.0, 10.0),
]
SWEEP_CONFIGS = TRAIN_CONFIGS + HELDOUT_CONFIGS


def tag_config(g, l):
    g_in = g in TRAIN_G
    l_in = l in TRAIN_L
    if g_in and l_in:
        return "in_training"
    g_extrap = (g < min(TRAIN_G)) or (g > max(TRAIN_G))
    l_extrap = (l < min(TRAIN_L)) or (l > max(TRAIN_L))
    if g_extrap or l_extrap:
        return "extrapolated"
    return "interpolated"


def estimate_effective_gl(checker, states, actions, dt):
    
    n_traj, T, _ = states.shape
    preds = []
    with torch.no_grad():
        for t0 in range(T - 1):
            z = checker.encode(states[:, t0, :])
            a = actions[:, t0].unsqueeze(-1)
            preds.append(checker.decode(checker.step(z, a)))
    pred_traj = torch.stack(preds, dim=1)

    if not torch.isfinite(pred_traj).all():
        return float("nan"), float("nan")

    theta = torch.arctan2(pred_traj[:, :, 1], pred_traj[:, :, 0])
    theta_u = np.unwrap(theta.detach().numpy(), axis=1)
    thetadot = pred_traj[:, :, 2].detach().numpy()
    theta_ddot = np.gradient(thetadot, dt, axis=1, edge_order=1)
    sin_th = np.sin(theta_u)

    if not (np.isfinite(theta_ddot).all() and np.isfinite(sin_th).all()):
        return float("nan"), float("nan")

    X = (-sin_th).reshape(-1, 1)
    y = theta_ddot.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    return float(reg.coef_[0][0]), float(reg.score(X, y))


def _supports_random_init():
    return False


def _build_random(mf, cfg, device):
    
    is_cp = cfg["env"] == "CartPoleSim"
    model = make_model(cfg["model_name"],
                       state_dim=5 if is_cp else 3, action_dim=1,
                       hidden_dim=HIDDEN_DIM.get(cfg["model_name"], 64),
                       latent_dim=cfg["latent"])
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--n_traj", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    model_files = glob.glob(str(Path(args.models_dir) / "**" / "*.pt"), recursive=True)
    csv_handles = {}
    model_cache = {}

    for mf in model_files:
        cfg = parse_model(Path(mf))
        # if cfg["env"] == "CartPoleSim":
        #     continue    # pendulum-only estimator

        if mf not in model_cache:
            trained = load_model(mf, cfg, args.device)
            random_m = load_model(mf, cfg, args.device, random_init=True) \
                if _supports_random_init() else _build_random(mf, cfg, args.device)
            model_cache[mf] = {"trained": trained, "random": random_m}
        models = model_cache[mf]

        policy_tag = "sparse" if cfg["impulse"] else "noise"
        env_tag = 'cartpole' if cfg['env'] == 'CartPoleSim' else 'pendulum'
        group = f"{env_tag}_{policy_tag}_{TAGS[cfg['model_name']]}"
        if group not in csv_handles:
            path = Path(f"{args.save_dir}/behavioural_{group}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            hdr = not path.exists()
            fh = open(path, "a", newline="")
            w = csv.writer(fh)
            if hdr:
                w.writerow(["checkpoint", "model_type", "latent_dim", "k", "regime",
                            "eval_g", "eval_l", "true_gl", "measured_gl", "r2", "config_tag"])
            csv_handles[group] = (fh, w)
        fh, w = csv_handles[group]

        for model_type, model in [("trained", models["trained"]), ("random", models["random"])]:
            checker = DoChecker(
                encode=model.encode_computational,
                step=model.step_computational,
                decode=model.decode_computational,
                env=cfg["env"], manifold_mult=2.0)

            for (g_eval, l_eval) in SWEEP_CONFIGS:
                states, actions = collect_for_config(
                    g_eval, l_eval, cfg["env"], cfg["impulse"],
                    seed=4200, n_traj=args.n_traj, steps=args.steps)
                states = states.float() if torch.is_tensor(states) else torch.from_numpy(states).float()
                actions = actions.float() if torch.is_tensor(actions) else torch.from_numpy(actions).float()

                k, r2 = estimate_effective_gl(checker, states, actions, args.dt)
                true_gl = g_eval / l_eval
                tag = tag_config(g_eval, l_eval)
                w.writerow([Path(mf).name, model_type, cfg["latent"], cfg["k"], cfg["regime"],
                            g_eval, l_eval, round(true_gl, 6),
                            (round(k, 6) if np.isfinite(k) else "nan"),
                            (round(r2, 6) if np.isfinite(r2) else "nan"), tag])
                fh.flush()
                print(f"[{Path(mf).name}] [{model_type:7s}] g={g_eval} l={l_eval} "
                      f"true={true_gl:.3f} meas={k:.3f} r2={r2:.3f} [{tag}]")

    for fh, _ in csv_handles.values():
        fh.close()
    print("Done.")