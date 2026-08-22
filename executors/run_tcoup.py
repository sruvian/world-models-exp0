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
    (15, 0.5),
]
SWEEP_CONFIGS = TRAIN_CONFIGS + HELDOUT_CONFIGS
def _predict_traj(checker, states, actions):
    n_traj, T, _ = states.shape
    preds = []
    with torch.no_grad():
        for t0 in range(T - 1):
            z = checker.encode(states[:, t0, :])
            a = actions[:, t0].unsqueeze(-1)
            preds.append(checker.decode(checker.step(z, a)))
    return torch.stack(preds, dim=1)

def _config_params(env_name, g, l):
    p = {"gravity": g, "length": l, "mass1": 1.0, "dt": 0.01,
         "damping": 0.0, "max_action": 10.0}
    if env_name == "CartPoleSim":
        p.update(mass1=0.1, mass2=1.0)
    return p
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

def estimate_translational_coupling(checker, states, actions, dt):

    pred_traj = _predict_traj(checker, states, actions)
    if not torch.isfinite(pred_traj).all():
        return float("nan"), float("nan")

    cos_th = pred_traj[:, :, 0].detach().numpy()
    sin_th = pred_traj[:, :, 1].detach().numpy()
    thetadot = pred_traj[:, :, 2].detach().numpy()
    x_dot = pred_traj[:, :, 4].detach().numpy()

    x_ddot = np.gradient(x_dot, dt, axis=1, edge_order=2)

    theta_ddot = np.gradient(thetadot, dt, axis=1, edge_order=2)

    r = (thetadot ** 2) * sin_th - theta_ddot * cos_th

    a = actions[:, :-1].detach().numpy() if torch.is_tensor(actions) else np.asarray(actions)[:, :-1]

    finite = np.isfinite(x_ddot).all() and np.isfinite(r).all()
    if not finite:
        return float("nan"), float("nan")

    X = r.reshape(-1, 1)
    y = x_ddot.reshape(-1)
    reg = LinearRegression().fit(X, y)
    coupling_coeff = float(reg.coef_[0])
    return coupling_coeff, float(reg.score(X, y))

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
    ap.add_argument("--action", type = float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--n_traj", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    model_files = glob.glob(str(Path(args.models_dir) / "**" / "*.pt"), recursive=True)
    csv_handles = {}
    model_cache = {}

    for mf in model_files:
        cfg = parse_model(Path(mf))
        if cfg["env"] != "CartPoleSim":
            continue

        if mf not in model_cache:
            trained = load_model(mf, cfg, device = args.device)
            random_m = load_model(mf, cfg, device = args.device, random_init=True) \
                if _supports_random_init() else _build_random(mf, cfg, args.device)
            model_cache[mf] = {"trained": trained, "random": random_m}
        models = model_cache[mf]

        policy_tag = "sparse" if cfg["impulse"] else "noise"
        env_tag = 'cartpole' if cfg['env'] == 'CartPoleSim' else 'pendulum'
        group = f"{env_tag}_{policy_tag}_{TAGS[cfg['model_name']]}"
        if group not in csv_handles:
            path = Path(f"{args.save_dir}/tcoupling_{group}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            hdr = not path.exists()
            fh = open(path, "a", newline="")
            w = csv.writer(fh)
            if hdr:
                w.writerow(["checkpoint", "model_type", "latent_dim", "k", "regime",
                            "eval_g", "eval_l", "true_coupling", "measured_coupling", "r2", "config_tag"])
            csv_handles[group] = (fh, w)
        fh, w = csv_handles[group]

        for model_type, model in [("trained", models["trained"]), ("random", models["random"])]:
            checker = DoChecker(
                encode=model.encode_computational,
                step=model.step_computational,
                decode=model.decode_computational,
                env=cfg["env"], manifold_mult=2.0)

            for (g_eval, l_eval) in SWEEP_CONFIGS:
                params = _config_params(cfg["env"], g_eval, l_eval)
                states, actions = collect_for_config(
                    cfg["env"], params, impulse=cfg["impulse"],
                    seed=4200, n_traj=args.n_traj, steps=args.steps, max_action=args.action)
                states  = states  if torch.is_tensor(states)  else torch.from_numpy(states)
                actions = actions if torch.is_tensor(actions) else torch.from_numpy(actions)
                states, actions = states.float(), actions.float()

                k, r2 = estimate_translational_coupling(checker, states, actions, args.dt)
                true_coupling = (0.1*l_eval*0.5)/(1.1)
                tag = tag_config(g_eval, l_eval)
                w.writerow([Path(mf).name, model_type, cfg["latent"], cfg["k"], cfg["regime"],
                            g_eval, l_eval, round(true_coupling, 6),
                            (round(k, 6)  if np.isfinite(k)  else "nan"),
                            (round(r2, 6) if np.isfinite(r2) else "nan"), tag])
                fh.flush()
                print(f"[{Path(mf).name}] [{model_type:7s}] g={g_eval} l={l_eval} "
                    f"true={true_coupling:.3f} meas={k:.3f} r2={r2:.3f} [{tag}]")

    for fh, _ in csv_handles.values():
        fh.close()
    print("Done.")