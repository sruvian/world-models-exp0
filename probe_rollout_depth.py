from pathlib import Path
import glob
import csv
import argparse
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from models import make_model
from sim_envs.envs import make_env
from collector.collect import collect_trajectories


def parse_model(path: Path) -> dict:
    flag = False
    g, l = 0.0, 0.0
    k = 0
    name = path.stem
    components = name.split("_")[2:]
    if "combined" in components:
        flag = True
        k = int(components[1][1:])
    else:
        g = float(components[0][1:])
        l = float(components[1][1:])
        k = int(components[2][1:])
    latent = int(components[-1][6:])
    config = "combined" if flag else f"g{g}_l{l}"
    return {"flag": flag, "g": g, "l": l, "latent": latent, "k": k, "name": config}

def probe_at_depth(z: np.ndarray, target: np.ndarray,
                   alpha: float = 10.0) -> tuple[float, float]:
    n = z.shape[0]
    train_idx = int(0.8 * n)
    
    train_z, val_z = z[:train_idx], z[train_idx:]
    train_t, val_t = target[:train_idx], target[train_idx:]
    
    if val_t.std() < 1e-6:
        return 0.0, 0.0
    
    probe = Ridge(alpha=alpha)
    probe.fit(train_z, train_t)
    r2 = r2_score(val_t, probe.predict(val_z))
    
    shuffled = train_t.copy()
    np.random.shuffle(shuffled)
    probe_s = Ridge(alpha=alpha)
    probe_s.fit(train_z, shuffled)
    r2_s = r2_score(val_t, probe_s.predict(val_z))
    
    return round(float(r2), 6), round(float(r2_s), 6)


ALL_CONFIGS = [
    (5.0,  2.0), (5.0,  10.0), (5.0,  18.0),
    (9.8,  2.0), (9.8,  10.0), (9.8,  18.0),
    (15.0, 2.0), (15.0, 10.0), (15.0, 18.0),
]

PROBE_DEPTHS = [0, 1, 3, 5, 10, 15, 25, 50]
COLLECTOR = {
    "num_trajectories" : 50,
    "episode_time" : 100,
    "policy_seed" : 35,
    "save" : False,
    "impulse_policy": False
}


def collect_for_config(g: float, l: float) -> tuple:
    env = make_env("PendulumSim",
        gravity=g, mass1=1.0, mass2=0.0,
        length=l, dt=0.01,
        max_action=10.0, damping=0.0, seed=42
    )
    states, actions, _ = collect_trajectories(
        env,
        COLLECTOR["num_trajectories"],
        COLLECTOR["episode_time"],
        COLLECTOR["policy_seed"],
        COLLECTOR["save"],
        COLLECTOR['correlated'],
        COLLECTOR['correlation']
    )
    return (torch.from_numpy(states).float(),
            torch.from_numpy(actions).float())
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--models_dir", type=Path, default=None)
    args.add_argument("--model_pt", type=Path, default=None)
    args.add_argument("--alpha", type=float, default=10.0)
    parser = args.parse_args()

    pt_files: list[Path] = []
    if parser.models_dir:
        pt_files = [Path(p) for p in
                    glob.glob(str(parser.models_dir / "*.pt"))]
    if parser.model_pt:
        pt_files.append(parser.model_pt)

    csv_path = Path("probe_results/probe_rollout_depth.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "checkpoint", "model_config", "eval_config",
            "latent_dim", "k", "depth",
            "r2_theta", "r2s_theta", "delta_theta",
            "r2_thetadot", "r2s_thetadot", "delta_thetadot",
        ])

    for file in pt_files:
        config = parse_model(file)
        print(f"\n[{file.name}]")

        model_params = {
            "state_dim": 3, "action_dim": 1,
            "hidden_dim": 64, "latent_dim": config["latent"]
        }
        model = make_model("WorldModel", **model_params)
        model.load_state_dict(torch.load(file))
        model.eval()
        eval_configs = ALL_CONFIGS if config["flag"] \
            else [(config["g"], config["l"])]

        for g_eval, l_eval in eval_configs:
            states_t, actions_t = collect_for_config(g_eval, l_eval)
            N = states_t.shape[0]

            with torch.no_grad():
                z0 = model.encode(states_t[:, 0, :])

            z_current = z0.clone()

            for depth in PROBE_DEPTHS:
                if depth > 0:
                    with torch.no_grad():
                        prev_depth = PROBE_DEPTHS[PROBE_DEPTHS.index(depth) - 1]
                        steps_to_take = depth - prev_depth
                        for step in range(steps_to_take):
                            t_idx = prev_depth + step
                            if t_idx >= actions_t.shape[1]:
                                break
                            a = actions_t[:, t_idx].unsqueeze(-1)
                            z_current = model.step(z_current, a)

                z_np = z_current.numpy()

                if depth < states_t.shape[1]:
                    s_true = states_t[:, depth, :]
                else:
                    break

                theta_true = torch.atan2(
                    s_true[:, 1], s_true[:, 0]).numpy()
                thetadot_true = s_true[:, 2].numpy()

                r2_th, r2s_th = probe_at_depth(z_np, theta_true, parser.alpha)
                r2_td, r2s_td = probe_at_depth(z_np, thetadot_true, parser.alpha)

                delta_th = round(r2_th - r2s_th, 6)
                delta_td = round(r2_td - r2s_td, 6)

                print(f"  g={g_eval} l={l_eval} depth={depth:3d} | "
                      f"theta_delta={delta_th:.4f} "
                      f"thetadot_delta={delta_td:.4f}")

                writer.writerow([
                    file.name, config["name"], f"g{g_eval}_l{l_eval}",
                    config["latent"], config["k"], depth,
                    r2_th, r2s_th, delta_th,
                    r2_td, r2s_td, delta_td,
                ])
                csv_file.flush()

    csv_file.close()
    print(f"\nDone. Saved to {csv_path}")