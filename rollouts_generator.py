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
from scipy.fft import fft

def parse_model(path: Path) -> dict:
    flag = False
    g, l = 0.0, 0.0
    latent = 0
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


def generate_trajectories(env_confs, collector, action):
    environment_config = {
            "gravity": 0.0,
            "mass1": 1.0,
            "mass2": 0,
            "length": 0.0,
            "dt" : 0.01,
            "max_action": action,
            "damping": 0.0,
            "seed": 80,
            }
    if env_confs is not None:
        environment_config["gravity"] = env_confs[0]
        environment_config["length"] = env_confs[1]
        environment = make_env("PendulumSim",**environment_config)
        states, actions, _ = collect_trajectories(
                                        environment,
                                        collector["num_trajectories"],
                                        collector["episode_time"],
                                        collector["policy_seed"],
                                        collector["save"],
                                        collector["impulse_policy"]
                                    )
        
    else:
        all_configs = [(5.0, 2.0), (5.0, 10.0), (5.0, 18.0),
                       (9.8, 2.0), (9.8, 10.0), (9.8, 18.0),
                       (15.0, 2.0), (15.0, 10.0), (15.0, 18.0)
            ]
        all_states, all_actions = [], []
        for conf in all_configs:
            environment_config["gravity"] = conf[0]
            environment_config["length"] = conf[1]
            environment = make_env("PendulumSim",**environment_config)
            states, actions, _ = collect_trajectories(
                                        environment,
                                        collector["num_trajectories"],
                                        collector["episode_time"],
                                        collector["policy_seed"],
                                        collector["save"],
                                        collector["impulse_policy"]
                                    )
            all_states.append(states)
            all_actions.append(actions)
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
    return states, actions



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--models_dir", type=Path, default=None)
    args.add_argument("--model_pt", type=Path, default=None)
    args.add_argument("--top_dir", type=Path, required=True)
    args.add_argument("--impulse_policy", action="store_true", default = False)
    parser = args.parse_args()

    pt_files: list[Path] = []
    if parser.models_dir:
        if parser.models_dir.exists():
            pt_files = [Path(p) for p in 
                       glob.glob(str(parser.models_dir / "*.pt"))]
        else:
            raise ValueError(f"Path does not exist: {parser.models_dir}")
    if parser.model_pt:
        if parser.model_pt.exists():
            pt_files.append(parser.model_pt)
        else:
            raise ValueError(f"Path does not exist: {parser.model_pt}")
        
    if parser.impulse_policy:
        temp_files = []
        for file in pt_files:
            if "impulse_policy" in str(file):
                temp_files.append(file)
        pt_files = temp_files
        
    top_dir = parser.top_dir
    if parser.impulse_policy:
        top_dir = os.path.join(top_dir, "impulse_policy")
    os.makedirs(top_dir, exist_ok=True)
    environment_actions = [0, 0.5, 5, 10, 15, 20, 30, 50]
    horizons = [50, 500, 5000]
    

    csv_path = Path(f"{top_dir}/pendulum_rollout_meta.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "checkpoint", "latent_dim", "k", "gravity", "length",
            "horizon", "torque", "mse_loss", "valid_horizon",
            "rmse_cos", "rmse_sin", "rmse_thetadot",
            "freq_error", "latent_div"
        ])

    loss_func = torch.nn.MSELoss()

    collector = {
                "num_trajectories": 1,
                "episode_time": 5001,
                "policy_seed": 35,
                "save": False,
                "impulse_policy" : parser.impulse_policy
                }

    for file in pt_files:
        config = parse_model(file)

        model_params = {
            "state_dim": 3, "action_dim": 1,
            "hidden_dim": 64, "latent_dim": config["latent"]
        }
        model = make_model("WorldModel", **model_params)
        model.load_state_dict(torch.load(file))
        model.eval()
        model_dir = top_dir / f"latent{config['latent']}" / \
                    f"k{config['k']}" / \
                    f"{config['name']}"
        model_dir.mkdir(parents=True, exist_ok=True)
        

        roll_eng = RolloutEngine(model, loss_func)
        if config["flag"]:
            env_confs = None
        else:
            env_confs = (config["g"], config["l"])
   
        for action in environment_actions:
            states, actions = generate_trajectories(env_confs, collector, action)
            for horizon in horizons:
                roll_states = torch.from_numpy(states).float()
                roll_actions = torch.from_numpy(actions).float()
                preds, roll_loss = roll_eng.rollout(roll_states, roll_actions, horizon)

                true_np = roll_states[0, 1:horizon+1].numpy()
                pred_np = preds[0].numpy()

                rmse = np.sqrt(np.mean((pred_np - true_np)**2, axis=0)) / (true_np.std(axis=0))

                errors = np.abs(pred_np - true_np).mean(axis=-1)
                valid_h = int(np.argmax(errors > 0.1))
                if valid_h == 0 and errors[0] <= 0.1:
                    valid_h = horizon

                freq_error = 0.0
                if horizon >= 500:
                    cos_pred = pred_np[:, 0].astype(np.float64)
                    cos_true = true_np[:, 0].astype(np.float64)
                    pred_freq = np.argmax(np.abs(np.array(fft(cos_pred))))
                    true_freq = np.argmax(np.abs(np.array(fft(cos_true))))
                    if true_freq < 2:
                        freq_error = 0.0
                    else:
                        freq_error = round(float(abs(pred_freq - true_freq) / (true_freq + 1e-8)), 6)

                with torch.no_grad():
                    z_true = model.encode(roll_states[0, 1:horizon+1])
                    z_pred = roll_eng.get_latents(roll_states, roll_actions, horizon)[0]
                    latent_div = round(float(torch.mean(torch.norm(z_pred - z_true, dim=-1)).item()), 6)

                g_label = config["g"] if not config["flag"] else "combined"
                l_label = config["l"] if not config["flag"] else "combined"    

                
                writer.writerow([
                    file.name, config["latent"], config["k"],
                    g_label, l_label, horizon, action,
                    round(roll_loss.item(), 6), valid_h,
                    round(float(rmse[0]), 6),
                    round(float(rmse[1]), 6),
                    round(float(rmse[2]), 6),
                    freq_error, latent_div
                ])
                csv_file.flush()

                labels = [r"$\cos\theta$", r"$\sin\theta$", r"$\dot\theta$"]
                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                for i, (ax, label) in enumerate(zip(axes, labels)):
                    ax.plot(true_np[:, i], label="True", linewidth=1.5)
                    ax.plot(pred_np[:, i], label="Pred",
                        linewidth=1.5, linestyle="--")
                    ax.set_ylabel(label)
                    ax.legend(loc="upper right")
                    ax.grid(True)
                axes[-1].set_xlabel("Step")
                fig.suptitle(
                    f"g={g_label} l={l_label} latent={config['latent']} "
                    f"k={config['k']} horizon={horizon} "
                    f"torque={action} loss={roll_loss.item():.4f}"
                )
                plt.tight_layout()

                if config["flag"]:
                    plot_dir = model_dir / f"g{g_label}_l{l_label}"
                else:
                    plot_dir = model_dir
                plot_dir.mkdir(parents=True, exist_ok=True)

                plot_path = plot_dir / f"horizon{horizon}_torque{action}.png"
                print(f"{plot_path}")
                plt.savefig(plot_path, dpi=80)
                plt.close()