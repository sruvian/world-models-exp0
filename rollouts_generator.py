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
from utils import parse_model



def generate_trajectories(env_confs, collector, action, env):
    if env == "CartPoleSim":
        base_config = {
            "gravity": 0.0, "mass1": 0.1, "mass2": 1.0,
            "length": 0.0, "dt": 0.01, "max_action": action,
            "damping": 0.0, "seed": 80,
        }
        env_name = "CartPoleSim"
    else:
        base_config = {
            "gravity": 0.0, "mass1": 1.0, "mass2": 0.0,
            "length": 0.0, "dt": 0.01, "max_action": action,
            "damping": 0.0, "seed": 80,
        }
        env_name = "PendulumSim"
    environment_config = base_config
    if env_confs is not None:
        environment_config["gravity"] = env_confs[0]
        environment_config["length"] = env_confs[1]
        environment = make_env(env_name,**environment_config)
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
            environment = make_env(env_name,**environment_config)
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
    
    policy_tag = "noise"
    if parser.impulse_policy:
        policy_tag = "sparse"
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
    

    env_tag = "cartpole" if "cartpole" in str(parser.models_dir).lower() else "pendulum"
    csv_path = Path(f"{top_dir}/{env_tag}_{policy_tag}_rollout_meta.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
       writer.writerow([
            "checkpoint", "latent_dim", "k", "gravity", "length",
            "horizon", "torque", "mse_loss", "valid_horizon",
            "rmse_cos", "rmse_sin", "rmse_thetadot", "rmse_x", "rmse_xdot",
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

        state_dim = 5 if config["env"] == "CartPoleSim" else 3
        model_params = {
            "state_dim": state_dim, "action_dim": 1,
            "hidden_dim": 64, "latent_dim": config["latent"]
}
        model = make_model("WorldModel", **model_params)
        model.load_state_dict(torch.load(file))
        model.eval()
        model_dir = top_dir / f"latent{config['latent']}" / \
                    f"k{config['k']}" / \
                    f"{config['config']}"
        model_dir.mkdir(parents=True, exist_ok=True)
        

        roll_eng = RolloutEngine(model, loss_func)
        if config["flag"]:
            env_confs = None
        else:
            env_confs = (config["g"], config["l"])
   
        for action in environment_actions:
            states, actions = generate_trajectories(env_confs, collector, action, config["env"])
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

                
                rmse_vals = [round(float(rmse[i]), 6) if i < len(rmse) else float('nan') for i in range(5)]
                writer.writerow([
                    file.name, config["latent"], config["k"],
                    g_label, l_label, horizon, action,
                    round(roll_loss.item(), 6), valid_h,
                    *rmse_vals,
                    freq_error, latent_div
                ])
                csv_file.flush()

                all_labels = [r"$\cos\theta$", r"$\sin\theta$", r"$\dot\theta$", r"$x$", r"$\dot{x}$"]
                labels = all_labels[:true_np.shape[1]]
                fig, axes = plt.subplots(len(labels), 1, figsize=(12, 3*len(labels)), sharex=True)
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
                plt.savefig(plot_path, dpi=60)
                plt.close()