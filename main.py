import numpy as np
import torch
from logger.logger import Logger
from sim_envs.envs import make_env
from models.model import make_model
from collector.collect import collect_trajectories
from trainer.trainer import split_gen, trainer
from rolloutEngine.rollout_engine import RolloutEngine
import matplotlib.pyplot as plt
import glob
import os

import yaml
import argparse


opts = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
losses = {"MSE": torch.nn.MSELoss}


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--yaml", type = str, required = True)

    parser = args.parse_args()

    with open(parser.yaml, 'r') as f:
        yaml_out = yaml.safe_load(f)
    
    env_config = yaml_out["environment"]
    collector_config = yaml_out["collector"]
    model_config = yaml_out["model"]
    trainer_config = yaml_out["trainer"]
    rollout_config = yaml_out["rollout_engine"]
    hyperparams_config = yaml_out["hyperparams"]
    device = yaml_out["settings"]["device"]
    trained_model = None
    if yaml_out["datasets"]["use_existing"]:

        all_states, all_actions, all_metadata = [], [], []
        for pattern in yaml_out["datasets"]["paths"]:
            for path in glob.glob(pattern):
                    data = np.load(path)
                    all_states.append(data["states"])
                    all_actions.append(data["actions"])
                    all_metadata.append({k: data[k].item() for k in 
                                ["gravity", "length", "dt", "damping", 
                                "mass", "env_seed", "pol_seed"]})
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        
    else:
        all_metadata = []
        env_params = {k: v for k, v in env_config.items() if k not in ("name", "run_env")}
        print(f"[ENV] {env_config['name']} | g={env_config['gravity']} | l={env_config['pen_length']}")
        environment = make_env(env_config["name"], **env_params)
        print(f"[COLLECTOR] {collector_config['num_trajectories']} trajectories x {collector_config['episode_time']} steps")
        states, actions, metadata = collect_trajectories(environment, collector_config["num_trajectories"], collector_config["episode_time"],
                                        collector_config["policy_seed"], collector_config["save"])
        all_metadata.append(metadata)

    model_params = {k: v for k, v in model_config.items() 
                        if k not in ("name", "run_model")}

    if model_config["run_model"] and trainer_config["run_trainer"]:
        
        model = make_model(model_config["name"], **model_params)

        if hyperparams_config["optimizer"] not in opts:
            raise ValueError("Supports Adam and SGD")

        optimizer = opts[hyperparams_config["optimizer"]](model.parameters(), lr = hyperparams_config["lr"])
        loss_func = losses[hyperparams_config["loss"]]()

        train_s, train_s_next, train_a, val_s, val_s_next, val_a = split_gen(states, actions, hyperparams_config["rollout_steps"], device)

        logger = Logger(model_config["name"], hyperparams_config["optimizer"], hyperparams_config["loss"], 
                hyperparams_config["lr"], trainer_config["batch_size"], trainer_config["steps"],
                env_config["gravity"], env_config["pen_length"], model_config["latent_dim"])
        logger.start()
        trained_model = trainer(train_s, train_s_next, train_a, val_s, val_s_next, val_a, model, logger, optimizer, loss_func, trainer_config["batch_size"], trainer_config["steps"],
                                hyperparams_config["rollout_decay"], hyperparams_config["gamma"], trainer_config["log_interval"])
        logger.finish()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "logfiles")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, 
            f"log_{model_config['name']}_g{env_config['gravity']}_l{env_config['pen_length']}"
            f"_k{hyperparams_config['rollout_steps']}_{hyperparams_config['rollout_decay']}"
            f"_steps{trainer_config['steps']}_latent{model_config['latent_dim']}.npz")
        logger.save(log_path, yaml_out["datasets"]["use_existing"], yaml_out["datasets"]["paths"])

        if yaml_out["checkpointing"]["save"]:
            os.makedirs(yaml_out["checkpointing"]["save_path"], exist_ok=True)
            checkpoint_path = os.path.join(
                yaml_out["checkpointing"]["save_path"],
                f"model_{model_config['name']}_g{env_config['gravity']}_l{env_config['pen_length']}"
                f"_k{hyperparams_config['rollout_steps']}_{hyperparams_config['rollout_decay']}"
                f"_steps{trainer_config['steps']}_latent{model_config['latent_dim']}.pt"
            )
            torch.save(trained_model.state_dict(), checkpoint_path)
            print(f"[CHECKPOINT] Saved to {checkpoint_path}")

    if rollout_config["run_rollouts"]:
        if yaml_out["checkpointing"]["load"]:
            rollout_model = make_model(model_config["name"], **model_params)
            rollout_model.load_state_dict(torch.load(yaml_out["checkpointing"]["load_path"]))
            
        elif trained_model is not None:
            rollout_model = trained_model
        else:
            raise ValueError("Rollout requested but no model available — set load: true or run_trainer: true")

        env_params = {k: v for k, v in env_config.items() if k not in ("name", "run_env")}
        environment = make_env(env_config["name"], **env_params)
        roll_states, roll_actions, _ = collect_trajectories(
            environment, 1, collector_config["episode_time"],
            collector_config["policy_seed"] + 20, False
        )
        roll_states = torch.from_numpy(roll_states).float().to(device)   
        roll_actions = torch.from_numpy(roll_actions).float().to(device) 

        loss_func = losses[hyperparams_config["loss"]]()
        roll_eng = RolloutEngine(rollout_model, loss_func)
        horizon = rollout_config["horizon"]
        preds, roll_loss = roll_eng.rollout(roll_states, roll_actions, horizon)
        print(f"[ROLLOUT] horizon={horizon} | loss={roll_loss.item():.6f}")

        
        true_np  = roll_states[0, 1:horizon+1, :].cpu().numpy()  
        pred_np  = preds[0].cpu().numpy()                         
        labels   = [r"$\cos\theta$", r"$\sin\theta$", r"$\dot\theta$"]

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(true_np[:, i], label="True", linewidth=1.5)
            ax.plot(pred_np[:, i], label="Predicted", linewidth=1.5, linestyle="--")
            ax.set_ylabel(label)
            ax.legend(loc="upper right")
            ax.grid(True)
        axes[-1].set_xlabel("Rollout Step")
        fig.suptitle(
            f"Rollout | g={env_config['gravity']} l={env_config['pen_length']} "
            f"horizon={horizon} loss={roll_loss.item():.4f}"
        )
        plt.tight_layout()

        plot_path = os.path.join("rollout_plots",
            f"rollout_{model_config['name']}"
            f"_g{env_config['gravity']}_l{env_config['pen_length']}"
            f"_k{hyperparams_config['rollout_steps']}_{hyperparams_config['rollout_decay']}"
            f"_h{horizon}_latent{model_config['latent_dim']}"
            f"_steps{trainer_config['steps']}.png")
        plt.savefig(plot_path)
        print(f"[ROLLOUT] Plot saved to {plot_path}")
        plt.show()