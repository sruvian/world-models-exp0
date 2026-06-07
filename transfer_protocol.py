import argparse
import yaml
import numpy as np
import torch
import os

from collector import collect_trajectories
from models.transfer import ProtocolBModel, ProtocolAModel
from models import WorldModel
from sim_envs import make_env
from trainer import split_gen, trainer
from models import make_model
from models.simplenn import SimpleNN
from logger import Logger

opts = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
losses = {"MSE": torch.nn.MSELoss}


def warm_start_encoder(pendulum_encoder: torch.nn.Module, hidden_dim: int, latent_angular: int) -> SimpleNN:
    unified_encoder = SimpleNN(5, hidden_dim, latent_angular)
    with torch.inference_mode():
        unified_encoder.net[0].weight.data[:, :3] = pendulum_encoder.net[0].weight.data
        unified_encoder.net[0].bias.data.copy_(pendulum_encoder.net[0].bias.data)
        unified_encoder.net[2].weight.data.copy_(pendulum_encoder.net[2].weight.data)
        unified_encoder.net[2].bias.data.copy_(pendulum_encoder.net[2].bias.data)
        unified_encoder.net[4].weight.data.copy_(pendulum_encoder.net[4].weight.data)
        unified_encoder.net[4].bias.data.copy_(pendulum_encoder.net[4].bias.data)
    return unified_encoder

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--yaml", type=str, required=True)
    parser = args.parse_args()

    with open(parser.yaml, 'r') as f:
        yaml_out = yaml.safe_load(f)

    protocol = yaml_out["protocol"]
    hyperparams_config = yaml_out["hyperparams"]
    trainer_config = yaml_out["trainer"]
    collector_config = yaml_out["collector"]
    transfer_config = yaml_out["transfer"]
    device = yaml_out["settings"]["device"]
    baseline = yaml_out.get("baseline", False)

    pendulum_model_config = yaml_out["pendulum_model"]
    pendulum_model_params = {k: v for k, v in pendulum_model_config.items()
                             if k not in ("name", "model_path")}
    pendulum_model = make_model(pendulum_model_config["name"], **pendulum_model_params)
    pendulum_model.load_state_dict(torch.load(pendulum_model_config["model_path"]))
    pendulum_model.eval()
    pendulum_model.requires_grad_(False)

    latent_angular = transfer_config["latent_angular"]
    latent_B = transfer_config["latent_B"]
    hidden_dim = transfer_config["hidden_dim"]
    action_dim = transfer_config["action_dim"]
    if baseline:
        if protocol == "B":
            angular_encoder = SimpleNN(3, hidden_dim, latent_angular)
            cartpole_encoder = SimpleNN(2, hidden_dim, latent_B)
            model = ProtocolBModel(
                angular_encoder=angular_encoder,
                cartpole_encoder=cartpole_encoder,
                latent_angular=latent_angular,
                latent_B=latent_B,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(device)
        elif protocol == "A":
            unified_encoder = SimpleNN(5, hidden_dim, latent_angular)
            model = ProtocolAModel(
                unified_encoder=unified_encoder,
                latent_angular=latent_angular,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(device)
    else:
        if protocol == "B":
            angular_encoder = pendulum_model.encoder
            angular_encoder.requires_grad_(False)
            cartpole_encoder = SimpleNN(2, hidden_dim, latent_B)
            model = ProtocolBModel(
                angular_encoder=angular_encoder,
                cartpole_encoder=cartpole_encoder,
                latent_angular=latent_angular,
                latent_B=latent_B,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(device)

        elif protocol == "A":
            unified_encoder = warm_start_encoder(pendulum_model.encoder, hidden_dim, latent_angular)
            model = ProtocolAModel(
                unified_encoder=unified_encoder,
                latent_angular=latent_angular,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(device)
        
    env_params = {k: v for k, v in yaml_out["environment"].items() if k not in ("name",)}
    environment = make_env("CartPoleSim", **env_params)
    states, actions, metadata = collect_trajectories(
        environment,
        collector_config["num_trajectories"],
        collector_config["episode_time"],
        collector_config["policy_seed"],
        collector_config["save"],
        collector_config["impulse_policy"]
    )

    train_s, train_s_next, train_a, val_s, val_s_next, val_a = split_gen(
        states, actions, hyperparams_config["rollout_steps"], device,
        windows_per_traj=collector_config.get("windows_per_traj", 10)
    )

    optimizer = opts[hyperparams_config["optimizer"]](
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hyperparams_config["lr"]
    )
    loss_func = losses[hyperparams_config["loss"]]()


    logger = Logger(
        f"Protocol{protocol}", hyperparams_config["optimizer"], hyperparams_config["loss"],
        hyperparams_config["lr"], trainer_config["batch_size"], trainer_config["steps"],
        yaml_out["environment"]["gravity"], yaml_out["environment"].get("length", 0.0),
        latent_angular
    )
    logger.start()

    trained_model = trainer(
        train_s, train_s_next, train_a,
        val_s, val_s_next, val_a,
        model, logger, optimizer, loss_func,
        trainer_config["batch_size"], trainer_config["steps"],
        hyperparams_config["rollout_decay"], hyperparams_config["gamma"],
        trainer_config["log_interval"]
    )
    logger.finish()


    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logfiles", f"protocol_{protocol.lower()}")
    os.makedirs(log_dir, exist_ok=True)
    config_tag = f"g{yaml_out['environment']['gravity']}_l{yaml_out['environment'].get('length', 0.0)}"
    log_path = os.path.join(log_dir,
        f"log_Protocol{protocol}_{config_tag}"
        f"_N{collector_config['num_trajectories']}"
        f"_k{hyperparams_config['rollout_steps']}"
        f"_latentA{latent_angular}_latentB{latent_B if protocol == 'B' else 0}"
        f"_steps{trainer_config['steps']}.npz"
    )
    logger.save(log_path, False, [])

    if yaml_out["checkpointing"]["save"]:
        save_path = os.path.join(
            yaml_out["checkpointing"]["save_path"],
            f"protocol_{protocol.lower()}"
        )
        os.makedirs(save_path, exist_ok=True)
        baseline_tag = "_baseline" if baseline else ""
        checkpoint_path = os.path.join(save_path,
            f"model_Protocol{protocol}{baseline_tag}_{config_tag}"
            f"_N{collector_config['num_trajectories']}"
            f"_k{hyperparams_config['rollout_steps']}"
            f"_latentA{latent_angular}_latentB{latent_B if protocol == 'B' else 0}"
            f"_steps{trainer_config['steps']}_test.pt"
        )
        torch.save(trained_model.state_dict(), checkpoint_path)
        print(f"[CHECKPOINT] Saved to {checkpoint_path}")