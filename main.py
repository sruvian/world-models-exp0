import numpy as np
import torch
from logger.logger import Logger
from sim_envs.envs import make_env
from models.model import make_model
from collector.collect import collect_trajectories
from trainer.trainer import split_gen, trainer
from rolloutEngine.rollout_engine import RolloutEngine

import yaml
import argparse


opts = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}


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

    env_params = {k: v for k, v in env_config.items() if k not in ("name", "run_env")}
    if env_config["run_env"]:
        print(f"[ENV] {env_config['name']} | g={env_config['gravity']} | l={env_config['pen_length']}")
        environment = make_env(env_config["name"], **env_params)
    if collector_config["run_collector"]:
        print(f"[COLLECTOR] {collector_config['num_trajectories']} trajectories x {collector_config['episode_time']} steps")
        collector = collect_trajectories(environment, collector_config["num_trajectories"], collector_config["episode_time"],
                                        collector_config["policy_seed"], collector_config["save"])
    
    if model_config["run_model"]:
        model_params = {k: v for k, v in model_config.items() 
                        if k not in ("name", "run_model")}
        model = make_model(model_config["name"], **model_params)

        if hyperparams_config["optimizer"] not in opts:
            raise ValueError("Supports Adam and SGD")
        opt = hyperparams_config["optimizer"]
        optimizer = opt(model.parameters(), hyperparams_config["lr"])

    

