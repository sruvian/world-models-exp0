from pathlib import Path
import os
import csv
import glob
import argparse
import matplotlib.pyplot as plt
import pandas

from models import make_model
from rolloutEngine import RolloutEngine
from sim_envs.envs import make_env

from collector import collect_trajectories


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    group = args.add_argument_group()
    group.add_argument("--models_dir", type=Path)
    group.add_argument("--model_pt", type = Path)
    parser = args.parse_args()
    
    top_dir = "rollout_plots"

    pt_files: list[Path] = []
    if parser.models_dir:
        if os.path.exists(parser.models_dir):
            pt_files = glob.glob(os.path.join(parser.models_dir))
        else:
            raise ValueError("Path does not exist")
    
    if parser.model_pt:
        if os.path.exists(parser.model_pt):
            pt_files.append(parser.model_pt)
        else:
            raise ValueError("Path does not exist")
    
    environment_torques = [0, 0.5, 5, 10, 15, 20, 30, 50]
    horizons = [50, 500, 5000]
    for file in pt_files:
        file_name = file.name[:-3]
        file_components = file_name.split("_")[2:]

        model_config = {"state_dim" : 3, "action_dim": 1, "hidden_dim": 64, "latent_dim": int(file_name[7:])}
        model = make_model("WorldModel", **model_config)
        model_dir = os.path.join(top_dir, file_name)
        environment_config = {
            "gravity": float(file_components[0][1:]),
            "pen_mass": 1.0,
            "pen_length": float(file_components[1][1:]),
            "dt": 0.01,
            "max_torque": 0,
            "damping": 0.0,
            "seed": 80,

        }
        collector = {
            "num_trajectories": 500,
            "episode_time": 100000,
            "policy_seed": 35,
            "save": False
            }
        for torque in environment_torques:
            environment_config["max_torque"] = torque
            environment = make_env("PendulumSim",**environment_config)
            collect_trajectories(environment, **collector)
            for horizon in horizons:
                pass
        

