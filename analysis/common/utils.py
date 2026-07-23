from pathlib import Path
import torch
import glob
from collections import defaultdict
import os
from models import make_model

def _is_float(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False

KNOWN_REGIMES = {"combined", "holdg", "holdl", "single"}

MODEL_NAMES = ("WorldModelDMD", "WorldModelGRU", "WorldModelRSSM", "WorldModelVAE", "Protocol A", "Protocol B")
TAGS = {"WorldModel": 'mlp', "WorldModelVAE": 'vae', "WorldModelDMD": 'dmd', "WorldModelGRU": 'gru', "WorldModelRSSM": 'rssm'}
def rollout_state(model, states, T_roll, device="cpu"):
    model.eval()
    with torch.no_grad():
        s = states.float().to(device)
        comp = model.encode_computational(s[:, 0, :])
        a = torch.zeros(s.shape[0], 1, device=device)
        for t in range(min(T_roll, s.shape[1] - 1)):
            comp = model.step_computational(comp, a)
            obs = model.encode_computational(s[:, t + 1, :])
            if isinstance(comp, tuple):
                comp = (comp[0], obs[1] if isinstance(obs, tuple) else obs)
        return comp
    
def parse_model(path: Path) -> dict:
    name = path.stem
    parts = name.split("_")
    result = {
        "flag": False, "g": 0.0, "l": 0.0,
        "latent": 0, "latent_B": 0, "k": 0, "steps": 0,
        "config": "", "env": "", "protocol": None, "baseline": False,
        "impulse": "impulse_policy" in str(path),
        "N": 0, "beta": None, "seed": 0, "regime": None,
        "model_name": "WorldModel",
    }
    if len(parts) > 2 and parts[2].isdigit():
        result["seed"] = int(parts[2])

    for mn in MODEL_NAMES:
        if mn in name:
            result["model_name"] = mn
            break

    if len(parts) > 1 and parts[1].startswith("Protocol"):
        result["protocol"] = parts[1][-1]
        idx = 2
        if parts[idx] == "baseline":
            result["baseline"] = True
            idx += 1
        result["g"] = float(parts[idx][1:])
        result["l"] = float(parts[idx + 1][1:])
        result["config"] = f"g{result['g']}_l{result['l']}"
        idx += 2
        for part in parts[idx:]:
            if part.startswith("N"): result["N"] = int(part[1:])
            elif part.startswith("k"): result["k"] = int(part[1:])
            elif part.startswith("latentA"): result["latent"] = int(part[7:])
            elif part.startswith("latentB"): result["latent_B"] = int(part[7:])
            elif part.startswith("steps"): result["steps"] = int(part[5:])
        result["env"] = "CartPoleSim" if "cartpole" in str(path).lower() else "PendulumSim"
        return result

    for part in parts:
        if part in KNOWN_REGIMES:
            result["regime"] = part
            result["config"] = part
            result["flag"] = (part == "combined" or part == "holdg" or part == "holdl")
        # elif part.startswith("seed") and part[4:].isdigit():
        #     result["seed"] = int(part[4:])
        elif part.startswith("k") and part[1:].isdigit():
            result["k"] = int(part[1:])
        elif part.startswith("latent") and part[6:].isdigit():
            result["latent"] = int(part[6:])
        elif part.startswith("steps") and part[5:].isdigit():
            result["steps"] = int(part[5:])
        elif part.startswith("beta") and _is_float(part[4:]):
            result["beta"] = float(part[4:])
        elif part.startswith("g") and _is_float(part[1:]):
            result["g"] = float(part[1:])
        elif part.startswith("l") and _is_float(part[1:]):
            result["l"] = float(part[1:])

    if result["regime"] is None and (result["g"] or result["l"]):
        result["config"] = f"g{result['g']}_l{result['l']}"

    result["env"] = "CartPoleSim" if "cartpole" in str(path).lower() else "PendulumSim"
    return result


def iter_model_groups(top_dir):
    groups = defaultdict(list)
    
    for path in glob.glob(os.path.join(top_dir, "**", "*.pt"), recursive=True):
        cfg = parse_model(Path(path))
        env_tag = "cartpole" if cfg["env"] == "CartPoleSim" else "pendulum"
        policy_tag = "sparse" if cfg["impulse"] else "noise"
        model_tag = TAGS[cfg["model_name"]]
        groups[(env_tag, policy_tag, model_tag)].append(path)
    return groups


def load_model(model_file, cfg, device="cpu"):
    is_cp = cfg["env"] == "CartPoleSim"
    model = make_model(cfg["model_name"], state_dim=5 if is_cp else 3, action_dim=1,
                       hidden_dim=HIDDEN_DIM.get(cfg["model_name"], 64), latent_dim=cfg["latent"])
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model

HIDDEN_DIM = {"WorldModelDMD": 128}