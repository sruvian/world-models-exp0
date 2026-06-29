import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from models.wmodel import WorldModel, WorldModelVAE
from models import make_model


def make_hook(acts, name):
    def fn(module, inp, out):
        acts[name] = out.detach()
    return fn

def collect_activations(model: WorldModel| WorldModelVAE, states: torch.Tensor, actions: torch.Tensor):
    spec = model.layer_spec()
    acts = {}
    handles = [m.register_forward_hook(make_hook(acts, name)) for name, m in spec.items()]

    with torch.no_grad():
        z = model.encode(states)
        t = model.step(z, actions)
        _ = model.decode(t)
    
    for handle in handles:
        handle.remove()
    
    acts["latent"] = z.detach()
    return acts, model.layer_timesteps()

def multilayer_probe(acts, timesteps, current_target, next_target,
                     n_seeds=20, alpha=10.0):
    results = {}
    for layer_name, activation in acts.items():
        X = activation.numpy() if hasattr(activation, "numpy") else activation
        y = current_target if timesteps.get(layer_name, "current") == "current" else next_target
    
        r2s, baseline_r2s = [], []
        for seed in range(n_seeds):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)

            r2s.append(Ridge(alpha=alpha).fit(Xtr, ytr).score(Xte, yte))

            y_shuf = np.random.default_rng(seed).permutation(ytr)
            baseline_r2s.append(Ridge(alpha=alpha).fit(Xtr, y_shuf).score(Xte, yte))
        
        results[layer_name] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "baseline_mean": float(np.mean(baseline_r2s)),
            "above_baseline": float(np.mean(r2s) - np.mean(baseline_r2s)),
            "coef": Ridge(alpha=alpha).fit(X, y).coef_,
        }
        mi_mean, mi_std = layer_mi(activation, y, n_seeds=5)
        results[layer_name]["mi_mean"] = mi_mean
        results[layer_name]["mi_std"] = mi_std
    return results

def layer_mi(activation, target, n_seeds=5):
    X = activation.numpy() if hasattr(activation, "numpy") else activation
    mis = []
    for seed in range(n_seeds):
        mi = mutual_info_regression(X, target, random_state=seed)
        mis.append(mi.sum())
    return float(np.mean(mis)), float(np.std(mis))


if __name__ == "__main__":
    pass