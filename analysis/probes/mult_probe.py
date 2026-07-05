import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from analysis.common import parse_model, iter_model_groups, load_model, prepare_probe_data, probeable_vars, get_data


def layer_mi(activation, target, n_seeds=5):
    X = activation.numpy() if hasattr(activation, "numpy") else activation
    mis, mis_shuf = [], []
    for seed in range(n_seeds):
        mis.append(mutual_info_regression(X, target, random_state=seed).mean())
        y_shuf = np.random.default_rng(seed).permutation(target)
        mis_shuf.append(mutual_info_regression(X, y_shuf, random_state=seed).mean())
    return float(np.mean(mis)), float(np.std(mis)), float(np.mean(mis_shuf)), float(np.std(mis_shuf))


def multilayer_probe(acts, timesteps, current_target, next_target, config_labels,
                     n_seeds=20, alpha=10.0):
    results = {}
    for layer_name, activation in acts.items():
        X = activation.numpy() if hasattr(activation, "numpy") else activation
        y = current_target if timesteps.get(layer_name, "current") == "current" else next_target
        r2s, base_r2s = [], []
        for seed in range(n_seeds):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=config_labels)
            r2s.append(Ridge(alpha=alpha).fit(Xtr, ytr).score(Xte, yte))
            y_shuf = np.random.default_rng(seed).permutation(ytr)
            base_r2s.append(Ridge(alpha=alpha).fit(Xtr, y_shuf).score(Xte, yte))
        mi_m, mi_s, mi_sh_m, mi_sh_s = layer_mi(activation, y, n_seeds=5)
        results[layer_name] = {
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
            "baseline_mean": float(np.mean(base_r2s)),
            "above_baseline": float(np.mean(r2s) - np.mean(base_r2s)),
            "mi_mean": mi_m, "mi_std": mi_s, "mi_shuf_mean": mi_sh_m, "mi_shuf_std": mi_sh_s,
        }
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    for (env_tag, policy_tag, model_tag), files in iter_model_groups(args.models_dir).items():
        csv_path = Path(f"probe_results/multilayer_probe_{env_tag}_{policy_tag}_{model_tag}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        csv_file = open(csv_path, "a", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow([
                "checkpoint", "model_config", "variable", "layer", "latent_dim", "k",
                "r2_mean", "r2_std", "baseline_mean", "above_baseline",
                "mi_mean", "mi_std", "mi_shuf_mean", "mi_shuf_std",
            ])

        for mf in files:
            cfg = parse_model(Path(mf))
            print(f"\n[{Path(mf).name}]")
            is_cp = cfg["env"] == "CartPoleSim"
            model = load_model(mf, cfg, args.device)

            states, gravities, lengths = get_data(cfg)
            acts, timesteps, cur_t, nxt_t, config_labels = prepare_probe_data(
                model, states, gravities, lengths, cfg["regime"], is_cartpole=is_cp)

            for variable in probeable_vars(cfg["regime"], is_cp):
                res = multilayer_probe(acts, timesteps, cur_t[variable], nxt_t[variable],
                                       config_labels, alpha=args.alpha)
                for layer_name, m in res.items():
                    writer.writerow([
                        Path(mf).name, cfg["config"], variable, layer_name,
                        cfg["latent"], cfg["k"],
                        round(m["r2_mean"], 4), round(m["r2_std"], 4),
                        round(m["baseline_mean"], 4), round(m["above_baseline"], 4),
                        round(m["mi_mean"], 4), round(m["mi_std"], 4),
                        round(m["mi_shuf_mean"], 4), round(m["mi_shuf_std"], 4),
                    ])
                    csv_file.flush()
        csv_file.close()
        print(f"[{env_tag}_{policy_tag}_{model_tag}] done -> {csv_path}")