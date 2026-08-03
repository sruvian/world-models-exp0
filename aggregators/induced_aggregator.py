import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

PARAM_VARS = ["gravity", "length"]
STATE_VARS = ["theta_dot"]


def load_arch(csv_glob):
    dfs = []
    for f in glob.glob(csv_glob):
        df = pd.read_csv(f)
        df["env"] = "cartpole" if "cartpole" in Path(f).name.lower() else "pendulum"
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["checkpoint", "variable", "env"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="induced")
    args = ap.parse_args()

    arch_dfs = {}
    for fold in sorted(glob.glob(f"{args.pattern}_*/")):
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        df = load_arch(f"{fold}/*.csv")
        if df is not None:
            arch_dfs[arch] = df
    for f in glob.glob(f"{args.pattern}_*.csv"):
        arch = Path(f).stem.split("_")[-1]
        arch_dfs.setdefault(arch, pd.read_csv(f))

    if not arch_dfs:
        print("No induced-direction data found."); return
    pd.set_option("display.width", 200)

    print("=" * 88)
    print("INDUCED-DIRECTION DISSOCIATION   cos( w_probe , J_E S_x v )   (oracle-free)")
    print("=" * 88)

    for arch, df in arch_dfs.items():
        print(f"\n--- {arch.upper()} ---")
        for env in ["pendulum", "cartpole"]:
            envdf = df[df["env"] == env]
            if len(envdf) == 0:
                continue
            note = "" if env == "pendulum" else "   [g,l FIM is FULL RANK here: identifiable but test still valid]"
            print(f"\n  [{env}]{note}")
            g = envdf.groupby("variable").agg(
                cos_mean=("cos_mean", "mean"),
                abscos_mean=("abscos_mean", "mean"),
                cos_spread=("cos_mean", "std"),
                latent_dim=("latent_dim", "mean"),
                n=("cos_mean", "count"),
            )
            g["chance_abscos"] = np.sqrt(2.0 / (np.pi * g["latent_dim"]))
            order = [v for v in STATE_VARS + PARAM_VARS if v in g.index] + \
                    [v for v in g.index if v not in STATE_VARS + PARAM_VARS]
            g = g.reindex(order)
            print(g.round(4).to_string())
            for var in PARAM_VARS:
                sub = envdf[envdf["variable"] == var]
                a = sub["abscos_mean"].dropna().values
                if len(a) < 3:
                    continue
                d = float(sub["latent_dim"].mean())
                chance = np.sqrt(2.0 / (np.pi * d))
                t, p_two = ttest_1samp(a, chance)
                p_one = p_two / 2 if a.mean() < chance else 1 - p_two / 2
                ratio = a.mean() / chance
                verdict = ("BELOW chance" if (a.mean() < chance and p_one < 0.05)
                           else "at/above chance")
                signed = sub["cos_mean"].mean()
                print(f"      {var:8s}: |cos|={a.mean():.4f} ({ratio:.2f}x chance)  "
                      f"signed={signed:+.4f}  p_below={p_one:.3f} -> {verdict} (n={len(a)})")
            for var in STATE_VARS:
                sub = envdf[envdf["variable"] == var]
                a = sub["abscos_mean"].dropna().values
                if len(a) < 3:
                    continue
                d = float(sub["latent_dim"].mean())
                chance = np.sqrt(2.0 / (np.pi * d))
                ratio = a.mean() / chance
                print(f"      {var:8s}: |cos|={a.mean():.4f} ({ratio:.2f}x chance)  "
                      f"-> {'ABOVE chance (control OK)' if a.mean() > chance else 'FAILED control'} (n={len(a)})")


if __name__ == "__main__":
    main()