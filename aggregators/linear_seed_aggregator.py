import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

OBSERVABLES = {"cos_theta", "sin_theta", "theta_dot", "x", "x_dot",
               "cos_phase", "sin_phase"}


def parse_seed(name):
    name = str(name)
    m = re.search(r"_seed(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"WorldModel[A-Za-z]*_(\d+)_", name)
    if m: return int(m.group(1))
    return -1


def load(csv_glob):
    frames = []
    for f in glob.glob(csv_glob):
        df = pd.read_csv(f)
        fn = Path(f).name.lower()
        df["env"] = ("cartpole" if "cartpole" in fn
                     else "driven" if "driven" in fn else "pendulum")
        df["policy"] = "sparse" if "sparse" in fn else "noise"
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["seed"] = df["checkpoint"].map(parse_seed)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["is_random"] = df["label"].str.endswith("_random")
    df["target"] = df["label"].str.replace("_random$", "", regex=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="probe")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    arch_dfs = {}
    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        if arch == "seed":
            continue
        df = load(f"{fold}/*.csv")
        if df is not None:
            arch_dfs[arch] = df
    if not arch_dfs:
        print("No probe data found."); return
    pd.set_option("display.width", 200)

    for arch, df in arch_dfs.items():
        print("\n" + "=" * 90)
        print(f"{arch.upper()}  —  linear probe dissociation (R² mean±SD across seeds)")
        print("=" * 90)
        for env in sorted(df["env"].unique()):
            for policy in ["noise", "sparse"]:
                sub = df[(df["env"] == env) & (df["policy"] == policy)]
                if len(sub) == 0:
                    continue
                print(f"\n  --- {env} / {policy} ---")

                targets = sorted(sub["target"].unique())
                obs = [t for t in targets if t in OBSERVABLES]
                par = [t for t in targets if t not in OBSERVABLES]

                def stat(tgt, is_rand):
                    rows = sub[(sub["target"] == tgt) & (sub["is_random"] == is_rand)]
                    by_seed = rows.groupby("seed")["r2"].mean()
                    return by_seed.mean(), by_seed.std(), len(by_seed)

                print("    OBSERVABLES (decode high from both encoders):")
                for t in obs:
                    tm, ts, n = stat(t, False)
                    rm, rs, _ = stat(t, True)
                    print(f"      {t:14s}: trained R²={tm:.3f}±{ts:.3f}  random R²={rm:.3f}  (n_seeds={n})")

                print("    PARAMETERS (claim: floored, gain over random ≈ 0):")
                for t in par:
                    tm, ts, n = stat(t, False)
                    rm, rs, _ = stat(t, True)
                    gain = tm - rm
                    print(f"      {t:14s}: trained R²={tm:.3f}±{ts:.3f}  random R²={rm:.3f}  "
                          f"gain={gain:+.3f}  (n_seeds={n})")


if __name__ == "__main__":
    main()