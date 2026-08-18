import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from _repr_tag import repr_of

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
        df["representation"] = repr_of(f)
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["seed"] = df["checkpoint"].map(parse_seed)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["is_random"] = df["target"].astype(str).str.endswith("_random")
    df["target"] = df["target"].astype(str).str.replace(r"_random$", "", regex=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="linear_probe")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    arch_dfs = {}
    for fold in folders:
        name = Path(fold.rstrip("/")).name
        arch = name.removeprefix(f"{args.pattern}_")
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
                for rep in sorted(df["representation"].unique()):
                    sub = df[(df["env"] == env) & (df["policy"] == policy)
                             & (df["representation"] == rep)]
                    if len(sub) == 0:
                        continue
                    print(f"\n  --- {env} / {policy} / {rep} ---")

                    targets = sorted(sub["target"].unique())
                    obs = [t for t in targets if t in OBSERVABLES]
                    par = [t for t in targets if t not in OBSERVABLES]

                    for k_val in sorted(sub["k"].unique()):
                        ksub = sub[sub["k"] == k_val]
                        print(f"\n    k={k_val}:")
                        def stat(tgt, is_rand):
                            rows = ksub[(ksub["target"] == tgt) & (ksub["is_random"] == is_rand)]
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