import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from _repr_tag import repr_of

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
    for c in ("trained_mi", "random_mi", "gain", "gain_sd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="smile")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    arch_dfs = {}
    for fold in folders:
        name = Path(fold.rstrip("/")).name
        arch = name.removeprefix(f"{args.pattern}_")
        if "seed" in arch:
            print(f"[skip seeded] {arch}")
            continue
        df = load(f"{fold}/*.csv")
        if df is None:
            print(f"[skip empty] {fold}")
            continue
        arch_dfs[arch] = df
    if not arch_dfs:
        print("No SMILE data found."); return
    pd.set_option("display.width", 200)

    for arch, df in arch_dfs.items():
        print("\n" + "=" * 88)
        print(f"{arch.upper()}  —  InfoNCE MI dissociation (mean±SD across seeds)")
        print("=" * 88)
        for env in sorted(df["env"].unique()):
            for policy in ["noise", "sparse"]:
                for rep in sorted(df["representation"].unique()):
                    sub = df[(df["env"] == env) & (df["policy"] == policy)
                             & (df["representation"] == rep)]
                    if len(sub) == 0:
                        continue
                    print(f"\n  --- {env} / {policy} / {rep} ---")

                    for k_val in sorted(sub["k"].unique()):
                        ksub = sub[sub["k"] == k_val]
                        params = ksub[ksub["is_control"] == 0]
                        ctrls = ksub[ksub["is_control"] == 1]
                        if len(params) == 0 and len(ctrls) == 0:
                            continue
                        print(f"\n    k={k_val}:")
                        print("      PARAMETERS (claim: gain over random ≈ 0):")
                        for tgt, g in params.groupby("target"):
                            gain_by_seed = g.groupby("seed")["gain"].mean()
                            tmi = g.groupby("seed")["trained_mi"].mean()
                            print(f"        {tgt:16s}: trained_MI={tmi.mean():.3f}  "
                                f"gain={gain_by_seed.mean():+.4f} ± {gain_by_seed.std():.4f} "
                                f"(n_seeds={len(gain_by_seed)})")
                        print("      OBSERVABLES (positive control — absolute MI high):")
                        for tgt, g in ctrls.groupby("target"):
                            tmi = g.groupby("seed")["trained_mi"].mean()
                            print(f"        {tgt:16s}: trained_MI={tmi.mean():.3f} ± {tmi.std():.3f} "
                                f"(n_seeds={len(tmi)})")
                        if len(params) and len(ctrls):
                            pgain = params["gain"].mean()
                            cmi = ctrls["trained_mi"].mean()
                            pmi = params["trained_mi"].mean()
                            print(f"      => obs MI ≈ {cmi:.2f} vs param MI ≈ {pmi:.2f} "
                                f"({cmi/max(pmi,1e-6):.0f}x gap); param gain ≈ {pgain:+.3f}")


if __name__ == "__main__":
    main()