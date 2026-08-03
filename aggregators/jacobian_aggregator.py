import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

FEEDFORWARD = {"dmd", "reg", "mlp"}
RECURRENT = {"gru", "rssm"}


def load_arch(csv_glob):
    dfs = []
    for f in glob.glob(csv_glob):
        df = pd.read_csv(f)
        fname = Path(f).name.lower()
        df["env"] = "cartpole" if "cartpole" in fname else "pendulum"
        df["policy"] = "sparse" if "sparse" in fname else "noise"
        dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="jacobian")
    ap.add_argument("--in_dist_only", action="store_true",
                    help="restrict to is_ood == False")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    arch_dfs = {}
    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        df = load_arch(f"{fold}/*.csv")
        if df is None:
            print(f"[skip empty] {fold}")
            continue
        if args.in_dist_only and "is_ood" in df.columns:
            df = df[df["is_ood"] == False]
        arch_dfs[arch] = df

    if not arch_dfs:
        print("No jacobian data found.")
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 25)


    print("\n" + "=" * 90)
    print("1. ACTION-INVARIANCE LADDER  (j_a_norm: ~0 = action-invariant)")
    print("=" * 90)

    print("\n  --- pooled over env/policy ---")
    ladder = {}
    for arch, df in arch_dfs.items():
        ladder[arch] = {
            "j_a_norm_mean": df["j_a_norm"].mean(),
            "j_a_norm_std": df["j_a_norm"].std(),
            "j_a_mean_mean": df["j_a_mean"].mean(),
            "j_a_max_mean": df["j_a_max"].mean(),
            "n": len(df),
        }
    ladder_df = pd.DataFrame(ladder).T.sort_values("j_a_norm_mean")
    print(ladder_df.round(5).to_string())

    print("\n  --- by env ---")
    rows = {}
    for arch, df in arch_dfs.items():
        for env in ["pendulum", "cartpole"]:
            sub = df[df["env"] == env]
            if len(sub) == 0:
                continue
            rows[f"{arch}/{env}"] = {
                "j_a_norm_mean": sub["j_a_norm"].mean(),
                "j_a_norm_std": sub["j_a_norm"].std(),
                "n": len(sub),
            }
    if rows:
        print(pd.DataFrame(rows).T.round(5).to_string())

    print("\n  --- by policy ---")
    rows = {}
    for arch, df in arch_dfs.items():
        for pol in ["noise", "sparse"]:
            sub = df[df["policy"] == pol]
            if len(sub) == 0:
                continue
            rows[f"{arch}/{pol}"] = {
                "j_a_norm_mean": sub["j_a_norm"].mean(),
                "j_a_norm_std": sub["j_a_norm"].std(),
                "n": len(sub),
            }
    if rows:
        print(pd.DataFrame(rows).T.round(5).to_string())

    print("\n" + "=" * 90)
    print("2. CONDITION NUMBER by latent_dim  (high = rank-deficient => null-space geometry)")
    print("=" * 90)
    for env in ["pendulum", "cartpole"]:
        cond_rows = {}
        for arch, df in arch_dfs.items():
            sub = df[df["env"] == env]
            if len(sub) == 0:
                continue
            cond_rows[arch] = sub.groupby("latent_dim")["condition_number"].mean()
        if not cond_rows:
            continue
        print(f"\n  --- {env} ---")
        print(pd.DataFrame(cond_rows).round(1).to_string())


    print("\n  --- min_singular by latent_dim ---")
    for env in ["pendulum", "cartpole"]:
        ms_rows = {}
        for arch, df in arch_dfs.items():
            sub = df[df["env"] == env]
            if len(sub) == 0:
                continue
            ms_rows[arch] = sub.groupby("latent_dim")["min_singular"].mean()
        if not ms_rows:
            continue
        print(f"\n    [{env}]")
        print(pd.DataFrame(ms_rows).round(5).to_string())

    if any("regime" in df.columns for df in arch_dfs.values()):
        print("\n  --- condition_number by regime (mean over latents) ---")
        for env in ["pendulum", "cartpole"]:
            rows = {}
            for arch, df in arch_dfs.items():
                sub = df[df["env"] == env]
                if len(sub) == 0 or "regime" not in sub.columns:
                    continue
                rows[arch] = sub.groupby("regime")["condition_number"].mean()
            if not rows:
                continue
            print(f"\n    [{env}]")
            print(pd.DataFrame(rows).round(1).to_string())
    print("\n" + "=" * 90)
    print("3. SPECTRAL STABILITY  (spectral_radius, unit_circle_frac, contracting/expanding)")
    print("=" * 90)
    for env in ["pendulum", "cartpole"]:
        stab = {}
        for arch, df in arch_dfs.items():
            sub = df[df["env"] == env]
            if len(sub) == 0:
                continue
            stab[arch] = {
                "spectral_radius": sub["spectral_radius"].mean(),
                "unit_circle_frac": sub["unit_circle_frac"].mean(),
                "contracting_frac": sub["contracting_frac"].mean(),
                "expanding_frac": sub["expanding_frac"].mean(),
                "n": len(sub),
            }
        if not stab:
            continue
        print(f"\n  --- {env} ---")
        print(pd.DataFrame(stab).T.round(4).to_string())


    print("\n" + "=" * 90)
    print("4. PHASE ERROR  (eigen-phase vs sqrt(g/l)*dt — FEEDFORWARD archs only)")
    print("=" * 90)
    for arch, df in arch_dfs.items():
        if arch in RECURRENT:
            print(f"  {arch}: [recurrent — h-phase not physically interpretable, skipped]")
            continue
        pe = df["phase_error"].dropna()
        if len(pe) == 0:
            print(f"  {arch}: no valid phase_error")
            continue
        print(f"  {arch}: phase_error mean={pe.mean():.4f} std={pe.std():.4f} "
              f"(expected_phase mean={df['expected_phase'].mean():.4f})")

    print("\n" + "=" * 90)
    print("MISSING ARCHITECTURES")
    print("=" * 90)
    have = set(arch_dfs.keys())
    expected = {"dmd", "reg", "gru", "rssm", "mlp"}
    missing = expected - have
    print(f"  present: {sorted(have)}")
    if missing:
        print(f"  MISSING: {sorted(missing)}  (add their jacobian_<arch>/ folder)")


if __name__ == "__main__":
    main()