import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SEEDED = {"seed"}
TAG_ORDER = ["in_training", "interpolated", "extrapolated"]


def load_arch(csv_glob):
    frames = []
    for f in glob.glob(csv_glob):
        df = pd.read_csv(f)
        fname = Path(f).name.lower()
        df["env"] = "cartpole" if "cartpole" in fname else "pendulum"
        df["policy"] = "sparse" if "sparse" in fname else "noise"
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["checkpoint", "eval_g", "eval_l", "env", "policy"])
    return df


def fit_line(true, meas):
    if len(true) < 3 or np.std(true) < 1e-9:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(true, meas, 1)
    r, _ = pearsonr(true, meas)
    return slope, intercept, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="behavioural")
    ap.add_argument("--min_r2", type=float, default=0.0,
                    help="drop per-config estimates with fit R2 below this (unreliable)")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    arch_dfs = {}
    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        if "seed" in arch:
            continue
        df = load_arch(f"{fold}/*.csv")
        if df is not None:
            arch_dfs[arch] = df

    if not arch_dfs:
        print("No behavioural data found."); return

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

    RECURRENT = {"gru", "rssm"}
    for arch, df in arch_dfs.items():
        print("\n" + "=" * 85)
        print(f"{arch.upper()}  —  effective g/l tracking (measured vs true)")
        print("=" * 85)
        for env in ["pendulum", "cartpole"]:
            for policy in ["noise", "sparse"]:
                envdf = df[(df["env"] == env) & (df["policy"] == policy)]
                if len(envdf) == 0:
                    continue
                good = envdf[envdf["r2"] >= args.min_r2]
                dropped = len(envdf) - len(good)
                notes = []
                if env == "cartpole":
                    notes.append("NOT a true g/l -- cartpole isn't pure pendulum")
                if policy == "sparse" and arch in RECURRENT:
                    notes.append("TIMING-CONFOUNDED: period-scaled impulses can leak g/l "
                                 "to a recurrent model via action timing")
                note = ("   [" + "; ".join(notes) + "]") if notes else ""
                print(f"\n  --- {env} / {policy} ---{note}")
                if len(good) < 3:
                    print(f"    insufficient reliable points (n={len(good)}, dropped {dropped})")
                    continue
                slope, intercept, r = fit_line(good["true_coupling"].values, good["measured_coupling"].values)
                print(f"    OVERALL: slope={slope:.3f} intercept={intercept:+.3f} pearson_r={r:.3f}  "
                      f"(n={len(good)}, dropped {dropped} low-R2)")
                print(f"             mean|meas-true|={np.abs(good['measured_coupling']-good['true_coupling']).mean():.4f}  "
                      f"mean_fit_R2={good['r2'].mean():.3f}")
                print("    by config_tag:")
                for tag in TAG_ORDER:
                    sub = good[good["config_tag"] == tag]
                    if len(sub) == 0:
                        continue
                    s, i, rr = fit_line(sub["true_coupling"].values, sub["measured_coupling"].values)
                    err = np.abs(sub["measured_coupling"] - sub["true_coupling"]).mean()
                    print(f"      {tag:14s}: slope={s:.3f} r={rr:.3f} mean|err|={err:.4f} "
                          f"mean_R2={sub['r2'].mean():.3f} (n={len(sub)})")

    print("\n" + "=" * 85)
    print("CROSS-ARCHITECTURE  —  PENDULUM  TRAINED vs RANDOM (null control)")
    print("=" * 85)
    rows = {}
    for arch, df in arch_dfs.items():
        pend = df[df["env"] == "pendulum"].copy()
        pend["measured_coupling"] = pd.to_numeric(pend["measured_coupling"], errors="coerce")
        pend["r2"] = pd.to_numeric(pend["r2"], errors="coerce")
        has_type = "model_type" in pend.columns
        for mtype in (["trained", "random"] if has_type else ["trained"]):
            sub = pend[pend["model_type"] == mtype] if has_type else pend
            good = sub[(sub["r2"] >= args.min_r2) & np.isfinite(sub["measured_coupling"])]
            n_valid = len(good)
            n_total = len(sub)
            if n_valid < 3:
                rows[f"{arch}/{mtype}"] = {
                    "slope": np.nan, "r": np.nan, "mean_fit_r2": np.nan,
                    "n_valid": n_valid, "n_total": n_total,
                    "note": "no coherent g/l (null)" if mtype == "random" else "insufficient"}
                continue
            slope, _, r = fit_line(good["true_coupling"].values, good["measured_coupling"].values)
            rows[f"{arch}/{mtype}"] = {
                "slope": round(slope, 3), "r": round(r, 3),
                "mean_fit_r2": round(good["r2"].mean(), 3),
                "n_valid": n_valid, "n_total": n_total, "note": ""}
    print(pd.DataFrame(rows).T.to_string())


    print("\n[tracking data available per-config in the CSVs: true_coupling, measured_coupling, r2, config_tag]")


if __name__ == "__main__":
    main()