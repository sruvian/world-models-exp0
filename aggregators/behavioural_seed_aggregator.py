import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

TAG_ORDER = ["in_training", "interpolated", "extrapolated"]
RECURRENT = {"gru", "rssm"}
CARTPOLE_INERTIA = 1.555


def parse_seed(checkpoint):
    name = str(checkpoint)
    m = re.search(r"_seed(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"WorldModel[A-Za-z]*_(\d+)_", name)
    if m:
        return int(m.group(1))
    return -1


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
    df["seed"] = df["checkpoint"].map(parse_seed)
    for c in ("true_gl", "measured_gl", "r2"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["checkpoint", "eval_g", "eval_l", "env", "policy", "model_type"])
    return df


def fit_slope(true, meas):
    if len(true) < 3 or np.std(true) < 1e-9:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(true, meas, 1)
    r, _ = pearsonr(true, meas)
    return slope, intercept, r


def per_seed_slopes(sub):
    slopes, rs = [], []
    for seed, g in sub.groupby("seed"):
        s, _, r = fit_slope(g["true_gl"].values, g["measured_gl"].values)
        if np.isfinite(s):
            slopes.append(s); rs.append(r)
    return np.array(slopes), np.array(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="behavioural")
    ap.add_argument("--min_r2", type=float, default=0.0)
    ap.add_argument("--cartpole_correct", action="store_true",
                    help="divide cartpole measured_gl by the effective-inertia factor")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    arch_dfs = {}
    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        if arch == "seed":
            continue
        df = load_arch(f"{fold}/*.csv")
        if df is not None:
            arch_dfs[arch] = df
    if not arch_dfs:
        print("No behavioural data found."); return

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

    for arch, df in arch_dfs.items():
        print("\n" + "=" * 90)
        print(f"{arch.upper()}  —  effective g/l tracking (slope mean±SD across seeds)")
        print("=" * 90)
        for env in ["pendulum", "cartpole"]:
            for policy in ["noise", "sparse"]:
                envdf = df[(df["env"] == env) & (df["policy"] == policy)
                           & (df["model_type"] == "trained")].copy()
                if len(envdf) == 0:
                    continue
                if env == "cartpole" and args.cartpole_correct:
                    envdf["measured_gl"] = envdf["measured_gl"] / CARTPOLE_INERTIA

                good = envdf[envdf["r2"] >= args.min_r2]
                dropped = len(envdf) - len(good)
                notes = []
                if env == "cartpole":
                    notes.append("effective coeff (×2C∈[1.5,1.61] inertia factor)"
                                 + (" [corrected]" if args.cartpole_correct else " [RAW]"))
                if policy == "sparse" and arch in RECURRENT:
                    notes.append("TIMING-CONFOUNDED (recurrent + period-scaled impulse)")
                note = ("   [" + "; ".join(notes) + "]") if notes else ""
                print(f"\n  --- {env} / {policy} ---{note}")

                slopes, rs = per_seed_slopes(good)
                if len(slopes) < 3:
                    print(f"    insufficient seeds with reliable fits (n_seeds={len(slopes)}, dropped {dropped} low-R2)")
                    # still show a pooled fallback
                    s, i, r = fit_slope(good["true_gl"].values, good["measured_gl"].values)
                    if np.isfinite(s):
                        print(f"    pooled fallback: slope={s:.3f} r={r:.3f} (n_pts={len(good)})")
                    continue
                print(f"    slope = {slopes.mean():.3f} ± {slopes.std():.3f}   "
                      f"pearson_r = {rs.mean():.3f} ± {rs.std():.3f}   "
                      f"(n_seeds={len(slopes)}, mean_fit_R2={good['r2'].mean():.3f}, dropped {dropped})")
                print(f"    mean|meas-true| = {np.abs(good['measured_gl']-good['true_gl']).mean():.4f}")

                print("    by config_tag:")
                for tag in TAG_ORDER:
                    sub = good[good["config_tag"] == tag]
                    if len(sub) == 0:
                        continue
                    tslopes, trs = per_seed_slopes(sub)
                    err = np.abs(sub["measured_gl"] - sub["true_gl"]).mean()
                    if len(tslopes) >= 3:
                        print(f"      {tag:14s}: slope={tslopes.mean():.3f}±{tslopes.std():.3f} "
                              f"r={trs.mean():.3f} mean|err|={err:.4f} (n_seeds={len(tslopes)})")
                    else:
                        s, i, r = fit_slope(sub["true_gl"].values, sub["measured_gl"].values)
                        print(f"      {tag:14s}: slope={s:.3f} (pooled) r={r:.3f} "
                              f"mean|err|={err:.4f} (n_pts={len(sub)})")

    print("\n" + "=" * 90)
    print("TRAINED vs RANDOM  (null control) — slope mean±SD across seeds, pendulum")
    print("=" * 90)
    rows = {}
    for arch, df in arch_dfs.items():
        pend = df[df["env"] == "pendulum"]
        for mtype in ["trained", "random"]:
            sub = pend[(pend["model_type"] == mtype) & (pend["r2"] >= args.min_r2)]
            slopes, rs = per_seed_slopes(sub)
            if len(slopes) < 3:
                rows[f"{arch}/{mtype}"] = {
                    "slope_mean": np.nan, "slope_sd": np.nan, "r_mean": np.nan,
                    "n_seeds": len(slopes),
                    "note": "no coherent g/l (null)" if mtype == "random" else "insufficient"}
            else:
                rows[f"{arch}/{mtype}"] = {
                    "slope_mean": round(slopes.mean(), 3), "slope_sd": round(slopes.std(), 3),
                    "r_mean": round(rs.mean(), 3), "n_seeds": len(slopes), "note": ""}
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()