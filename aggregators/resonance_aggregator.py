import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

WRAP_SWING = 300.0 
NUMERIC = ["true_omega0", "true_gl", "meas_gl", "meas_b", "meas_A",
           "onestep_r2", "w_steep", "swing", "amp_peak", "peakiness"]


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
    for c in NUMERIC:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fit_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < 1e-9:
        return np.nan, np.nan
    slope = np.polyfit(x, y, 1)[0]
    r = pearsonr(x, y)[0]
    return slope, r


def per_seed(sub, xcol, ycol):
    slopes, rs = [], []
    for _, g in sub.groupby("seed"):
        s, r = fit_slope(g[xcol].values, g[ycol].values)
        if np.isfinite(s):
            slopes.append(s); rs.append(r)
    return np.array(slopes), np.array(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="resonance")
    ap.add_argument("--min_r2", type=float, default=0.0,
                    help="drop joint-readout rows with onestep_r2 below this")
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
        print("No resonance data found."); return
    pd.set_option("display.width", 200)

    print("\n" + "=" * 92)
    print("JOINT ONE-STEP READOUT — g/l, b, A recovery (mean±SD across seeds)")
    print("=" * 92)
    for arch, df in arch_dfs.items():
        tr = df[df["model_type"] == "trained"]
        rd = df[df["model_type"] == "random"]
        for env in sorted(df["env"].unique()):
            for policy in ["noise", "sparse"]:
                sub = tr[(tr["env"] == env) & (tr["policy"] == policy)]
                if len(sub) == 0:
                    continue
                good = sub[sub["onestep_r2"] >= args.min_r2]
                if len(good) == 0:
                    continue
                slopes, rs = per_seed(good, "true_gl", "meas_gl")
                b_by_seed = good.groupby("seed")["meas_b"].mean()
                A_by_seed = good.groupby("seed")["meas_A"].mean()
                r2m = good["onestep_r2"].mean()
                rsub = rd[(rd["env"] == env) & (rd["policy"] == policy)
                          & (rd["onestep_r2"] >= args.min_r2)]
                rslopes, _ = per_seed(rsub, "true_gl", "meas_gl")

                print(f"\n  {arch.upper()} / {env} / {policy}:")
                if len(slopes) >= 3:
                    print(f"    g/l slope = {slopes.mean():.3f} ± {slopes.std():.3f}  "
                          f"r = {rs.mean():.3f}  (n_seeds={len(slopes)}, mean_R2={r2m:.3f})")
                else:
                    s, r = fit_slope(good["true_gl"].values, good["meas_gl"].values)
                    print(f"    g/l slope = {s:.3f} (pooled, n_seeds<3) r={r:.3f} mean_R2={r2m:.3f}")
                print(f"    b   = {b_by_seed.mean():.3f} ± {b_by_seed.std():.3f}   "
                      f"A = {A_by_seed.mean():.3f} ± {A_by_seed.std():.3f}")
                if len(rslopes) >= 1:
                    print(f"    random-null g/l slope = {rslopes.mean():.3f} ± {rslopes.std():.3f} "
                          f"(should be ~0)")

    print("\n" + "=" * 92)
    print("RESONANCE READOUT — does w_steep track √(g/l)?  (expect: NO)")
    print("  swing = trained-vs-random discriminator;  k=1 wrapping (swing>300°) filtered")
    print("=" * 92)
    for arch, df in arch_dfs.items():
        for env in sorted(df["env"].unique()):
            for policy in ["noise", "sparse"]:
                base = df[(df["env"] == env) & (df["policy"] == policy)]
                if len(base) == 0:
                    continue
                # report k=1 wrapping incidence before filtering
                k1 = base[base["k"] == 1]
                n_wrap = int((base["swing"] > WRAP_SWING).sum())
                clean = base[base["swing"] <= WRAP_SWING]

                tr = clean[clean["model_type"] == "trained"]
                rd = clean[clean["model_type"] == "random"]
                if len(tr) == 0:
                    continue

                slopes, rs = per_seed(tr, "true_omega0", "w_steep")
                tr_swing = tr.groupby("seed")["swing"].mean()
                rd_swing = rd.groupby("seed")["swing"].mean() if len(rd) else pd.Series([np.nan])

                print(f"\n  {arch.upper()} / {env} / {policy}:  "
                      f"(filtered {n_wrap} wrapping rows; {len(k1)} k=1 rows present)")
                if len(slopes) >= 3:
                    print(f"    w_steep vs √(g/l): slope={slopes.mean():.3f}±{slopes.std():.3f} "
                          f"r={rs.mean():.3f}  -> {'TRACKS' if abs(rs.mean())>0.8 else 'does NOT track'}")
                else:
                    s, r = fit_slope(tr["true_omega0"].values, tr["w_steep"].values)
                    print(f"    w_steep vs √(g/l): slope={s:.3f} r={r:.3f} (pooled)"
                          f"  -> {'TRACKS' if abs(r)>0.8 else 'does NOT track'}")
                print(f"    swing: trained={tr_swing.mean():.1f}°±{tr_swing.std():.1f}  "
                      f"random={rd_swing.mean():.1f}°  "
                      f"(gap = learned frequency structure)")


if __name__ == "__main__":
    main()