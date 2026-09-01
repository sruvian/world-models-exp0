import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

WRAP_SWING = 300.0 
NUMERIC = ["true_omega0", "true_gl", "meas_gl", "meas_b", "meas_A",
           "onestep_r2", "w_steep", "swing", "amp_peak", "peakiness"]
SWING_MIN = 8.0     
   
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
                print(f"\n  {arch.upper()} / {env} / {policy}:")
                for k_val in sorted(sub["k"].unique()):
                    ksub = sub[(sub["k"] == k_val) & (sub["onestep_r2"] >= args.min_r2)]
                    if len(ksub) == 0:
                        continue
                    # b, A reported once per k (not split by tag)
                    b_by_seed = ksub.groupby("seed")["meas_b"].mean()
                    A_by_seed = ksub.groupby("seed")["meas_A"].mean()
                    r2m = ksub["onestep_r2"].mean()
                    rsub = rd[(rd["env"] == env) & (rd["policy"] == policy)
                              & (rd["k"] == k_val) & (rd["onestep_r2"] >= args.min_r2)]
                    rslopes, _ = per_seed(rsub, "true_gl", "meas_gl")
                    # g/l split by tag
                    for tag in ["in_range", "held_out"]:
                        tsub = ksub[ksub["config_tag"] == tag]
                        if len(tsub) == 0:
                            continue
                        slopes, rs = per_seed(tsub, "true_gl", "meas_gl")
                        if len(slopes) >= 3:
                            print(f"  tag={tag:8s} k={k_val:2d}: g/l slope = {slopes.mean():.3f} ± {slopes.std():.3f}  "
                                  f"r={rs.mean():.3f}  (n_seeds={len(slopes)})")
                        else:
                            s, r = fit_slope(tsub["true_gl"].values, tsub["meas_gl"].values)
                            print(f"  tag={tag:8s} k={k_val:2d}: g/l slope={s:.3f} (pooled, n_seeds<3) r={r:.3f}")
                    print(f"    -> k={k_val:2d}: b={b_by_seed.mean():+.3f}±{b_by_seed.std():.3f}  "
                          f"A={A_by_seed.mean():.3f}±{A_by_seed.std():.3f}  R2={r2m:.3f}  "
                          f"rand_slope={rslopes.mean() if len(rslopes) else float('nan'):+.3f}")
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
                print(f"\n  {arch.upper()} / {env} / {policy}:")
                for k_val in sorted(base["k"].unique()):
                    kbase = base[base["k"] == k_val]
                    n_wrap = int((kbase["swing"] > WRAP_SWING).sum())
                    n_flat = int((kbase["swing"] < SWING_MIN).sum())
                    clean = kbase[(kbase["swing"] <= WRAP_SWING) & (kbase["swing"] >= SWING_MIN)]
                    tr = clean[clean["model_type"] == "trained"]
                    rd = clean[clean["model_type"] == "random"]
                    if len(tr) < 3:
                        print(f"    k={k_val:2d}: <3 trained rows with signal "
                              f"({n_wrap} wrapped, {n_flat} flat) — no readable resonance")
                        continue
                    # fit recovered w_steep against true natural frequency, per seed
                    slopes, intercepts, rs = [], [], []
                    for _, g in tr.groupby("seed"):
                        if g["true_omega0"].std() < 1e-9 or len(g) < 3:
                            continue
                        coef = np.polyfit(g["true_omega0"].values, g["w_steep"].values, 1)
                        slopes.append(coef[0]); intercepts.append(coef[1])
                        rs.append(pearsonr(g["true_omega0"].values, g["w_steep"].values)[0])
                    if len(slopes) < 3:
                        print(f"    k={k_val:2d}: <3 usable seeds — insufficient")
                        continue
                    slopes = np.array(slopes); intercepts = np.array(intercepts); rs = np.array(rs)
                    # TRACKS requires the recovered freq to SIT ON sqrt(g/l): slope~1 AND intercept~0
                    tracks = (abs(slopes.mean() - 1.0) < 0.25) and (abs(intercepts.mean()) < 0.25 * tr["true_omega0"].mean()) and (rs.mean() > 0.9)
                    verdict = "TRACKS" if tracks else "does NOT track"
                    tr_swing = tr.groupby("seed")["swing"].mean()
                    rd_swing = rd.groupby("seed")["swing"].mean() if len(rd) else pd.Series([np.nan])
                    print(f"    k={k_val:2d}: w_steep~√(g/l) slope={slopes.mean():.3f}±{slopes.std():.3f} "
                          f"intercept={intercepts.mean():+.3f} r={rs.mean():.3f} -> {verdict}  "
                          f"| swing tr={tr_swing.mean():.0f}° rd={rd_swing.mean():.0f}° "
                          f"({n_wrap} wrap, {n_flat} flat)")
                print(f"    swing: trained={tr_swing.mean():.1f}°±{tr_swing.std():.1f}  "
                      f"random={rd_swing.mean():.1f}°  "
                      f"(gap = learned frequency structure)")
    print("\n" + "=" * 92)
    print("DAMPING IDENTIFIABILITY — does meas_b approach true b as excitation rises?")
    print("  true b = 0.5;  corr(swing, |b_err|) < 0  =>  b recovers where damping term is better excited")
    print("=" * 92)
    TRUE_B = 0.5
    for arch, df in arch_dfs.items():
        for env in sorted(df["env"].unique()):
            for policy in ["noise", "sparse"]:
                base = df[(df["env"] == env) & (df["policy"] == policy)
                          & (df["model_type"] == "trained")
                          & (df["swing"] <= WRAP_SWING)
                          & (df["onestep_r2"] >= args.min_r2)].copy()
                if len(base) < 3:
                    continue
                base["b_err"] = (base["meas_b"] - TRUE_B).abs()
                c_err   = base["swing"].corr(base["b_err"])
                c_meas  = base["swing"].corr(base["meas_b"])
                print(f"\n  {arch.upper()} / {env} / {policy}:  "
                      f"corr(swing,|b_err|)={c_err:+.3f}  corr(swing,meas_b)={c_meas:+.3f}  "
                      f"mean|b_err|={base['b_err'].mean():.3f}  n={len(base)}")
                for k_val in sorted(base["k"].unique()):
                    kb = base[base["k"] == k_val]
                    if len(kb) < 3:
                        continue
                    ck = kb["swing"].corr(kb["b_err"])
                    print(f"    k={k_val:2d}: mean meas_b={kb['meas_b'].mean():+.3f}±{kb['meas_b'].std():.3f}  "
                          f"mean|b_err|={kb['b_err'].mean():.3f}  corr(swing,|b_err|)={ck:+.3f}  "
                          f"swing range=[{kb['swing'].min():.0f},{kb['swing'].max():.0f}]°  n={len(kb)}")


if __name__ == "__main__":
    main()