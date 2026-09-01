import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

TAGS_FIG = ["in_training", "extrapolated"]   # interpolated dropped from figure
KS = [1, 15, 50]
ARCH_ORDER = ["mlp", "reg", "rssm"]        # <-- must match your folder suffixes
ARCH_LABEL = {"mlp": "MLP", "reg": "DMD-reg", "rssm": "RSSM"}
ARCH_COLOR = {"mlp": "#4C6EF5", "reg": "#2F9E44", "rssm": "#F76707"}

def parse_seed(checkpoint):
    name = str(checkpoint)
    m = re.search(r"_seed(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"WorldModel[A-Za-z]*_(\d+)_", name)
    if m: return int(m.group(1))
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
    df = df.drop_duplicates(subset=["checkpoint","eval_g","eval_l","env","policy","model_type"])
    return df

def fit_slope(true, meas):
    if len(true) < 3 or np.std(true) < 1e-9:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(true, meas, 1)
    r, _ = pearsonr(true, meas)
    return slope, intercept, r

def per_seed_slopes(sub):
    slopes = []
    for seed, g in sub.groupby("seed"):
        s, _, r = fit_slope(g["true_gl"].values, g["measured_gl"].values)
        if np.isfinite(s):
            slopes.append(s)
    return np.array(slopes)

def cell(df, k, tag, min_r2):
    sub = df[(df.model_type=="trained") & (df.k==k) & (df.config_tag==tag)
             & (df.r2 >= min_r2)]
    sl = per_seed_slopes(sub)
    if len(sl) == 0:
        return np.nan, np.nan
    return sl.mean(), sl.std()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="behavioural")
    ap.add_argument("--min_r2", type=float, default=0.0)
    ap.add_argument("--env", default="cartpole")
    ap.add_argument("--policy", default="noise")
    ap.add_argument("--out", default="fig_horizon.pdf")
    args = ap.parse_args()

    arch_dfs = {}
    for fold in sorted(glob.glob(f"{args.pattern}_*/")):
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        if arch == "seed":
            continue
        df = load_arch(f"{fold}/*.csv")
        if df is not None:
            arch_dfs[arch] = df[(df.env==args.env) & (df.policy==args.policy)].copy()
    if not arch_dfs:
        print("No behavioural data found."); return
    print("loaded arch keys:", list(arch_dfs))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    panel_title = {"in_training": "In-training", "extrapolated": "Extrapolated"}

    print(f"\n=== validation: {args.env}/{args.policy}, min_r2={args.min_r2} ===")
    for ax, tag in zip(axes, TAGS_FIG):
        for arch in ARCH_ORDER:
            if arch not in arch_dfs:
                continue
            df = arch_dfs[arch]
            ys, es = [], []
            for k in KS:
                m, s = cell(df, k, tag, args.min_r2)
                ys.append(m); es.append(s)
                print(f"{ARCH_LABEL.get(arch,arch):8s} {tag:13s} k={k:<2d} "
                      f"slope={m:.3f}±{0 if np.isnan(s) else s:.3f}")
            x = np.arange(len(KS))
            ax.errorbar(x, ys, yerr=es, marker="o", markersize=5, capsize=3,
                        color=ARCH_COLOR.get(arch, None), label=ARCH_LABEL.get(arch, arch),
                        lw=1.6)
        ax.axhline(1.0, ls="--", color="0.5", lw=1)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_xticks(np.arange(len(KS))); ax.set_xticklabels(KS)
        ax.set_xlabel(r"Training horizon $k$")
        ax.set_title(panel_title[tag], fontsize=10)
        ax.set_ylim(-0.15, 1.15)

    axes[0].set_ylabel("Readout slope")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    # ---- random-encoder null, for the caption ----
    print(f"\n=== random-encoder null: {args.env}/{args.policy} (all tags pooled) ===")
    null_slopes, null_rs = [], []
    for arch in ARCH_ORDER:
        if arch not in arch_dfs:
            continue
        df = arch_dfs[arch]
        for k in KS:
            sub = df[(df.model_type=="random") & (df.k==k) & (df.r2 >= args.min_r2)]
            sl, rs = [], []
            for seed, g in sub.groupby("seed"):
                s, _, r = fit_slope(g["true_gl"].values, g["measured_gl"].values)
                if np.isfinite(s):
                    sl.append(s); rs.append(r)
            if sl:
                print(f"{ARCH_LABEL.get(arch,arch):8s} k={k:<2d} "
                      f"null slope={np.mean(sl):+.3f}±{np.std(sl):.3f}  "
                      f"|r|={np.mean(np.abs(rs)):.3f}  (n_seeds={len(sl)})")
                null_slopes.extend(sl); null_rs.extend(np.abs(rs))
    if null_slopes:
        print(f"\nPOOLED NULL: slope={np.mean(null_slopes):+.3f}±{np.std(null_slopes):.3f}  "
              f"max|slope|={np.max(np.abs(null_slopes)):.3f}  "
              f"mean|r|={np.mean(null_rs):.3f}  max|r|={np.max(null_rs):.3f}")

if __name__ == "__main__":
    main()