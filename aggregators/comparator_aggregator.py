import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _repr_tag import repr_of
from scipy.stats import pearsonr, ttest_1samp, wilcoxon

CONFIG_VARS = {"gravity", "length"}
SCALE_JUNK = {"theta_dot"}


def load_folders(pattern, skip_seeded=True):

    out = {}
    for fold in sorted(glob.glob(f"{pattern}_*/")):
        fname_full = Path(fold.rstrip("/")).name.lower()
        if skip_seeded and re.search(r"(^|_)seed(_|$)", fname_full):
            continue
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        fold_rep = repr_of(fold)
        arch_label = arch if fold_rep == "instantaneous" else f"{arch}/{fold_rep}"
        dfs = []
        for f in glob.glob(f"{fold}/*.csv"):
            df = pd.read_csv(f)
            fname = Path(f).name.lower()
            df["env"] = "cartpole" if "cartpole" in fname else "pendulum"
            df["policy"] = "sparse" if "sparse" in fname else "noise"
            df["representation"] = fold_rep
            dfs.append(df)
        if dfs:
            out[arch_label] = pd.concat(dfs, ignore_index=True)
    return out


def jac_condition_per_checkpoint(jac_df):
    d = jac_df
    if "is_ood" in d.columns:
        d = d[d["is_ood"] == False]
    med = d.groupby("checkpoint")["condition_number"].median()
    iqr = d.groupby("checkpoint")["condition_number"].agg(
        lambda x: x.quantile(0.75) - x.quantile(0.25))
    return med, iqr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_pattern", default="comparator")
    ap.add_argument("--jac_pattern", default="jacobian")
    ap.add_argument("--cond_thresh", type=float, default=1000.0,
                    help="condition_number below this = 'well-conditioned' (Theorem 2 subset)")
    args = ap.parse_args()

    comp = load_folders(args.comp_pattern)
    jac = load_folders(args.jac_pattern)
    if not comp:
        print("No comparator data."); return

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 25)

    for arch, cdf in comp.items():
        cdf["ckpt_name"] = cdf["checkpoint"].apply(lambda c: Path(c).name)
        if arch in jac:
            med, iqr = jac_condition_per_checkpoint(jac[arch])
            cdf["cond_median"] = cdf["ckpt_name"].map(med)
            cdf["cond_iqr"] = cdf["ckpt_name"].map(iqr)
        else:
            cdf["cond_median"] = np.nan
            cdf["cond_iqr"] = np.nan
        comp[arch] = cdf

    print("\n" + "=" * 90)
    print("1. DISSOCIATION")

    print("=" * 90)
    for arch, cdf in comp.items():
        print(f"\n--- {arch.upper()} ---")
        for env in ["pendulum", "cartpole"]:
            envdf = cdf[cdf["env"] == env]
            if len(envdf) == 0:
                continue
            agg_kwargs = dict(
                dz_probe_cossim=("dz_probe_cossim", "mean"),
                dz_probe_std=("dz_probe_cossim", "std"),
                clears_null_rate=("clears_null", "mean"),
                n=("dz_probe_cossim", "count"),
            )
            pc1col = "dzopt_pc1_probe_cos" if "dzopt_pc1_probe_cos" in envdf.columns else \
                     ("pc1_probe_cos" if "pc1_probe_cos" in envdf.columns else None)
            if pc1col:
                agg_kwargs["pc1_probe_cos"] = (pc1col, "mean")
            if "dzopt_probe_cossim" in envdf.columns:
                agg_kwargs["dzopt_probe_cossim"] = ("dzopt_probe_cossim", "mean")
                agg_kwargs["dzopt_probe_std"] = ("dzopt_probe_cossim", "std")
            g = envdf.groupby("variable").agg(**agg_kwargs)
            col_order = ["dz_probe_cossim", "dz_probe_std"]
            if "dzopt_probe_cossim" in g.columns:
                col_order += ["dzopt_probe_cossim", "dzopt_probe_std"]
            col_order += ["clears_null_rate", "pc1_probe_cos", "n"]
            g = g[[c for c in col_order if c in g.columns]]
            print(f"\n  [{env}]")
            print(g.round(4).to_string())
    print("\n" + "=" * 90)
    print(f"2. THEOREM 2 VALIDATION  (well-conditioned: cond_median < {args.cond_thresh})")
    print("=" * 90)
    for arch, cdf in comp.items():
        wc = cdf[cdf["cond_median"] < args.cond_thresh]
        if len(wc) == 0:
            print(f"\n--- {arch.upper()} --- (no well-conditioned rows)")
            continue
        g = wc.groupby(["env", "variable"]).agg(
            dz_opt_cossim=("dz_opt_cossim", "mean"),
            ceiling_mean=("ceiling_err", "mean"),
            ceiling_std=("ceiling_err", "std"),
            ceiling_median=("ceiling_err", "median"),
            n=("dz_opt_cossim", "count"),
        )
        for env_v in g.index:
            if env_v[1] in SCALE_JUNK:
                g.loc[env_v, ["ceiling_mean", "ceiling_std", "ceiling_median"]] = np.nan
        print(f"\n--- {arch.upper()} ---")
        print(g.round(4).to_string())


    print("\n" + "=" * 90)
    print("3. NULL-SPACE EVIDENCE  Pearson(cond_median, ceiling_rel), config vars, PER ENV")

    print("=" * 90)
    for env in ["pendulum", "cartpole"]:
        print(f"\n  --- {env} ---")
        pooled_env = []
        for arch, cdf in comp.items():
            sub = cdf[(cdf["variable"].isin(CONFIG_VARS)) & (cdf["env"] == env)]
            sub = sub.dropna(subset=["cond_median", "ceiling_err"])
            sub = sub[np.isfinite(sub["cond_median"]) & np.isfinite(sub["ceiling_err"])]
            if len(sub) >= 3 and sub["ceiling_err"].std() > 1e-9 and sub["cond_median"].std() > 1e-9:
                r, p = pearsonr(np.log10(sub["cond_median"] + 1), sub["ceiling_err"])
                print(f"    {arch:6s} (log-cond): r={r:+.3f} (p={p:.2e})  n={len(sub)}  "
                      f"[ceiling {sub['ceiling_err'].mean():.3f}+/-{sub['ceiling_err'].std():.3f}]")
            else:
                print(f"    {arch:6s}: insufficient/degenerate (n={len(sub)})")
            pooled_env.append(sub[["cond_median", "ceiling_err"]])
        allsub = pd.concat(pooled_env, ignore_index=True) if pooled_env else pd.DataFrame()
        if len(allsub) >= 3 and allsub["ceiling_err"].std() > 1e-9:
            r, p = pearsonr(np.log10(allsub["cond_median"] + 1), allsub["ceiling_err"])
            print(f"    POOLED-{env} (log-cond): r={r:+.3f} (p={p:.2e})  n={len(allsub)}")

    print("\n" + "=" * 90)
    print("4. DISSOCIATION SIGNIFICANCE  (one-sample test vs 0)")
    print("=" * 90)
    cos_cols = ["dz_probe_cossim"]
    if any("dzopt_probe_cossim" in cdf.columns for cdf in comp.values()):
        cos_cols.append("dzopt_probe_cossim")
    for col in cos_cols:
        label = "probe vs TRUE-causal" if col == "dz_probe_cossim" else "probe vs OPERATOR-optimal (J+)"
        print(f"\n  [{col}]  ({label})")
        for arch, cdf in comp.items():
            if col not in cdf.columns:
                continue
            for env in ["pendulum", "cartpole"]:
                envdf = cdf[cdf["env"] == env]
                if len(envdf) == 0:
                    continue
                for scope, vars_ in [("config", CONFIG_VARS), ("all", None)]:
                    sub = envdf if vars_ is None else envdf[envdf["variable"].isin(vars_)]
                    x = sub[col].dropna().values
                    x = x[np.isfinite(x)]
                    if len(x) < 3:
                        continue
                    t, p = ttest_1samp(x, 0.0)
                    direction = "positive" if x.mean() > 0 else "negative"
                    verdict = ("~0 (n.s.)" if p >= 0.05 else f"{direction} sig")
                    print(f"    {arch:6s} [{env:8s}][{scope:6s}]: mean={x.mean():+.4f}  "
                          f"t={t:+.2f} p={p:.3f}  -> {verdict}  (n={len(x)})")



    if any("cos_Jw_r" in cdf.columns for cdf in comp.values()):
        print("\n" + "=" * 90)
        print("4b. OUTPUT-SPACE ALIGNMENT  cos(J w_probe, r)  vs norm-matched random null")
        print("=" * 90)
        for arch, cdf in comp.items():
            if "cos_Jw_r" not in cdf.columns:
                continue
            print(f"\n--- {arch.upper()} ---")
            for env in ["pendulum", "cartpole"]:
                envdf = cdf[cdf["env"] == env]
                if len(envdf) == 0:
                    continue
                agg = dict(
                    cos_Jw_r=("cos_Jw_r", "mean"),
                    null_mean=("cos_Jw_r_null_mean", "mean"),
                    null_95=("cos_Jw_r_null_95", "mean"),
                    n=("cos_Jw_r", "count"),
                )
                if "Jw_rel_norm" in envdf.columns:
                    agg["Jw_rel"] = ("Jw_rel_norm", "mean")
                if "Jw_rel_null_mean" in envdf.columns:
                    agg["Jw_rel_null"] = ("Jw_rel_null_mean", "mean")
                g = envdf.groupby("variable").agg(**agg)
                g["beats_null"] = g["cos_Jw_r"].abs() > g["null_95"]
                if "Jw_rel" in g.columns and "Jw_rel_null" in g.columns:
                    g["rel_ratio"] = g["Jw_rel"] / (g["Jw_rel_null"] + 1e-12)
                print(f"\n  [{env}]")
                print(g.round(4).to_string())

    geom_cols = ["probe_slope", "survival", "dz_opt_norm", "dz_norm",
                 "dzopt_top3_var", "analytical_search_gap"]
    if any(c in cdf.columns for cdf in comp.values() for c in geom_cols):
        print("\n" + "=" * 90)
        print("4d. TRANSPORT & OPERATOR GEOMETRY  (config variables)")
        print("=" * 90)
        for arch, cdf in comp.items():
            sub = cdf[cdf["variable"].isin(CONFIG_VARS)]
            if len(sub) == 0:
                continue
            print(f"\n--- {arch.upper()} ---")
            for env in ["pendulum", "cartpole"]:
                edf = sub[sub["env"] == env]
                if len(edf) == 0:
                    continue
                agg = {}
                for c, nm in [("probe_slope", "probe_slope"), ("survival", "survival"),
                              ("dz_opt_norm", "dz_opt_norm"), ("dz_norm", "dz_norm"),
                              ("dzopt_top3_var", "dzopt_top3_var"),
                              ("analytical_search_gap", "asearch_gap")]:
                    if c in edf.columns:
                        agg[nm] = (c, "mean")
                if not agg:
                    continue
                g = edf.groupby("variable").agg(**agg)
                print(f"\n  [{env}]")
                print(g.round(4).to_string())

    SUPP_ROWS = ["gravity", "length", "theta_dot"]
    supp_map = [
        ("asearch_gap", "analytical_search_gap", "mean"),
        ("dz_opt_norm", "dz_opt_norm", "mean"),
        ("dz_norm", "dz_norm", "mean"),
        ("dzopt_top3_var", "dzopt_top3_var", "mean"),
        ("probe_slope", "probe_slope", "mean"),
        ("survival", "survival", "mean"),
        ("clears_null", "clears_null", "mean"),
        ("cos_Jw_r", "cos_Jw_r", "mean"),
        ("cos_Jw_r_null95", "cos_Jw_r_null_95", "mean"),
        ("Jw_rel", "Jw_rel_norm", "mean"),
        ("Jw_rel_null", "Jw_rel_null_mean", "mean"),
    ]
    print("\n" + "=" * 100)
    print("4e. CONSOLIDATED SUPPLEMENT TABLE  (columns Table 4 does not show)")
    print("    rows: gravity, length (config) + theta_dot (state control)")
    print("=" * 100)
    for arch, cdf in comp.items():
        sub = cdf[cdf["variable"].isin(SUPP_ROWS)]
        if len(sub) == 0:
            continue
        print(f"\n--- {arch.upper()} ---")
        for env in ["pendulum", "cartpole"]:
            edf = sub[sub["env"] == env]
            if len(edf) == 0:
                continue
            agg = {out: (col, how) for out, col, how in supp_map if col in edf.columns}
            if not agg:
                continue
            g = edf.groupby("variable").agg(**agg)
            g = g.reindex([v for v in SUPP_ROWS if v in g.index])
            if "cos_Jw_r" in g.columns and "cos_Jw_r_null95" in g.columns:
                g["Jw_beats_null"] = g["cos_Jw_r"].abs() > g["cos_Jw_r_null95"]
            if "Jw_rel" in g.columns and "Jw_rel_null" in g.columns:
                g["Jw_rel_ratio"] = (g["Jw_rel"] / (g["Jw_rel_null"] + 1e-12)).round(3)
            print(f"\n  [{env}]")
            print(g.round(4).to_string())
    diag_cols = ["probe_target_value", "null_target_mean", "null_nonnan", "null_95"]
    have_diag = any(all(c in cdf.columns for c in diag_cols) for cdf in comp.values())
    if have_diag:
        print("\n" + "=" * 90)
        print("5. NULL CALIBRATION  (does clears_null compare like-for-like targets?)")
        print("=" * 90)
        for arch, cdf in comp.items():
            if not all(c in cdf.columns for c in diag_cols):
                continue
            print(f"\n--- {arch.upper()} ---")
            for env in ["pendulum", "cartpole"]:
                envdf = cdf[cdf["env"] == env]
                if len(envdf) == 0:
                    continue
                g = envdf.groupby("variable").agg(
                    probe_target=("probe_target_value", "mean"),
                    null_target=("null_target_mean", "mean"),
                    null_nonnan=("null_nonnan", "mean"),
                    clears_null=("clears_null", "mean"),
                    n=("probe_target_value", "count"),
                )
                g["target_ratio"] = g["probe_target"] / (g["null_target"].abs() + 1e-12)
                g = g[["probe_target", "null_target", "target_ratio",
                       "null_nonnan", "clears_null", "n"]]
                print(f"\n  [{env}]")
                print(g.round(4).to_string())


if __name__ == "__main__":
    main()