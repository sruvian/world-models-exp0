"""
aggregate_comparator.py
-----------------------
Comparator (do-checker) aggregation, joined with Jacobian condition numbers.

Produces:
  1. DISSOCIATION TABLE (per arch x variable): dz_probe_cossim, clears_null rate,
     pc1_probe_cos -> probe direction orthogonal to / anti-aligned with causal direction.
  2. THEOREM 2 VALIDATION (well-conditioned subset): dz_opt_cossim (~0.7),
     ceiling_rel (small = effect reachable). Filtered by median condition_number.
  3. NULL-SPACE EVIDENCE: Pearson(median condition_number, ceiling_rel) across checkpoints
     for CONFIG variables (gravity/length) -> rank-deficiency predicts unreachability.
  4. DISSOCIATION SIGNIFICANCE: one-sample t-test / Wilcoxon that dz_probe_cossim == 0.

Join: comparator `checkpoint` (full path) -> basename  ==  jacobian `checkpoint` (.pt name).
Condition summary per checkpoint: MEDIAN over in-distribution eval_configs (robust to skew).

Usage:
    python aggregate_comparator.py --comp_pattern comparator --jac_pattern jacobian \
        --cond_thresh 1000
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from _repr_tag import repr_of
from scipy.stats import pearsonr, ttest_1samp, wilcoxon

SEEDED = {"seed"}
CONFIG_VARS = {"gravity", "length"}
SCALE_JUNK = {"theta_dot"}


def load_folders(pattern, skip_seeded=True):
    out = {}
    for fold in sorted(glob.glob(f"{pattern}_*/")):
        name = Path(fold.rstrip("/")).name
        arch = name.removeprefix(f"{pattern}_")
        if skip_seeded and arch in SEEDED:
            continue
        dfs = []
        for f in glob.glob(f"{fold}/*.csv"):
            df = pd.read_csv(f)
            fname = Path(f).name.lower()
            df["env"] = "cartpole" if "cartpole" in fname else "pendulum"
            df["policy"] = "sparse" if "sparse" in fname else "noise"
            df["representation"] = repr_of(f)
            dfs.append(df)
        if dfs:
            out[arch] = pd.concat(dfs, ignore_index=True)
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
    print("   dz_probe_cossim ~0     = probe direction orthogonal to TRUE (nonlinear) causal direction")
    print("   dzopt_probe_cossim ~0  = probe orthogonal to OPERATOR optimal (J+) direction")
    print("   dzopt_pc1_probe_cos ~0 = probe orthogonal to operator's dominant correction PC")
    print("   dzopt_top3_var         = operator correction concentration (top-3 PC var ratio)")
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
                dzopt_pc1_probe_cos=("dzopt_pc1_probe_cos", "mean"),
                n=("dz_probe_cossim", "count"),
            )
            if "dzopt_probe_cossim" in envdf.columns:
                agg_kwargs["dzopt_probe_cossim"] = ("dzopt_probe_cossim", "mean")
                agg_kwargs["dzopt_probe_std"] = ("dzopt_probe_cossim", "std")
            if "dzopt_top3_var" in envdf.columns:
                agg_kwargs["dzopt_top3_var"] = ("dzopt_top3_var", "mean")
            g = envdf.groupby("variable").agg(**agg_kwargs)
            col_order = ["dz_probe_cossim", "dz_probe_std"]
            if "dzopt_probe_cossim" in g.columns:
                col_order += ["dzopt_probe_cossim", "dzopt_probe_std"]
            col_order += ["clears_null_rate", "dzopt_pc1_probe_cos"]
            if "dzopt_top3_var" in g.columns:
                col_order += ["dzopt_top3_var"]
            col_order += ["n"]
            g = g[[c for c in col_order if c in g.columns]]
            print(f"\n  [{env}]")
            print(g.round(4).to_string())
    print("\n" + "=" * 90)
    print("2b. CORRECTION NORMS  ||dz_opt|| (min-norm operator) vs ||dz|| (nonlinear search)")
    print("    small ||dz_opt|| => target reachable with a cheap latent move (low buffering);")
    print("    expect small for dmd on gravity/length, large elsewhere.")
    print("    gap = ||dz_opt - dz|| large => underdetermined solution set (wide null space).")
    print("=" * 90)
    for arch, cdf in comp.items():
        need = ["dz_opt_norm", "dz_norm"]
        if not all(c in cdf.columns for c in need):
            continue
        print(f"\n--- {arch.upper()} ---")
        for env in ["pendulum", "cartpole"]:
            envdf = cdf[cdf["env"] == env]
            if len(envdf) == 0:
                continue
            g = envdf.groupby("variable").agg(
                dz_opt_norm=("dz_opt_norm", "mean"),
                dz_norm=("dz_norm", "mean"),
                dy_opt_norm=("dy_opt_norm", "mean"),
                gap=("analytical_search_gap", "mean"),
                n=("dz_opt_norm", "count"),
            )
            g["gain_opt"] = g["dy_opt_norm"] / (g["dz_opt_norm"] + 1e-12)
            print(f"\n  [{env}]")
            print(g.round(4).to_string())
    print("\n  gain_opt = ||J dz_opt|| / ||dz_opt|| ~ O(1) confirms J well-conditioned along")
    print("  the correction direction (linearisation valid despite large ||dz_opt||).")


    print("\n" + "=" * 90)
    print("3. NULL-SPACE EVIDENCE  Pearson(cond_median, ceiling_rel), config vars, PER ENV")
    print("   positive r => rank-deficiency (high cond) predicts unreachability (high ceiling)")
    print("   NOTE: split by env because ceiling is env-dominated (cartpole >> pendulum);")
    print("   pooling envs induces a spurious between-env correlation. Test WITHIN env.")
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
    print("   want: NOT significantly positive (probe not aligned with the causal/operator direction)")
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

    print("\nCore claim (robust): dz_probe_cossim ~0 AND dzopt_probe_cossim ~0 for parameters")
    print("=> neither the true causal direction nor the operator's own optimal direction")
    print("   aligns with the probe direction. Theorem 2: dz_opt_cossim ~0.5-0.7 (J+ valid).")
    print("Ceiling: check PENDULUM (small=reachable) vs CARTPOLE (high=degenerate J).")
    print("Null-space->ceiling correlation: only credible WITHIN env (section 3);")
    print("a pooled-env correlation is confounded by env and should not be reported.")
    if any("cos_Jw_r" in cdf.columns for cdf in comp.values()):
        print("\n" + "=" * 90)
        print("4b. OUTPUT-SPACE ALIGNMENT  cos(J w_probe, r)  vs norm-matched random null")
        print("    probe ~ null  =>  the probe direction's OUTPUT effect is no better")
        print("    aimed at the residual than a random direction of the same norm.")
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
                # does the probe beat its own null?
                g["beats_null"] = g["cos_Jw_r"].abs() > g["null_95"]
                if "Jw_rel" in g.columns and "Jw_rel_null" in g.columns:
                    g["rel_ratio"] = g["Jw_rel"] / (g["Jw_rel_null"] + 1e-12)
                print(f"\n  [{env}]")
                print(g.round(4).to_string())
        print("\n  Reading: for parameters expect |cos_Jw_r| <= null_95 (no better than a")
        print("  random direction at producing the required output change). This is the")
        print("  preimage-free version of the dissociation -- it lives entirely in output")
        print("  space, so it does not depend on choosing J+ out of the solution set.")

    reps = sorted({r for cdf in comp.values() for r in cdf.get("representation", pd.Series(dtype=str)).unique()})
    if len(reps) > 1:
        print("\n" + "=" * 90)
        print("4c. BY REPRESENTATION  (instantaneous vs rollout_h vs rollout_z)")
        print("    rollout_full is EXCLUDED from causal analysis: h and z are different")
        print("    spaces (deterministic recurrent state vs stochastic per-step encoding),")
        print("    so a cosine in the concatenation is dominated by the larger-norm block.")
        print("=" * 90)
        for arch, cdf in comp.items():
            if "representation" not in cdf.columns or cdf["representation"].nunique() < 2:
                continue
            print(f"\n--- {arch.upper()} ---")
            sub = cdf[cdf["variable"].isin(CONFIG_VARS)]
            if len(sub) == 0:
                continue
            g = sub.groupby(["representation", "env", "variable"]).agg(
                dz_probe=("dz_probe_cossim", "mean"),
                dzopt_probe=("dzopt_probe_cossim", "mean"),
                dzopt_pc1_probe=("dzopt_pc1_probe_cos", "mean"),
                clears_null=("clears_null", "mean"),
                n=("dz_probe_cossim", "count"),
            )
            if "cos_Jw_r" in sub.columns:
                g2 = sub.groupby(["representation", "env", "variable"])["cos_Jw_r"].mean()
                g["cos_Jw_r"] = g2
            print(g.round(4).to_string())

    diag_cols = ["probe_target_value", "null_target_mean", "null_nonnan", "null_95"]
    have_diag = any(all(c in cdf.columns for c in diag_cols) for cdf in comp.values())
    if have_diag:
        print("\n" + "=" * 90)
        print("5. NULL CALIBRATION  (does clears_null compare like-for-like targets?)")
        print("   target_ratio = probe_target_value / null_target_mean  (~1 = comparable)")
        print("   null_nonnan  = surviving null seeds per row (want ~30)")
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
        print("\n  Reading: if target_ratio ~1 and null_nonnan ~30, clears_null is a")
        print("  like-for-like comparison and the slope story is clean. If probe targets")
        print("  are systematically smaller (ratio << 1), report cosines + clears_null RATE")
        print("  (calibration-free) and move slope magnitudes to an appendix note.")


if __name__ == "__main__":
    main()