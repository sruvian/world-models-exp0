import argparse
import glob
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from _repr_tag import repr_of
from scipy.stats import wilcoxon
 
SEEDED = {"seed"}
 
 
def load_arch(csv_glob):
    dfs = []
    for _f in glob.glob(csv_glob):
        _d = pd.read_csv(_f)
        _d["representation"] = repr_of(_f)
        dfs.append(_d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    if "base_dist" in df.columns and "baseline_err" not in df.columns:
        df = df.rename(columns={"base_dist": "baseline_err", "patch_dist": "patched_err"})
    keys = [c for c in ["checkpoint", "src_config", "tgt_config", "top_k", "patch_mode"]
            if c in df.columns]
    df = df.drop_duplicates(subset=keys)
    return df
 
 
def cell_means(df):
    """Per (target_var, top_k) mean shift per patch_mode -> dissociation contrasts.
    A 'cell' pools over config-pairs and models."""
    piv = df.pivot_table(index=["target_var", "top_k"],
                         columns="patch_mode", values="shift", aggfunc="mean")
    for m in ("real", "rand_dims", "rand_values"):
        if m not in piv.columns:
            return None
    piv = piv.dropna(subset=["real", "rand_dims", "rand_values"])
    piv["rv"] = piv["real"] - piv["rand_values"]
    piv["rd"] = piv["real"] - piv["rand_dims"]
    return piv
 
 
def wilcox_safe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.allclose(x, 0):
        return (np.nan, len(x))
    try:
        _, p = wilcoxon(x)
        return (p, len(x))
    except ValueError:
        return (np.nan, len(x))
 
 
def verdict(pos, n, p):
    if np.isnan(p):
        return "n/a"
    if p >= 0.05:
        return f"~0 (n.s., {pos}/{n})"
    frac = pos / n if n else float("nan")
    return (f"+ sig ({pos}/{n}, p={p:.1e})" if frac > 0.5
            else f"- sig ({pos}/{n}, p={p:.1e})")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="cross_config")
    args = ap.parse_args()
 
    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    rows = []
    detail = {}
 
    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        if arch in SEEDED:
            print(f"[skip seeded] {arch}")
            continue
        df = load_arch(f"{fold}/*.csv")
        if df is None:
            print(f"[skip empty] {fold}")
            continue
 
        piv = cell_means(df)
        if piv is None:
            print(f"[skip malformed] {fold}")
            continue
        detail[arch] = piv
 
        rv_pos = int((piv["rv"] > 0).sum()); n = len(piv)
        rd_pos = int((piv["rd"] > 0).sum())
        rv_p, _ = wilcox_safe(piv["rv"])
        rd_p, _ = wilcox_safe(piv["rd"])
 
        rows.append({
            "arch": arch,
            "n_cells": n,
            "real_vs_randvals": verdict(rv_pos, n, rv_p),
            "real_vs_randdims": verdict(rd_pos, n, rd_p),
            "mean_rv": round(float(piv["rv"].mean()), 4),
            "mean_rd": round(float(piv["rd"].mean()), 4),
        })
 
    if not rows:
        print("No cross-config data found.")
        return
 
    summary = pd.DataFrame(rows).set_index("arch")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 95)
    print("CROSS-CONFIG INTERCHANGE — three-way dissociation (config variables across configs)")
    print("=" * 95)
    print(summary.to_string())
    print("\nReading:")
    print("  real_vs_randvals '+ sig'  -> patching real structure transports the config effect.")
    print("  real_vs_randdims '~0'/'-' -> probe dims NOT causally privileged over random dims.")
    print("  (config variables gravity/length, patched SOURCE->TARGET across configs.)")
 
    print("\n" + "=" * 95)
    print("PER-VARIABLE (mean shift by patch_mode, pooled over config-pairs & top_k)")
    print("=" * 95)
    for arch, piv in detail.items():
        by_var = piv.groupby(level="target_var")[["real", "rand_dims", "rand_values", "rd", "rv"]].mean()
        print(f"\n--- {arch.upper()} ---")
        print(by_var.round(4).to_string())
 
 
if __name__ == "__main__":
    main()
