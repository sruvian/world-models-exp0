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


def cell_means(df, by_fraction=False):

    idx = ["target_var", "top_k"] + (["latent_dim"] if by_fraction else [])
    piv = df.pivot_table(index=idx,
                         columns="patch_mode", values="shift", aggfunc="mean")
    for m in ("real", "rand_dims", "rand_values"):
        if m not in piv.columns:
            return None
    piv = piv.dropna(subset=["real", "rand_dims", "rand_values"])
    piv["rv"] = piv["real"] - piv["rand_values"]
    piv["rd"] = piv["real"] - piv["rand_dims"]
    if by_fraction:
        piv = piv.reset_index()
        piv["fraction"] = piv["top_k"] / piv["latent_dim"]
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
    ap.add_argument("--pattern", default="cross_cfg")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    rows = []
    detail = {}
    raw_dfs = {}

    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]

        _rep = repr_of(fold)
        arch = arch if _rep == "instantaneous" else f"{arch}/{_rep}"
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
        raw_dfs[arch] = df


        PARAM = {"gravity", "length", "g_over_l", "sqrt_l_over_g"}
        tv = piv.index.get_level_values("target_var")
        state_present = (~tv.isin(PARAM)).any()
        if state_present:
            print(f"  [note] {arch}: non-parameter target_vars present in cross-config "
                  f"({sorted(set(tv[~tv.isin(PARAM)]))}); restricting rd test to parameters.")
        piv_param = piv[tv.isin(PARAM)]
        if len(piv_param) == 0:
            piv_param = piv
        rv_pos = int((piv["rv"] > 0).sum()); n = len(piv)
        rv_p, _ = wilcox_safe(piv["rv"])
        rd_pos = int((piv_param["rd"] > 0).sum()); n_param = len(piv_param)
        rd_p, _ = wilcox_safe(piv_param["rd"])

        rows.append({
            "arch": arch,
            "n_param": n_param,
            "real_vs_randvals": verdict(rv_pos, n, rv_p),
            "real_vs_randdims": verdict(rd_pos, n_param, rd_p),
            "mean_rv": round(float(piv["rv"].mean()), 4),
            "mean_rd": round(float(piv_param["rd"].mean()), 4),
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
    print("\n" + "=" * 95)
    print("NON-LOCALISATION BY PATCHED FRACTION  (top_k / latent_dim)")
    print("=" * 95)
    for arch, df in raw_dfs.items():
        pf = cell_means(df, by_fraction=True)
        if pf is None or len(pf) == 0:
            continue
        pf = pf.copy()
        pf["frac_bin"] = pd.cut(pf["fraction"],
                                bins=[0, 0.10, 0.25, 0.50, 1.01],
                                labels=["<=10%", "10-25%", "25-50%", ">50%"])
        g = pf.groupby("frac_bin", observed=True).agg(
            rd_mean=("rd", "mean"),
            rd_abs=("rd", lambda x: np.abs(x).mean()),
            real_mean=("real", "mean"),
            randdims_mean=("rand_dims", "mean"),
            n=("rd", "count"),
        )
        print(f"\n--- {arch.upper()} ---")
        print(g.round(4).to_string())

        PARAM = {"gravity", "length", "g_over_l", "sqrt_l_over_g"}
        small_bin = pf[(pf["fraction"] <= 0.10) & (pf["target_var"].isin(PARAM))]
        if len(small_bin) >= 3:
            p, nn = wilcox_safe(small_bin["rd"].values)
            pos = int((small_bin["rd"] > 0).sum())
            print(f"  <=10% bin, params: rd Wilcoxon -> {verdict(pos, len(small_bin), p)} "
                  f"(mean rd={small_bin['rd'].mean():+.4f})  [STRONG-CLAIM TEST]")
        else:
            print(f"  <=10% bin, params: n={len(small_bin)} cells -- too few for Wilcoxon; "
                  f"report means only and say 'consistent across fractions tested'")
        smallest = pf.nsmallest(min(5, len(pf)), "fraction")[
            ["target_var", "latent_dim", "top_k", "fraction", "real", "rand_dims", "rd"]]
        if len(smallest):
            print(f"  smallest-fraction cells (headline for strong-vs-weak):")
            print(smallest.round(4).to_string(index=False))

    print("\n" + "=" * 95)
    print("PER-VARIABLE (mean shift by patch_mode, pooled over config-pairs & top_k)")
    print("=" * 95)
    for arch, piv in detail.items():
        by_var = piv.groupby(level="target_var")[["real", "rand_dims", "rand_values", "rd", "rv"]].mean()
        print(f"\n--- {arch.upper()} ---")
        print(by_var.round(4).to_string())


if __name__ == "__main__":
    main()