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
    if "env" not in df.columns:
        df["env"] = df["checkpoint"].apply(
            lambda c: "cartpole" if "cartpole" in str(c).lower() else "pendulum")
    return df


def cell_means(df, env_filter=None):
    d = df if env_filter is None else df[df["env"] == env_filter]
    piv = d.pivot_table(index=["env", "target_var", "top_k"],
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
        return (np.nan, np.nan, len(x))
    try:
        stat, p = wilcoxon(x)
        return (stat, p, len(x))
    except ValueError:
        return (np.nan, np.nan, len(x))


def verdict(pos, n, p):
    if np.isnan(p):
        return "n/a"
    frac = pos / n if n else float("nan")
    if p >= 0.05:
        return f"~0 (n.s., {pos}/{n})"
    return (f"+ sig ({pos}/{n}, p={p:.1e})" if frac > 0.5
            else f"- sig ({pos}/{n}, p={p:.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="activation",
                    help="folder prefix, e.g. 'activation' matches activation_dmd/ etc.")
    ap.add_argument("--include_seeded", action="store_true")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    rows = []

    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        if arch in SEEDED and not args.include_seeded:
            print(f"[skip seeded] {arch}")
            continue

        df_all = load_arch(f"{fold}/*.csv")
        if df_all is None:
            print(f"[skip empty] {fold}")
            continue

        reps = [r for r in df_all["representation"].unique() if r != "rollout_full"]
        for rep in sorted(reps):
            df = df_all[df_all["representation"] == rep]
            if len(df) == 0:
                continue
            arch_label = arch if rep == "instantaneous" else f"{arch}/{rep}"

            # ALL cells
            piv_all = cell_means(df)
            if piv_all is None:
                print(f"[skip malformed] {fold} [{rep}]")
                continue
            rv_pos = int((piv_all["rv"] > 0).sum()); n_all = len(piv_all)
            rd_pos = int((piv_all["rd"] > 0).sum())
            _, rv_p, _ = wilcox_safe(piv_all["rv"])
            _, rd_p, _ = wilcox_safe(piv_all["rd"])

            piv_pend = cell_means(df, env_filter="pendulum")
            if piv_pend is not None and len(piv_pend):
                rd_pos_p = int((piv_pend["rd"] > 0).sum()); n_pend = len(piv_pend)
                _, rd_p_pend, _ = wilcox_safe(piv_pend["rd"])
            else:
                rd_pos_p, n_pend, rd_p_pend = 0, 0, np.nan

            rows.append({
                "arch": arch_label,
                "n_cells": n_all,
                "real_vs_randvals": verdict(rv_pos, n_all, rv_p),
                "real_vs_randdims (all)": verdict(rd_pos, n_all, rd_p),
                "real_vs_randdims (pend)": verdict(rd_pos_p, n_pend, rd_p_pend),
                "mean_rv": round(float(piv_all["rv"].mean()), 4),
                "mean_rd_all": round(float(piv_all["rd"].mean()), 4),
                "mean_rd_pend": round(float(piv_pend["rd"].mean()), 4) if piv_pend is not None else np.nan,
            })

    summary = pd.DataFrame(rows).set_index("arch")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print("THREE-WAY PATCHING DISSOCIATION — cross-architecture verdict")
    print("=" * 100)
    print(summary.to_string())
    print("\nReading:")
    print("  real_vs_randvals  should be '+ sig' everywhere -> interventions are effective.")
    print("  real_vs_randdims  should be '~0' or '- sig'    -> probe dims never causally privileged.")
    print("  (pend) column removes the cartpole [:, :2] under-measurement -> defends the strong claim.")


if __name__ == "__main__":
    main()