import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from _repr_tag import repr_of
from scipy.stats import wilcoxon


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


def cell_means(df, env_filter=None, by_fraction=False):
    d = df if env_filter is None else df[df["env"] == env_filter]
    idx = ["env", "target_var", "top_k"] + (["latent_dim"] if by_fraction else [])
    piv = d.pivot_table(index=idx,
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
    raw_dfs = {}

    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        if "seed" in arch:
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
            raw_dfs[arch_label] = df

            piv_pend = cell_means(df, env_filter="pendulum")

            PARAM = {"gravity", "length", "g_over_l", "sqrt_l_over_g"}

            def split_wilcox(piv):
                if piv is None or len(piv) == 0:
                    return {"param": (0, 0, np.nan), "state": (0, 0, np.nan)}
                tv = piv.index.get_level_values("target_var")
                p_mask = tv.isin(PARAM)
                out = {}
                for name, m in [("param", p_mask), ("state", ~p_mask)]:
                    sub = piv[m]
                    if len(sub) == 0:
                        out[name] = (0, 0, np.nan)
                    else:
                        _, p, _ = wilcox_safe(sub["rd"])
                        out[name] = (int((sub["rd"] > 0).sum()), len(sub), p)
                return out

            all_split = split_wilcox(piv_all)
            pend_split = split_wilcox(piv_pend)

            rv_pos = int((piv_all["rv"] > 0).sum()); n_all = len(piv_all)
            _, rv_p, _ = wilcox_safe(piv_all["rv"])

            p_pos, p_n, p_p = all_split["param"]
            s_pos, s_n, s_p = all_split["state"]
            pp_pos, pp_n, pp_p = pend_split["param"]
            ps_pos, ps_n, ps_p = pend_split["state"]

            rows.append({
                "arch": arch_label,
                "n_param": p_n,
                "real_vs_randvals": verdict(rv_pos, n_all, rv_p),
                "param rd (all)": verdict(p_pos, p_n, p_p),
                "param rd (pend)": verdict(pp_pos, pp_n, pp_p),
                "state rd (pend)": verdict(ps_pos, ps_n, ps_p),
                "mean_rd_param_pend": round(float(piv_pend[piv_pend.index.get_level_values("target_var").isin(PARAM)]["rd"].mean()), 4) if piv_pend is not None and len(piv_pend) else np.nan,
                "mean_rd_state_pend": round(float(piv_pend[~piv_pend.index.get_level_values("target_var").isin(PARAM)]["rd"].mean()), 4) if piv_pend is not None and len(piv_pend) else np.nan,
                "param_p (pend)": round(pp_p, 4),
            })

    summary = pd.DataFrame(rows).set_index("arch")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print("THREE-WAY PATCHING DISSOCIATION — cross-architecture verdict")
    print("=" * 100)
    print(summary.to_string())
    print("\nReading:")
    print("  real_vs_randvals  '+ sig' everywhere -> interventions are effective (sanity).")
    print("  param rd (pend)   '~0'/'- sig'       -> THE CLAIM: probe dims NOT privileged")
    print("                                          for physical parameters.")
    print("  state rd (pend)   '+ sig'            -> POWER CONTROL: localisation DOES work")
    print("                                          for state variables, so the test has power.")
    print("  Splitting param from state is essential: pooled, the state signal drags rd")
    print("  positive and masks the parameter non-localisation.")

    print("\n" + "=" * 100)
    print("NON-LOCALISATION BY PATCHED FRACTION  (top_k / latent_dim), pendulum only")
    print("  rd = real - rand_dims; rd ~ 0 => probe dims NOT privileged over random dims.")
    print("  Load-bearing cell = SMALLEST fraction (large d, small k): rd~0 there = strong claim.")
    print("  Pendulum only (cartpole [:, :2] under-measures real-dims transport).")
    print("=" * 100)
    for arch_label, df in raw_dfs.items():
        pf = cell_means(df, env_filter="pendulum", by_fraction=True)
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
        print(f"\n--- {arch_label.upper()} ---")
        print(g.round(4).to_string())
        smallest = pf.nsmallest(min(5, len(pf)), "fraction")[
            ["target_var", "latent_dim", "top_k", "fraction", "real", "rand_dims", "rd"]]
        if len(smallest):
            print("  smallest-fraction cells (headline for strong-vs-weak):")
            print(smallest.round(4).to_string(index=False))


if __name__ == "__main__":
    main()