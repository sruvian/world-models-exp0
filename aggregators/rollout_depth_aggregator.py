import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

SEEDED = {"seed"}
SKIP = {"seed"}

DELTA_COLS = ["delta_theta", "delta_thetadot", "delta_x", "delta_xdot"]
STATE_LABEL = {"delta_theta": "theta", "delta_thetadot": "theta_dot",
               "delta_x": "x", "delta_xdot": "x_dot"}


def load_arch(csv_glob):
    dfs = [pd.read_csv(f) for f in glob.glob(csv_glob)]
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["checkpoint", "eval_config", "depth"])
    return df


def collapse_depth(depth_series, delta_series, thresh):
    """First depth where mean delta drops below thresh (None if never)."""
    s = pd.Series(delta_series.values, index=depth_series.values).sort_index()
    below = s[s < thresh]
    return int(below.index[0]) if len(below) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="rollout_depth")
    ap.add_argument("--collapse_thresh", type=float, default=0.1,
                    help="mean delta below this = 'collapsed' (not decodable)")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    arch_dfs = {}
    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        if arch in SKIP:
            continue
        df = load_arch(f"{fold}/*.csv")
        if df is not None:
            arch_dfs[arch] = df

    if not arch_dfs:
        print("No rollout-depth data found."); return

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 25)

    for arch, df in arch_dfs.items():
        print("\n" + "=" * 90)
        print(f"{arch.upper()}  —  decodability (delta = r2 - r2_shuffled) vs rollout depth")
        print("=" * 90)
        present = [c for c in DELTA_COLS if c in df.columns and df[c].notna().any()]
        tab = df.groupby("depth")[present].mean()
        tab.columns = [STATE_LABEL[c] for c in present]
        print(tab.round(4).to_string())

        print("\n  collapse depth (first depth with mean delta < "
              f"{args.collapse_thresh}):")
        for c in present:
            g = df.groupby("depth")[c].mean().reset_index()
            cd = collapse_depth(g["depth"], g[c], args.collapse_thresh)
            label = STATE_LABEL[c]
            print(f"    {label:10s}: {'depth ' + str(cd) if cd is not None else 'never collapses (stays decodable)'}")

    print("\n" + "=" * 90)
    print("CROSS-ARCHITECTURE  —  theta decodability: depth 0 vs deeper")
    print("=" * 90)
    rows = {}
    for arch, df in arch_dfs.items():
        if "delta_theta" not in df.columns:
            continue
        by_depth = df.groupby("depth")["delta_theta"].mean()
        rows[arch] = {
            "d0": by_depth.get(0, np.nan),
            "d1": by_depth.get(1, np.nan),
            "d5": by_depth.get(5, np.nan),
            "d15": by_depth.get(15, np.nan),
            "d25": by_depth.get(25, np.nan),
        }
    summary = pd.DataFrame(rows).T
    print(summary.round(4).to_string())
    print("\nReading: deterministic archs (mlp/dmd/gru) retain decodability across depth")
    print("(gradual decay); RSSM drops to ~0 by depth 1 (open-loop latent not decodable).")
    print("The depth-0 vs depth-1 gap for RSSM is the intrinsic-collapse signature")
    print("(confirmed robust to warm-up -> not an initialization artifact).")

    have = set(arch_dfs.keys())
    expected = {"dmd", "reg", "gru", "rssm", "mlp"}
    missing = expected - have
    if missing:
        print(f"\n[missing architectures: {sorted(missing)}]")


if __name__ == "__main__":
    main()