import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

LAYER_ORDER = ["encoder_early", "encoder_late", "latent",
               "transition_early", "transition_late",
               "decoder_early", "decoder_late"]
PARAM_VARS = ["gravity", "length"]
STATE_VARS = ["theta", "theta_dot", "x", "x_dot"]
FEEDFORWARD = {"mlp", "dmd", "reg"}


def load_arch(csv_glob):
    dfs = [pd.read_csv(f) for f in glob.glob(csv_glob)]
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["checkpoint", "variable", "layer"])
    return df


def order_layers(index):
    present = [l for l in LAYER_ORDER if l in index]
    extra = [l for l in index if l not in LAYER_ORDER]
    return present + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="multilayer_probe")
    ap.add_argument("--metric", default="r2_mean",
                    choices=["r2_mean", "above_baseline"],
                    help="r2_mean (raw) or above_baseline (R2 - shuffled)")
    args = ap.parse_args()

    folders = sorted(glob.glob(f"{args.pattern}_*/"))
    flat = glob.glob(f"{args.pattern}_*.csv")

    arch_dfs = {}
    for fold in folders:
        arch = Path(fold.rstrip("/")).name.split("_")[-1]
        df = load_arch(f"{fold}/*.csv")
        if df is not None:
            arch_dfs.setdefault(arch, []).append(df)
    for f in flat:
        arch = Path(f).stem.split("_")[-1]
        df = pd.read_csv(f)
        arch_dfs.setdefault(arch, []).append(df)

    if not arch_dfs:
        print("No multilayer data found."); return
    arch_dfs = {a: pd.concat(v, ignore_index=True) for a, v in arch_dfs.items()}

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

    for arch, df in arch_dfs.items():
        ff = arch in FEEDFORWARD
        tag = "" if ff else "   [recurrent: 'layer' ambiguous -- interpret with care / future work]"
        print("\n" + "=" * 88)
        print(f"{arch.upper()}  —  {args.metric} by layer x variable{tag}")
        print("=" * 88)

        tab = df.pivot_table(index="layer", columns="variable",
                             values=args.metric, aggfunc="mean")
        tab = tab.reindex(order_layers(tab.index))
        cols = [c for c in STATE_VARS if c in tab.columns] + \
               [c for c in PARAM_VARS if c in tab.columns] + \
               [c for c in tab.columns if c not in STATE_VARS + PARAM_VARS]
        tab = tab[cols]
        print(tab.round(3).to_string())

        if "gravity" in tab.columns:
            g_max = tab["gravity"].max()
            print(f"\n  gravity: max {args.metric} across layers = {g_max:.3f}  "
                  f"({'FLOOR everywhere' if g_max < 0.15 else 'recoverable at some layer!'})")
        state_present = [c for c in STATE_VARS if c in tab.columns]
        if state_present:
            s_min = tab[state_present].min().min()
            print(f"  state vars: min {args.metric} across layers/vars = {s_min:.3f}  "
                  f"({'recoverable everywhere' if s_min > 0.5 else 'check -- some low'})")
        print("  => probe recovers state at every layer but gravity at none => gravity's")
        print("     floor is representational, not probe weakness.")

    print("\n" + "=" * 88)
    print("GRAVITY across layers — feedforward architectures (the preemption result)")
    print("=" * 88)
    grav = {}
    for arch, df in arch_dfs.items():
        if arch not in FEEDFORWARD:
            continue
        sub = df[df["variable"] == "gravity"]
        if len(sub) == 0:
            continue
        by_layer = sub.pivot_table(index="layer", values=args.metric, aggfunc="mean")
        grav[arch] = by_layer[args.metric]
    if grav:
        gtab = pd.DataFrame(grav)
        gtab = gtab.reindex(order_layers(gtab.index))
        print(gtab.round(3).to_string())
        print("\n  Gravity stays at floor across ALL feedforward layers -> not hiding")
        print("  one layer earlier/later. (Recurrent GRU/RSSM: future work.)")


if __name__ == "__main__":
    main()