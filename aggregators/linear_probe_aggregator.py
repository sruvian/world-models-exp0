import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _repr_tag import repr_of

SEEDED = {"seed"}

STATE_VARS = ["cos_theta", "sin_theta", "theta_dot", "x", "x_dot"]
PARAM_VARS = ["length", "gravity", "g_over_l", "sqrt_l_over_g"]
VAR_ORDER = STATE_VARS + PARAM_VARS

DEGENERATE = {
    "holdl": {"gravity", "g_over_l"},
    "holdg": {"length", "g_over_l", "sqrt_l_over_g"}
}


def infer_regime(row):
    """Regime from the checkpoint name (holdg / holdl / combined)."""
    s = str(row.get("checkpoint", "")).lower()
    for r in ("holdg", "holdl", "combined"):
        if r in s:
            return r
    return "unknown"


def load_arch(csv_glob):
    dfs = []
    for f in glob.glob(csv_glob):
        d = pd.read_csv(f)
        d["representation"] = repr_of(f)
        dfs.append(d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    for c in ["r2", "r2_shuffled", "delta", "mi_mean", "mi_max",
              "perm", "perm_max", "mi_shuf_ratio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["r2"]) & np.isfinite(df["r2_shuffled"])]

    df["is_random"] = df["target"].astype(str).str.endswith("_random")
    df["base_target"] = df["target"].astype(str).str.replace(r"_random$", "", regex=True)
    df["regime"] = df.apply(infer_regime, axis=1)
    df["degenerate"] = [
        t in DEGENERATE.get(r, set()) for t, r in zip(df["base_target"], df["regime"])
    ]
    return df


def summarize(df, random_rows=False, representation=None):
    sub = df[df["is_random"] == random_rows]
    if representation is not None:
        sub = sub[sub["representation"] == representation]
    if len(sub) == 0:
        return None
    agg = dict(
        r2=("r2", "mean"),
        r2_std=("r2", "std"),
        shuffled=("r2_shuffled", "mean"),
        delta=("delta", "mean"),
        n=("r2", "count"),
    )
    for c, name in [("mi_mean", "mi"), ("mi_max", "mi_max"), ("mi_shuf_ratio", "mi_ratio")]:
        if c in sub.columns:
            agg[name] = (c, "mean")
    g = sub.groupby("base_target").agg(**agg)
    present = [v for v in VAR_ORDER if v in g.index]
    return g.reindex(present + [v for v in g.index if v not in VAR_ORDER])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="linear_probes")
    ap.add_argument("--floor_thresh", type=float, default=0.05,
                    help="training gain below this = not learned")
    ap.add_argument("--by_regime", action="store_true",
                    help="also break the parameter table down by regime")
    args = ap.parse_args()

    per_arch = {}
    for fold in sorted(glob.glob(f"{args.pattern}_*/")):
        name = Path(fold.rstrip("/")).name
        arch = name.removeprefix(f"{args.pattern}_")
        if "seed" in arch:
            print(f"[skip seeded] {arch}")
            continue
        df = load_arch(f"{fold}/*.csv")
        if df is None:
            print(f"[skip empty] {fold}")
            continue
        per_arch[arch] = df

    if not per_arch:
        print("No architecture data found."); return

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    for arch, df in per_arch.items():
        print("\n" + "=" * 92)
        print(f"{arch.upper()}  —  linear-probe recoverability")
        print("=" * 92)

        tr = summarize(df, random_rows=False)
        rn = summarize(df, random_rows=True)
        if tr is None:
            continue
        print("\n  [trained]")
        print(tr.round(4).to_string())
        if rn is not None:
            print("\n  [random encoder]")
            print(rn.round(4).to_string())

            gain = pd.DataFrame({
                "r2_trained": tr["r2"],
                "r2_random": rn["r2"],
                "training_gain": tr["r2"] - rn["r2"],
            })
            if "mi_max" in tr.columns and "mi_max" in rn.columns:
                gain["mi_max_trained"] = tr["mi_max"]
                gain["mi_max_random"] = rn["mi_max"]
                gain["mi_gain"] = tr["mi_max"] - rn["mi_max"]
            print("\n  [TRAINING GAIN]  (trained - random; ~0 => random features already do it)")
            print(gain.round(4).to_string())

        deg = df[df["degenerate"] & ~df["is_random"]]
        if len(deg):
            combos = sorted(set(zip(deg["regime"], deg["base_target"])))
            byreg = {}
            for r, t in combos:
                byreg.setdefault(r, []).append(t)
            for r, ts in byreg.items():
                print(f"\n  ! DEGENERATE in regime '{r}': {', '.join(ts)} are monotone "
                      f"transforms of one another (not independent measurements)")

    tr_tabs = {a: summarize(d, False) for a, d in per_arch.items()}
    rn_tabs = {a: summarize(d, True) for a, d in per_arch.items()}
    tr_tabs = {a: t for a, t in tr_tabs.items() if t is not None}

    r2_tab = pd.DataFrame({a: t["r2"] for a, t in tr_tabs.items()})
    present = [v for v in VAR_ORDER if v in r2_tab.index]
    r2_tab = r2_tab.reindex(present + [v for v in r2_tab.index if v not in VAR_ORDER])
    print("\n" + "=" * 92)
    print("CROSS-ARCHITECTURE  R²  (trained)")
    print("=" * 92)
    print(r2_tab.round(3).to_string())

    if any(t is not None for t in rn_tabs.values()):
        gain_tab = pd.DataFrame({
            a: tr_tabs[a]["r2"] - rn_tabs[a]["r2"]
            for a in tr_tabs if rn_tabs.get(a) is not None
        })
        gain_tab = gain_tab.reindex(present + [v for v in gain_tab.index if v not in VAR_ORDER])
        print("\n" + "=" * 92)
        print("CROSS-ARCHITECTURE  TRAINING GAIN  (r2_trained - r2_random)")
        print("  This is the claim-bearing number, not delta: delta only rules out label")
        print("  leakage; training gain rules out 'random features already encode it'.")
        print("=" * 92)
        print(gain_tab.round(3).to_string())

        print("\n" + "=" * 92)
        print(f"VERDICT  (training gain < {args.floor_thresh} => NOT a learned representation)")
        print("=" * 92)
        for v in PARAM_VARS:
            if v not in gain_tab.index:
                continue
            row = gain_tab.loc[v]
            status = ", ".join(
                f"{a}:{'floor' if row[a] < args.floor_thresh else f'{row[a]:+.2f}'}"
                for a in row.index if np.isfinite(row[a])
            )
            print(f"  {v:14s} {status}")
        for v in STATE_VARS:
            if v not in gain_tab.index:
                continue
            row = gain_tab.loc[v]
            status = ", ".join(f"{a}:{row[a]:+.2f}" for a in row.index if np.isfinite(row[a]))
            print(f"  {v:14s} {status}")
        print("\n  NOTE: state variables often show SMALL training gain because random")
        print("  overcomplete ReLU features already encode observable state linearly.")
        print("  The dissociation to report is therefore: state recoverable from BOTH")
        print("  trained and random; parameters recoverable from NEITHER.")

    reps_present = sorted({r for d in per_arch.values() for r in d["representation"].unique()})
    if len(reps_present) > 1:
        print("\n" + "=" * 92)
        print("REPRESENTATION COMPARISON  (instantaneous vs rollout_z vs rollout_h)")
        print("  z / instantaneous low + h high  =>  parameter is TEMPORAL, not stored per-frame")
        print("=" * 92)
        for arch, df in per_arch.items():
            avail = [r for r in ["instantaneous", "rollout_z", "rollout_h", "rollout_full"]
                     if r in set(df["representation"])]
            if len(avail) < 2:
                continue
            print(f"\n--- {arch.upper()} ---")
            cols = {}
            for rep in avail:
                t = summarize(df, random_rows=False, representation=rep)
                if t is None:
                    continue
                cols[f"{rep}_r2"] = t["r2"]
                rn = summarize(df, random_rows=True, representation=rep)
                if rn is not None:
                    cols[f"{rep}_gain"] = t["r2"] - rn["r2"]
            tab = pd.DataFrame(cols)
            present_v = [v for v in VAR_ORDER if v in tab.index]
            tab = tab.reindex(present_v + [v for v in tab.index if v not in VAR_ORDER])
            print(tab.round(3).to_string())

    if args.by_regime:
        print("\n" + "=" * 92)
        print("PARAMETERS BY REGIME  (combined = independent; holdg/holdl = degenerate)")
        print("=" * 92)
        for arch, df in per_arch.items():
            sub = df[(~df["is_random"]) & (df["base_target"].isin(PARAM_VARS))]
            if len(sub) == 0:
                continue
            g = sub.groupby(["regime", "base_target"]).agg(
                r2=("r2", "mean"), delta=("delta", "mean"),
                degenerate=("degenerate", "first"), n=("r2", "count"),
            )
            print(f"\n--- {arch.upper()} ---")
            print(g.round(4).to_string())


if __name__ == "__main__":
    main()  