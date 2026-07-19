import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pooled_crb import fim_for_file, _safe_crb_cond



def crb_atlas(pattern, h=1e-2):
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=["g", "l", "cond", "log_cond", "crb_g", "crb_l"])
    rows = []
    for f in files:
        F, (g, l), N = fim_for_file(f, h=h)
        cg, cl, cond, ok = _safe_crb_cond(F)
        rows.append({
            "g": round(float(g), 3), "l": round(float(l), 3),
            "cond": cond if ok else np.nan,
            "log_cond": np.log10(cond) if (ok and cond > 0) else np.nan,
            "crb_g": cg if ok else np.nan,
            "crb_l": cl if ok else np.nan,
        })
    return pd.DataFrame(rows)


def panel(ax, df, value, title):
    if len(df) == 0 or df[value].isna().all():
        ax.text(0.5, 0.5, "rank-deficient\n(FIM singular\nper config)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set(title=title, xlabel="$g$", ylabel="$l$")
        return None
    grid = df.pivot_table(index="l", columns="g", values=value, aggfunc="mean")
    grid = grid.sort_index().sort_index(axis=1)
    im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{c:g}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{r:g}" for r in grid.index])
    ax.set(title=title, xlabel="$g$", ylabel="$l$")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--value", default="log_cond",
                    choices=["log_cond", "crb_g", "crb_l"],
                    help="quantity to colour by")
    ap.add_argument("--h", type=float, default=1e-2)
    ap.add_argument("--out", default="./fim_atlas.png")
    args = ap.parse_args()

    DATA = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "datasets")
    IMPULSE = os.path.join(DATA, "impulse_policy")

    panels = [
        ("Pendulum (forced)", IMPULSE, "PendulumSim"),
        ("CartPole", DATA, "CartPoleSim"),
    ]

    vlabel = {"log_cond": r"$\log_{10}\,\mathrm{cond}\,\mathcal{I}(g,l)$",
              "crb_g": r"CRB$(g)$", "crb_l": r"CRB$(l)$"}[args.value]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ims = []
    for ax, (label, d, env) in zip(axes, panels):
        df = crb_atlas(os.path.join(d, f"*{env}*"), h=args.h)
        im = panel(ax, df, args.value, label)
        ims.append(im)

    im_ref = next((im for im in ims if im is not None), None)
    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes, label=vlabel, fraction=0.046, pad=0.04)
    
    out_file = args.out[:-4]+ "_"+ args.value + ".png"
    fig.suptitle(f"FIM atlas: {vlabel}  across the $(g,l)$ grid", y=1.02)
    fig.savefig(out_file, bbox_inches="tight", dpi=200)
    print(f"saved -> {out_file}")

    for label, d, env in panels:
        df = crb_atlas(os.path.join(d, f"*{env}*"), h=args.h)
        if len(df) and not df[args.value].isna().all():
            print(f"\n{label}  ({args.value}):")
            print(df.pivot_table(index="l", columns="g", values=args.value).round(3).to_string())