import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GL = "g_over_l"
CTRL = "theta_dot"


def parse_seed(ckpt):
    m = re.search(r"model_WorldModel[A-Za-z]*_(\d+)_", str(ckpt))
    return int(m.group(1)) if m else -1


def repr_of(path):
    s = str(path).lower().replace("\\", "/")
    if "rollout" not in s:
        return "instantaneous"
    m = re.search(r"rollout[_-]?(full|h|z)", s)
    return f"rollout_{m.group(1)}" if m else "rollout_unknown"


def arch_of(folder):
    return Path(folder.rstrip("/")).name.split("_")[-1]


def find_folders(roots, stem):
    out = []
    for root in roots:
        out += glob.glob(str(Path(root) / f"{stem}*/"))
        if Path(root).name.startswith(stem):
            out.append(root if root.endswith("/") else root + "/")
    return sorted(set(out))


def _load(roots, stem):
    out = {}
    for fold in find_folders(roots, stem):
        key = (arch_of(fold), repr_of(fold))
        dfs = []
        for f in glob.glob(f"{fold}/*.csv"):
            d = pd.read_csv(f)
            d["seed"] = d["checkpoint"].apply(parse_seed)
            fn = Path(f).name.lower()
            d["env"] = "cartpole" if "cartpole" in fn else "pendulum"
            d["policy"] = "sparse" if "sparse" in fn else "noise"
            dfs.append(d)
        if dfs:
            out[key] = pd.concat(dfs, ignore_index=True)
    return out


def seed_stat(df, value_col, row_filter):
    sub = df[row_filter(df)]
    if len(sub) == 0:
        return None
    per_seed = sub.groupby("seed")[value_col].mean()
    return float(per_seed.mean()), float(per_seed.std()), int(per_seed.shape[0])


def probe_gl(probe, arch, rep, random=False):
    df = probe.get((arch, rep))
    if df is None:
        return None
    tgt = f"{GL}_random" if random else GL
    return seed_stat(df, "r2", lambda d: d["target"] == tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dir", action="append", required=True,
                    help="parent dir with linear_probe_seed_* and/or comparator_seed_* (repeatable)")
    ap.add_argument("--out", default="panel.pdf")
    args = ap.parse_args()

    roots = args.seed_dir
    probe = _load(roots, "linear_probe_seed")
    comp = _load(roots, "comparator_seed")

    if not probe:
        raise SystemExit(f"No linear_probe_seed_* folders under {roots}")
    if not comp:
        print(f"[warn] no comparator_seed_* under {roots}; Panel C empty")

    def pend(df):
        return df[df["env"] == "pendulum"] if "env" in df.columns else df
    comp = {k: pend(v) for k, v in comp.items()}
    probe = {k: pend(v) for k, v in probe.items()}

    A = {}
    A_rand = {}
    for arch, rep, lab in [("mlp", "instantaneous", "MLP\n(instant.)"),
                           ("rssm", "rollout_z", "RSSM\n(rollout_z)"),
                           ("rssm", "instantaneous", "RSSM\n(instant.)")]:
        v = probe_gl(probe, arch, rep)
        if v:
            A[lab] = v
            vr = probe_gl(probe, arch, rep, random=True)
            if vr:
                A_rand[lab] = vr

    B = {}
    v = probe_gl(probe, "rssm", "rollout_h")
    if v: B["RSSM\n(rollout_h)"] = v
    v = probe_gl(probe, "rssm", "rollout_h", random=True)
    if v: B["random\n(rollout_h)"] = v

    C = {}
    C_null = {}
    cdf = comp.get(("rssm", "rollout_h"))
    if cdf is not None and "cos_Jw_r" in cdf.columns:
        g = seed_stat(cdf, "cos_Jw_r", lambda d: d["variable"].isin(["gravity", "length"]))
        if g: C["g/l\n(rollout_h)"] = g
        t = seed_stat(cdf, "cos_Jw_r", lambda d: d["variable"] == CTRL)
        if t: C["theta_dot\n(control)"] = t
        if "cos_Jw_r_null_95" in cdf.columns:
            gn = seed_stat(cdf, "cos_Jw_r_null_95",
                           lambda d: d["variable"].isin(["gravity", "length"]))
            if gn: C_null["g/l\n(rollout_h)"] = gn[0]
            tn = seed_stat(cdf, "cos_Jw_r_null_95", lambda d: d["variable"] == CTRL)
            if tn: C_null["theta_dot\n(control)"] = tn[0]


    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    col_floor, col_recover, col_random, col_ctrl = "#9aa0a6", "#2a7de1", "#c0c4cc", "#e08a1e"

    if A:
        labels = list(A.keys())
        means = [A[k][0] for k in labels]; stds = [A[k][1] for k in labels]
        x = np.arange(len(labels))
        axA.bar(x, means, yerr=stds, capsize=5, width=0.6, color=col_floor,
                error_kw={"elinewidth": 1.2}, label="trained")
        for i, k in enumerate(labels):
            if k in A_rand:
                axA.hlines(A_rand[k][0], i - 0.3, i + 0.3, color="k", lw=1.5,
                           ls="--", zorder=5)
        axA.plot([], [], color="k", lw=1.5, ls="--", label="random (matched)")
        axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=8)
        axA.set_ylabel(r"probe $R^2$ ($g/l$)")
        axA.set_ylim(0, 1)
        axA.legend(fontsize=7, loc="upper right")
    axA.set_title("A  Instantaneous state\n(weak accessibility)", fontsize=10)

    if B:
        Bre = {("random RSSM\n(rollout_h)" if "random" in k else k): v for k, v in B.items()}
        labels = list(Bre.keys())
        means = [Bre[k][0] for k in labels]; stds = [Bre[k][1] for k in labels]
        x = np.arange(len(labels))
        axB.bar(x, means, yerr=stds, capsize=5, width=0.6,
                color=[col_random if "random" in k else col_recover for k in labels],
                error_kw={"elinewidth": 1.2})
        axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=8)
        axB.set_ylim(0, 1)
    axB.set_title("B  Recurrent state\n($g/l$ becomes recoverable)", fontsize=10)

    if C:
        labels = list(C.keys())
        means = [C[k][0] for k in labels]; stds = [C[k][1] for k in labels]
        x = np.arange(len(labels))
        axC.axhline(0, color="k", lw=0.6, alpha=0.4)
        for i, k in enumerate(labels):
            nb = C_null.get(k)
            if nb is not None:
                axC.fill_between([i - 0.35, i + 0.35], [-nb, -nb], [nb, nb],
                                 color="k", alpha=0.10, zorder=0)
                axC.hlines([nb, -nb], i - 0.35, i + 0.35, color="k", lw=0.8,
                           alpha=0.4, zorder=1)
        axC.bar(x, means, yerr=stds, capsize=5, width=0.5, zorder=3,
                color=[col_ctrl if "theta" in k else col_floor for k in labels],
                error_kw={"elinewidth": 1.2})
        axC.set_xticks(x); axC.set_xticklabels(labels, fontsize=8)
        axC.set_ylabel(r"$\cos(J w_{\mathrm{probe}}, r)$")
        allv = means + [C_null.get(k, 0) for k in labels]
        lo = min(-0.35, min(means) - max(stds) - 0.05)
        hi = max(0.5, max(allv) + 0.08)
        axC.set_ylim(lo, hi)
    else:
        axC.text(0.5, 0.5, "no comparator data", ha="center", va="center",
                 transform=axC.transAxes)
    axC.set_title("C  Causal fidelity\n(probe direction remains null)", fontsize=10)

    fig.suptitle("Recurrent accessibility does not imply causal fidelity",
                 fontsize=12, y=1.03)
    ns = [d[k][2] for d in (A, B) for k in d] or [1]
    fig.text(0.5, -0.04,
             r"Bars: mean $\pm$ SD across seeds ($n{=}$" + str(max(ns)) + " training seeds). "
             r"Panel A dashed = architecture-matched random encoder. "
             r"Panel C shading = per-variable two-sided 95% null envelope from "
             r"norm-matched random directions; $\dot\theta$ is a relative positive control.",
             ha="center", fontsize=7.5, alpha=0.75)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"saved -> {args.out}")
    for name, d in [("A", A), ("B", B), ("C", C)]:
        for k, val in d.items():
            print(f"  [{name}] {k.replace(chr(10),' ')}: {val[0]:.4f} +/- {val[1]:.4f} (n={val[2]})")
    for k, nb in C_null.items():
        print(f"  [C null_95] {k.replace(chr(10),' ')}: {nb:.4f}")
    for k, val in A_rand.items():
        print(f"  [A random] {k.replace(chr(10),' ')}: {val[0]:.4f}")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"saved -> {args.out}")
    for name, d in [("A", A), ("B", B), ("C", C)]:
        for k, val in d.items():
            print(f"  [{name}] {k.replace(chr(10),' ')}: {val[0]:.4f} +/- {val[1]:.4f} (n={val[2]})")



if __name__ == "__main__":
    main()