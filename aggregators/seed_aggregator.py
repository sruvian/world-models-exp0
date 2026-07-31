import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PARAM_VARS = ["gravity", "length", "g_over_l", "sqrt_l_over_g"]
STATE_VARS = ["cos_theta", "sin_theta", "theta_dot", "x", "x_dot"]

FLOOR_R2 = 0.10


def parse_seed(checkpoint):
    m = re.search(r"model_WorldModel[A-Za-z]*_(\d+)_", str(checkpoint))
    return int(m.group(1)) if m else -1


def repr_of(path):
    s = str(path).lower().replace("\\", "/")
    if "rollout" not in s:
        return "instantaneous"
    m = re.search(r"rollout[_-]?(full|h|z)", s)
    return f"rollout_{m.group(1)}" if m else "rollout_unknown"


def arch_of(folder):
    return Path(folder.rstrip("/")).name.split("_")[-1]


def load_folders(pattern):
    out = {}
    for fold in sorted(glob.glob(f"{pattern}*/")):
        arch = arch_of(fold)
        rep = repr_of(fold)
        dfs = []
        for f in glob.glob(f"{fold}/*.csv"):
            d = pd.read_csv(f)
            d["seed"] = d["checkpoint"].apply(parse_seed)
            d["env"] = "cartpole" if "cartpole" in Path(f).name.lower() else "pendulum"
            d["policy"] = "sparse" if "sparse" in Path(f).name.lower() else "noise"
            dfs.append(d)
        if dfs:
            out[(arch, rep)] = pd.concat(dfs, ignore_index=True)
    return out


def kofn(series, predicate):
    vals = series.dropna()
    if len(vals) == 0:
        return "0/0"
    k = int(sum(predicate(v) for v in vals))
    return f"{k}/{len(vals)}"


def agg_probe(data):
    print("=" * 96)
    print("SEED STABILITY — LINEAR PROBES  (seeds 1-5)")
    print(f"  parameter 'at floor' = R2 < {FLOOR_R2}. Want floor in ALL seeds.")
    print("=" * 96)
    for (arch, rep), df in data.items():
        df = df[~df["target"].astype(str).str.endswith("_random")]
        for gtag, gdf in split_groups(df, by_env=True, by_policy=True):
            print(f"\n--- {arch.upper()} / {rep} / {gtag} ---")
            g = gdf.groupby("target").agg(
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                r2_min=("r2", "min"),
                r2_max=("r2", "max"),
                n_seeds=("seed", "nunique"),
            )
            order = [v for v in STATE_VARS + PARAM_VARS if v in g.index]
            g = g.reindex(order + [v for v in g.index if v not in order])
            floor = {}
            for v in PARAM_VARS:
                sub = gdf[gdf["target"] == v]
                if len(sub):
                    per_seed = sub.groupby("seed")["r2"].mean()
                    floor[v] = kofn(per_seed, lambda x: x < FLOOR_R2)
            g["floor_kofn"] = pd.Series(floor)
            print(g.round(4).to_string())


def _cell_rd(df, by_seed=True):
    """real - rand_dims per cell; cells keyed by (seed?, target_var, top_k)."""
    idx = (["seed"] if by_seed else []) + ["target_var", "top_k"]
    piv = df.pivot_table(index=idx, columns="patch_mode", values="shift", aggfunc="mean")
    for m in ("real", "rand_dims", "rand_values"):
        if m not in piv.columns:
            return None
    piv = piv.dropna(subset=["real", "rand_dims", "rand_values"]).reset_index()
    piv["rd"] = piv["real"] - piv["rand_dims"]
    piv["rv"] = piv["real"] - piv["rand_values"]
    return piv


def agg_patch(data, envs=("pendulum",), label="WITHIN-CONFIG"):
    print("=" * 96)
    print(f"SEED STABILITY — {label} PATCHING  (seeds 1-5)")
    print("  rd = real - rand_dims. Parameter non-localisation = rd n.s./negative per seed.")
    print("  Want: parameter rd NOT positive in each seed; state rd positive (power) in each.")
    if len(envs) > 1:
        print("  Envs reported separately: cartpole has the [:, :2] under-measurement caveat")
        print("  on real-dims transport, so it is NOT pooled with pendulum.")
    print("=" * 96)
    for (arch, rep), df_full in data.items():
        do_env = not (len(envs) == 1 and envs[0] is None)
        for gtag, df in split_groups(df_full, by_env=do_env, by_policy=True):
            env_in_tag = gtag.split("/")[0] if "/" in gtag else gtag
            note = "   [caveat: real-dims transport under-measured]" if "cartpole" in gtag else ""
            print(f"\n--- {arch.upper()} / {rep} / {gtag} ---{note}")
            rows = []
            for seed, sdf in df.groupby("seed"):
                piv = _cell_rd(sdf, by_seed=False)
                if piv is None or len(piv) == 0:
                    continue
                for vtype, vset in [("param", PARAM_VARS), ("state", STATE_VARS)]:
                    sub = piv[piv["target_var"].isin(vset)]
                    if len(sub) < 3:
                        rows.append({"seed": seed, "vtype": vtype, "mean_rd": np.nan,
                                     "p": np.nan, "pos": np.nan, "n": len(sub)})
                        continue
                    try:
                        _, p = wilcoxon(sub["rd"])
                    except ValueError:
                        p = np.nan
                    rows.append({"seed": seed, "vtype": vtype,
                                 "mean_rd": sub["rd"].mean(), "p": p,
                                 "pos": int((sub["rd"] > 0).sum()), "n": len(sub)})
            if not rows:
                continue
            r = pd.DataFrame(rows)
            for vtype in ["param", "state"]:
                v = r[r["vtype"] == vtype]
                if len(v) == 0 or v["mean_rd"].notna().sum() == 0:
                    continue
                mean_rd = v["mean_rd"].mean()
                std_rd = v["mean_rd"].std()
                if vtype == "param":
                    cons = kofn(v.dropna(subset=["mean_rd"]).apply(
                        lambda row: (not np.isfinite(row["p"])) or (row["p"] >= 0.05) or (row["mean_rd"] < 0),
                        axis=1), lambda x: bool(x))
                    want = "n.s.-or-negative"
                else:
                    cons = kofn(v.dropna(subset=["mean_rd"]).apply(
                        lambda row: np.isfinite(row["p"]) and (row["p"] < 0.05) and (row["mean_rd"] > 0),
                        axis=1), lambda x: bool(x))
                    want = "positive-sig"
                print(f"  {vtype:5s}: mean_rd={mean_rd:+.4f} +/- {std_rd:.4f}  "
                      f"[{want} in {cons} seeds]")


def agg_comparator(data):
    print("=" * 96)
    print("SEED STABILITY — COMPARATOR  (seeds 1-5)")
    print("  [param] gravity/length: cosines ~0, cos_Jw_r within null. Want in ALL seeds.")
    print("  [theta_dot CTRL] power control: cosines ABOVE, cos_Jw_r EXCEEDS null -- proves")
    print("  the test has power (a decodable state var DOES align with the operator direction).")
    print("  If theta_dot were also ~0, the parameter nulls would be uninformative.")
    print("=" * 96)
    CFG = ["gravity", "length"]
    CTRL = "theta_dot"
    for (arch, rep), df_full in data.items():
        for gtag, df in split_groups(df_full, by_env=True, by_policy=True):
            print(f"\n--- {arch.upper()} / {rep} / {gtag} ---")
            sub = df[df["variable"].isin(CFG)]
            ctrl = df[df["variable"] == CTRL]
            if len(sub) == 0 and len(ctrl) == 0:
                continue
            for col in ["dz_probe_cossim", "dzopt_probe_cossim"]:
                if col not in df.columns:
                    continue
                if len(sub):
                    ps = sub.groupby("seed")[col].mean()
                    cons = kofn(ps, lambda x: abs(x) < 0.05)
                    print(f"  {col:20s} [param]: mean={ps.mean():+.4f} +/- {ps.std():.4f}  "
                          f"[|cos|<0.05 in {cons} seeds]  per-seed={[round(float(x),3) for x in ps.values]}")
                if len(ctrl):
                    cs = ctrl.groupby("seed")[col].mean()
                    cons = kofn(cs, lambda x: abs(x) > 0.10)
                    print(f"  {col:20s} [theta_dot CTRL]: mean={cs.mean():+.4f} +/- {cs.std():.4f}  "
                          f"[|cos|>0.10 in {cons} seeds]  (power control: should be ABOVE)")
            if "cos_Jw_r" in df.columns and "cos_Jw_r_null_95" in df.columns:
                if len(sub):
                    ps = sub.groupby("seed").agg(cos=("cos_Jw_r", "mean"),
                                                 null95=("cos_Jw_r_null_95", "mean"))
                    ps["within"] = ps["cos"].abs() <= ps["null95"]
                    print(f"  {'cos_Jw_r':20s} [param]: mean={ps['cos'].mean():+.4f} +/- {ps['cos'].std():.4f}  "
                          f"[|cos|<=null in {kofn(ps['within'], lambda x: bool(x))} seeds]")
                if len(ctrl):
                    cs = ctrl.groupby("seed").agg(cos=("cos_Jw_r", "mean"),
                                                  null95=("cos_Jw_r_null_95", "mean"))
                    cs["above"] = cs["cos"].abs() > cs["null95"]
                    print(f"  {'cos_Jw_r':20s} [theta_dot CTRL]: mean={cs['cos'].mean():+.4f} +/- {cs['cos'].std():.4f}  "
                          f"[|cos|>null in {kofn(cs['above'], lambda x: bool(x))} seeds]  (should be ABOVE)")
            if "Jw_rel_norm" in df.columns and "Jw_rel_null_mean" in df.columns and len(sub):
                ps = sub.groupby("seed").agg(rel=("Jw_rel_norm", "mean"),
                                             rel_null=("Jw_rel_null_mean", "mean"))
                ps["ratio"] = ps["rel"] / (ps["rel_null"] + 1e-12)
                print(f"  {'Jw_rel_norm':20s} [param]: ratio-to-null={ps['ratio'].mean():.3f} "
                      f"+/- {ps['ratio'].std():.3f}")


def agg_behavioural(data):
    print("=" * 96)
    print("SEED STABILITY — BEHAVIOURAL g/l TRACKING  (seeds 1-5)")
    print("  The load-bearing POSITIVE result: effective g/l tracks true g/l (slope ~1).")
    print("  Positive claims are where seed-fragility bites -- tight slope across seeds")
    print("  is what rebuts 'you found one lucky initialisation'.")
    print("  Watch RSSM: if slope is consistently ~0.67 (vs MLP ~1.0), the compression")
    print("  is architectural, not an initialisation accident -> reportable finding.")
    print("=" * 96)
    for (arch, rep), df_full in data.items():
        base = df_full[df_full["env"] == "pendulum"] if "env" in df_full.columns else df_full
        if "r2" in base.columns:
            base = base[base["r2"] >= 0.5]
        for gtag, df in split_groups(base, by_env=False, by_policy=True):
            if len(df) == 0:
                continue
            print(f"\n--- {arch.upper()} / {rep} / pendulum/{gtag} ---")
            rows = []
            for seed, sdf in df.groupby("seed"):
                tcol = "true_gl" if "true_gl" in sdf else "true_g_over_l"
                mcol = "measured_gl" if "measured_gl" in sdf else "measured_g_over_l"
                if tcol not in sdf or mcol not in sdf:
                    continue
                tv = sdf[tcol].values.astype(float)
                mv = sdf[mcol].values.astype(float)
                m = np.isfinite(tv) & np.isfinite(mv)
                if m.sum() < 3:
                    continue
                slope, intercept = np.polyfit(tv[m], mv[m], 1)
                r_ = np.corrcoef(tv[m], mv[m])[0, 1]
                rows.append({"seed": seed, "slope": slope, "r": r_,
                             "mean_abs_err": np.abs(mv[m] - tv[m]).mean(), "n": int(m.sum())})
            if not rows:
                print("  (no fittable seeds -- check true_gl/measured_gl columns)")
                continue
            rr = pd.DataFrame(rows).set_index("seed").sort_index()
            print(rr.round(4).to_string())
            print(f"  SLOPE across seeds: {rr['slope'].mean():.3f} +/- {rr['slope'].std():.3f}   "
                  f"r: {rr['r'].mean():.3f} +/- {rr['r'].std():.3f}")
            band = kofn(rr["slope"], lambda x: 0.5 <= x <= 1.5)
            print(f"  slope in [0.5, 1.5] (tracking) in {band} seeds")


def agg_induced(data):
    print("=" * 96)
    print("SEED STABILITY — INDUCED DIRECTION  cos(w_probe, J_E S_x v)  (seeds 1-5)")
    print("  Oracle-free: parameters should sit BELOW chance |cos| = sqrt(2/pi d) in every")
    print("  seed; theta_dot (state control) ABOVE chance. Reported as ratio-to-chance.")
    print("=" * 96)
    for (arch, rep), df_full in data.items():
        for gtag, df in split_groups(df_full, by_env=True, by_policy=True):
            print(f"\n--- {arch.upper()} / {rep} / {gtag} ---")
            for var, want in [("theta_dot", "ABOVE"), ("gravity", "BELOW"), ("length", "BELOW")]:
                sub = df[df["variable"] == var]
                if len(sub) == 0:
                    continue
                rows = []
                for seed, sdf in sub.groupby("seed"):
                    d = float(sdf["latent_dim"].mean())
                    chance = np.sqrt(2.0 / (np.pi * d))
                    abscos = sdf["abscos_mean"].mean()
                    rows.append({"seed": seed, "abscos": abscos, "chance": chance,
                                 "ratio": abscos / chance})
                if not rows:
                    continue
                r = pd.DataFrame(rows)
                if want == "BELOW":
                    cons = kofn(r["ratio"], lambda x: x < 1.0)
                    tag = f"below chance in {cons} seeds"
                else:
                    cons = kofn(r["ratio"], lambda x: x > 1.0)
                    tag = f"above chance in {cons} seeds"
                print(f"  {var:10s}: |cos|={r['abscos'].mean():.4f} +/- {r['abscos'].std():.4f}  "
                      f"ratio-to-chance={r['ratio'].mean():.3f} +/- {r['ratio'].std():.3f}  "
                      f"[{want}: {tag}]")


def split_groups(df, by_env=True, by_policy=True):
    envs = sorted(df["env"].unique()) if (by_env and "env" in df.columns) else [None]
    pols = sorted(df["policy"].unique()) if (by_policy and "policy" in df.columns) else [None]
    for e in envs:
        for p in pols:
            sub = df
            if e is not None:
                sub = sub[sub["env"] == e]
            if p is not None:
                sub = sub[sub["policy"] == p]
            if len(sub) == 0:
                continue
            tag = "/".join(x for x in [e, p] if x is not None)
            yield (tag or "all"), sub


def agg_multilayer(data):
    print("=" * 96)
    print("SEED STABILITY — MULTILAYER PROBES  (seeds 1-5)")
    print("  Probe R2 per network layer. Parameter 'at floor' if training gain (above")
    print(f"  baseline) < {FLOOR_R2} AND R2 < {2*FLOOR_R2}. Want floor at EVERY layer in ALL seeds:")
    print("  the parameter is not linearly present anywhere in the network, not just the")
    print("  final latent. State variables (theta_dot, x) should be high at every layer.")
    print("  Generic over architectures: layers come from the CSV, not hardcoded.")
    print("=" * 96)
    R2 = "r2_mean"
    GAIN = "above_baseline"
    LAYER = "layer"
    VARC = "variable"

    for (arch, rep), df_full in data.items():
        for gtag, df in split_groups(df_full, by_env=True, by_policy=True):
            if LAYER not in df.columns or VARC not in df.columns:
                print(f"  [skip {arch}/{rep}/{gtag}: no {LAYER}/{VARC} columns]")
                continue
            canon = ["encoder_early", "encoder_late", "latent", "computational",
                     "transition_early", "transition_late",
                     "decoder_early", "decoder_late"]
            layers = [L for L in canon if L in set(df[LAYER])] + \
                     [L for L in df[LAYER].unique() if L not in canon]
            print(f"\n--- {arch.upper()} / {rep} / {gtag} ---")

            for vgroup, vlist in [("PARAMETERS", PARAM_VARS), ("STATE", STATE_VARS)]:
                present_v = [v for v in vlist if v in set(df[VARC])]
                if not present_v:
                    continue
                print(f"  [{vgroup}]")
                header = "    " + f"{'layer':16s}" + "".join(f"{v:>16s}" for v in present_v)
                print(header)
                for L in layers:
                    cells = []
                    for v in present_v:
                        sub = df[(df[LAYER] == L) & (df[VARC] == v)]
                        if len(sub) == 0:
                            cells.append(f"{'--':>16s}"); continue
                        per_seed_r2 = sub.groupby("seed")[R2].mean()
                        per_seed_gain = sub.groupby("seed")[GAIN].mean() if GAIN in sub.columns else per_seed_r2
                        r2m = per_seed_r2.mean(); r2s = per_seed_r2.std()
                        if vgroup == "PARAMETERS":

                            floors = (per_seed_r2 < 2 * FLOOR_R2)
                            k = kofn(floors, lambda x: bool(x))
                            cells.append(f"{r2m:.2f}±{r2s:.2f}({k})".rjust(16))
                        else:
                            k = kofn(per_seed_r2, lambda x: x > 0.7)
                            cells.append(f"{r2m:.2f}±{r2s:.2f}[{k}]".rjust(16))
                    print(f"    {L:16s}" + "".join(cells))
                if vgroup == "PARAMETERS":
                    print(f"    (value = mean R2 ± SD across seeds; (k/N) = floor in k seeds)")
                else:
                    print(f"    ([k/N] = R2>0.7 in k seeds; state should be high everywhere)")

            all_floored = True
            n_checks = 0
            for L in layers:
                for v in [x for x in PARAM_VARS if x in set(df[VARC])]:
                    sub = df[(df[LAYER] == L) & (df[VARC] == v)]
                    if len(sub) == 0:
                        continue
                    per_seed_r2 = sub.groupby("seed")[R2].mean()
                    floors = (per_seed_r2 < 2 * FLOOR_R2)
                    n_checks += 1
                    if not floors.all():
                        all_floored = False
            verdict = "YES" if all_floored else "NO (some layer/seed clears floor)"
            print(f"  => parameters floored at EVERY layer in EVERY seed: {verdict}  "
                  f"({n_checks} layer×param checks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["probe", "patch", "cross", "comparator", "behavioural", "induced", "multilayer"])
    ap.add_argument("--pattern", required=True)
    args = ap.parse_args()

    data = load_folders(args.pattern)
    if not data:
        print(f"No folders match {args.pattern}*"); return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # sanity: report which seeds were actually found
    seeds_found = sorted({s for df in data.values() for s in df["seed"].unique() if s >= 0})
    print(f"[seeds found: {seeds_found}]  [groups: {sorted(data.keys())}]\n")

    if args.kind == "probe":
        agg_probe(data)
    elif args.kind == "patch":
        agg_patch(data, envs=("pendulum", "cartpole"), label="WITHIN-CONFIG")
    elif args.kind == "cross":
        agg_patch(data, envs=(None,), label="CROSS-CONFIG")
    elif args.kind == "comparator":
        agg_comparator(data)
    elif args.kind == "behavioural":
        agg_behavioural(data)
    elif args.kind == "induced":
        agg_induced(data)
    elif args.kind == "multilayer":
        agg_multilayer(data)


if __name__ == "__main__":
    main()