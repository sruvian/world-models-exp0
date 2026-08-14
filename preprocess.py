# preprocess_windows.py
import numpy as np, glob, argparse, yaml
from trainer.trainer import split_gen

ap = argparse.ArgumentParser()
ap.add_argument("--yaml", required=True)   # any one yaml, to get paths + K
ap.add_argument("--out", required=True)    # e.g. windowed/K15.npz
a = ap.parse_args()

cfg = yaml.safe_load(open(a.yaml))
K = 50
transient = cfg["datasets"].get("transient", 1200)

all_s, all_a = [], []
for pattern in cfg["datasets"]["paths"]:
    for path in sorted(glob.glob(pattern)):
        d = np.load(path)
        all_s.append(d["states"]); all_a.append(d["actions"])
states = np.concatenate(all_s); actions = np.concatenate(all_a)

tr_s, tr_ns, tr_a, va_s, va_ns, va_a = split_gen(
    states, actions, rollout=K, windows_per_traj=5, split_seed=0, transient=transient)

import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez(a.out,
    tr_s=tr_s.numpy(), tr_ns=tr_ns.numpy(), tr_a=tr_a.numpy(),
    va_s=va_s.numpy(), va_ns=va_ns.numpy(), va_a=va_a.numpy())
print(f"saved {a.out}: tr_s {tr_s.shape}")