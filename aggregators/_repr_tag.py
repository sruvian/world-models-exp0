import re
from pathlib import Path

def repr_of(path):
    s = str(path).lower().replace("\\", "/")
    if "rollout" not in s:
        return "instantaneous"
    for tok in ("full", "_h_", "_h/", "_h.", "_z_", "_z/", "_z."):
        if tok in s:
            return "rollout_" + tok.strip("_/.")
    m = re.search(r"rollout[_-]?(full|h|z)", s)
    return f"rollout_{m.group(1)}" if m else "rollout_unknown"