# metrics.py
import numpy as np
import torch


def transport_metrics(gtruths: np.ndarray, mshifts: np.ndarray, n_total: int) -> dict:
    n_used = len(gtruths)
    survival = n_used / n_total if n_total else 0.0
    if n_used < 2:
        return {"fraction": np.nan, "slope": np.nan, "r2": np.nan,
                "n_used": n_used, "n_total": n_total, "survival": survival,
                "gtruths": gtruths, "mshifts": mshifts}

    slope, intercept = np.polyfit(gtruths, mshifts, 1)
    pred = slope * gtruths + intercept
    ss_res = np.sum((mshifts - pred) ** 2)
    ss_tot = np.sum((mshifts - mshifts.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    fraction = mshifts.mean() / gtruths.mean() if gtruths.mean() != 0 else np.nan
    return {"fraction": fraction, "slope": slope, "r2": r2,
            "n_used": n_used, "n_total": n_total, "survival": survival,
            "gtruths": gtruths, "mshifts": mshifts}


def direction_alignment(dz, w) -> float:
    w = torch.from_numpy(w) if isinstance(w, np.ndarray) else w
    w = w.flatten().to(dz.dtype)
    if dz.dim() == 1:
        dz = dz.unsqueeze(0)
    num = dz @ w
    den = dz.norm(dim=1) * w.norm() + 1e-9
    cos = (num / den).abs()
    return float(cos.mean())


def null_summary(null_dicts):
    slopes = np.array([d["slope"] for d in null_dicts], dtype=float)
    return {"null_mean": float(np.nanmean(slopes)),
            "null_95": float(np.nanpercentile(slopes, 95)),
            "null_max": float(np.nanmax(slopes)),
            "null_slopes": slopes}