# metrics.py
import numpy as np
import torch


def transport_metrics(gtruths, mshifts, n_total):
    n_used = len(gtruths)
    survival = n_used / n_total if n_total else 0.0
    finite = np.isfinite(gtruths) & np.isfinite(mshifts)
    gtruths, mshifts = gtruths[finite], mshifts[finite]
    n_used = len(gtruths)

    if n_used < 2:
        return {"fraction": np.nan, "slope": np.nan, "r2": np.nan,
                "n_used": n_used, "n_total": n_total, "survival": survival,
                "gtruths": gtruths, "mshifts": mshifts}
    if np.ptp(gtruths) < 1e-9 or np.ptp(mshifts) < 1e-9:
        return {"fraction": np.nan, "slope": np.nan, "r2": np.nan,
                "n_used": n_used, "n_total": n_total, "survival": survival,
                "gtruths": gtruths, "mshifts": mshifts}
    try:
        slope, intercept = np.polyfit(gtruths, mshifts, 1)
    except np.linalg.LinAlgError:
        return {"fraction": np.nan, "slope": np.nan, "r2": np.nan,
                "n_used": n_used, "n_total": n_total, "survival": survival,
                "gtruths": gtruths, "mshifts": mshifts}
    pred = slope * gtruths + intercept
    ss_res = np.sum((mshifts - pred) ** 2)
    ss_tot = np.sum((mshifts - mshifts.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    fraction = mshifts.mean() / gtruths.mean() if gtruths.mean() != 0 else np.nan
    return {"fraction": fraction, "slope": slope, "r2": r2,
            "n_used": n_used, "n_total": n_total, "survival": survival,
            "gtruths": gtruths, "mshifts": mshifts}


def direction_alignment(A, B):
    A = torch.as_tensor(A).reshape(A.shape[0] if hasattr(A,'shape') and len(A.shape)>1 else 1, -1).float()
    B = torch.as_tensor(B).float()
    if B.ndim == 1:
        B = B.unsqueeze(0).expand(A.shape[0], -1)
    else:
        B = B.reshape(A.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(A, B, dim=1).detach()
    return float(cos.mean())

def null_summary(null_dicts):
    slopes = np.array([d["slope"] for d in null_dicts], dtype=float)
    return {"null_mean": float(np.nanmean(slopes)),
            "null_95": float(np.nanpercentile(slopes, 95)),
            "null_max": float(np.nanmax(slopes)),
            "null_slopes": slopes,
            "null_nonnan": int(np.sum(~np.isnan(slopes))),}


def pca_operator(dz: torch.Tensor, probe_direction: np.ndarray)->dict:
    D = dz.detach().numpy()
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    U, S, Vt = np.linalg.svd(Dn, full_matrices=False)
    var = (S**2) / (S**2).sum()
    pc1 = Vt[0]
    w = probe_direction / np.linalg.norm(probe_direction)
    return {
        "pc1_var": float(var[0]),
        "top3_var": float(var[:3].sum()),
        "dz_pc1_cos": float(np.mean(np.abs(Dn @ pc1))),
        "pc1_probe_cos": float(abs(pc1 @ w)),
    }