from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

class SeparableCritic(nn.Module):
    def __init__(self, z_dim, y_dim, hidden=256, embed=32, temp=1.0):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed))
        self.h = nn.Sequential(
            nn.Linear(y_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed))
        self.temp = temp

    def forward(self, z, y):
        gz = self.g(z)
        hy = self.h(y)
        return gz @ hy.t() / self.temp


class ConcatCritic(nn.Module):
    def __init__(self, z_dim, y_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, z, y):
        B = z.shape[0]
        zi = z.unsqueeze(1).expand(B, B, z.shape[1])
        yj = y.unsqueeze(0).expand(B, B, y.shape[1])
        pair = torch.cat([zi, yj], dim=-1).reshape(B * B, -1)
        return self.net(pair).reshape(B, B)

def infonce_lower_bound(scores):

    B = scores.shape[0]
    pos = scores.diag()
    logdenom = torch.logsumexp(scores, dim=1)
    return (pos - logdenom).mean() + np.log(B)

def _standardize(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    mu = a.mean(0, keepdims=True)
    sd = a.std(0, keepdims=True) + 1e-6
    return (a - mu) / sd


def estimate_mi_smile(Z, Y, *, critic="separable",
                      hidden=256, embed=32, temp=1.0,
                      steps=4000, batch=256, lr=1e-4,
                      ema=0.99, device="cpu", seed=0, eval_tail=200):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    Zs = _standardize(Z)
    Ys = _standardize(Y)
    N = Zs.shape[0]
    Zt = torch.from_numpy(Zs).to(device)
    Yt = torch.from_numpy(Ys).to(device)

    if critic == "separable":
        net = SeparableCritic(Zs.shape[1], Ys.shape[1], hidden, embed, temp).to(device)
    else:
        net = ConcatCritic(Zs.shape[1], Ys.shape[1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    history = np.empty(steps, dtype=np.float32)
    ema_val = 0.0
    for t in range(steps):
        idx = rng.choice(N, size=min(batch, N), replace=False)
        z = Zt[idx]; y = Yt[idx]
        scores = net(z, y)
        lb = infonce_lower_bound(scores)
        loss = -lb
        opt.zero_grad(); loss.backward(); opt.step()
        val = float(lb.detach().cpu())
        history[t] = val
        ema_val = ema * ema_val + (1 - ema) * val if t else val

    tail = history[-eval_tail:]
    return {"mi": float(np.mean(tail)),
            "mi_std": float(np.std(tail)),
            "mi_ema": float(ema_val),
            "history": history}


def estimate_mi_multiseed(Z, Y, *, seeds=(0, 1, 2), **kw):
    vals = []
    hists = []
    for s in seeds:
        r = estimate_mi_smile(Z, Y, seed=s, **kw)
        vals.append(r["mi"]); hists.append(r["history"])
    vals = np.array(vals)
    return {"mi_mean": float(vals.mean()), "mi_sd": float(vals.std()),
            "mi_per_seed": vals.tolist(), "histories": hists}

def mi_dissociation(latents_trained, latents_random, targets, *, seeds=(0, 1, 2),
                    device="cpu", **kw):
    out = {}
    for name, y in targets.items():
        out[name] = {}
        
        tr = estimate_mi_multiseed(latents_trained, y,
                                    seeds=seeds, device=device, **kw)
        rd = estimate_mi_multiseed(latents_random, y,
                                    seeds=seeds, device=device, **kw)
        gain = tr["mi_mean"] - rd["mi_mean"]
        gain_sd = float(np.hypot(tr["mi_sd"], rd["mi_sd"]))
        out[name] = {
            "trained_mi": tr["mi_mean"], "trained_sd": tr["mi_sd"],
            "random_mi":  rd["mi_mean"], "random_sd":  rd["mi_sd"],
            "gain": gain, "gain_sd": gain_sd,
            "trained_per_seed": tr["mi_per_seed"],
            "random_per_seed":  rd["mi_per_seed"],
        }
        print(f"  {name:16s} "
                f"trained={tr['mi_mean']:.4f}±{tr['mi_sd']:.4f}  "
                f"random={rd['mi_mean']:.4f}±{rd['mi_sd']:.4f}  "
                f"gain={gain:+.4f}±{gain_sd:.4f}")
    return out