import argparse, glob, csv
from pathlib import Path
import numpy as np
import torch
from analysis.common.utils import ENV_NAMES, STATE_CHANNELS, parse_model, load_model, TAGS
from analysis.common.data_provider import collect_for_config
from analysis.common.utils import make_model, HIDDEN_DIM
from sklearn.linear_model import LinearRegression

from models.wmodel import WorldModelRSSM



DRIVEN_CONFIGS = [(15.0, 0.5), (9.8, 1.0), (15.0, 2.0), (9.8, 2.0), (5.0, 2.0)]
# DRIVEN_CONFIGS = [(15.0, 10.0)]
HELDOUT_CONFIGS = [(15, 10), (15, 18)]
SWEEP_CONFIGS =  DRIVEN_CONFIGS + HELDOUT_CONFIGS

def make_s0_driven(theta0=0.1, batch=8, device="cpu"):
    s0 = torch.zeros(batch, 5, device=device)
    s0[:, 0] = np.cos(theta0); s0[:, 1] = np.sin(theta0)
    s0[:, 2] = 0.0
    s0[:, 3] = 1.0; s0[:, 4] = 0.0
    return s0

def resonance_rollout_rssm(model, s0, drive_omega, dt, horizon, device="cpu"):
    model.eval()
    preds = []
    s = s0.clone().to(device)
    with torch.inference_mode():
        h, z = model.encode_computational(s)
        for k in range(horizon):
            t = (k + 1) * dt
            a0 = torch.zeros(s.shape[0], 1, device=device)
            h, z = model.step_computational((h, z), a0)
            s = model.decode_computational((h, z)).clone()
            ph = torch.as_tensor(drive_omega * t, device=device, dtype=s.dtype)
            s[:, 3] = torch.cos(ph); s[:, 4] = torch.sin(ph)
            z, mu, _ = model.posterior(h, s)
            z = mu
            preds.append(s)
    return torch.stack(preds, dim=1)

def resonance_rollout(model, s0, drive_omega, dt, horizon, device="cpu"):
    model.eval()
    preds = []; s = s0.clone().to(device)
    with torch.inference_mode():
        for k in range(horizon):
            t = (k + 1) * dt
            comp = model.encode_computational(s)
            a0 = torch.zeros(s.shape[0], 1, device=device)
            comp = model.step_computational(comp, a0)
            s = model.decode_computational(comp).clone()
            ph = torch.as_tensor(drive_omega * t, device=device, dtype=s.dtype)
            preds.append(s)
    return torch.stack(preds, dim=1)

def estimate_resonance(model, g, l, dt, horizon=3000, transient=1200,
                       n_omega=25, device="cpu"):
    omega_0 = (g / l) ** 0.5
    omegas = np.linspace(0.3 * omega_0, 2.5 * omega_0, n_omega)
    s0 = make_s0_driven(device=device)
    amps = []
    for w in omegas:
        if isinstance(model, WorldModelRSSM):
            preds = resonance_rollout_rssm(model, s0, w, dt, horizon, device)
        else:
            preds = resonance_rollout(model, s0, w, dt, horizon, device)
        if not torch.isfinite(preds).all():
            amps.append(np.nan); continue
        theta = torch.atan2(preds[:, transient:, 1], preds[:, transient:, 0])
        # amps.append(theta.abs().amax(dim=1).mean().item())
        amp = theta.pow(2).mean(dim=1).sqrt().mean().item()
        amps.append(amp)
    amps = np.array(amps)
    
    if not np.isfinite(amps).any():
        return float("nan"), float("nan")
    pk = np.nanargmax(amps)
    if pk == 0 or pk == len(amps) - 1:
        print(f"WARNING g={g:.1f} l={l:.2f}: peak at grid EDGE (pk={pk}/{len(amps)-1}) "
            f"— true peak likely outside sweep window, widen it.")
        peak = omegas[pk]
    else:
        y0, y1, y2 = amps[pk-1], amps[pk], amps[pk+1]
        d = y0 - 2*y1 + y2
        off = 0.5*(y0 - y2)/d if abs(d) > 1e-12 else 0.0
        peak = omegas[pk] + off * (omegas[1] - omegas[0])
    peakiness = amps[pk] / (np.nanmedian(amps) + 1e-8)
    return float(peak), float(peakiness)

def estimate_resonance_phaselag(model, g, l, dt, horizon=3000, transient=1200,
                                n_omega=25, device="cpu"):
    omega_0 = (g / l) ** 0.5
    omegas = np.linspace(0.3*omega_0, 2.5*omega_0, n_omega)
    s0 = make_s0_driven(device=device)
    lags = []
    for w in omegas:
        if isinstance(model, WorldModelRSSM):
            preds = resonance_rollout_rssm(model, s0, w, dt, horizon, device)
        else:
            preds = resonance_rollout(model, s0, w, dt, horizon, device)
        if not torch.isfinite(preds).all():
            lags.append(np.nan); continue
        theta = torch.atan2(preds[0, transient:, 1], preds[0, transient:, 0]).cpu().numpy()
        drive = torch.atan2(preds[0, transient:, 4], preds[0, transient:, 3]).cpu().numpy()
        
        th = theta - theta.mean()
        dr = np.cos(drive)
        t_arr = np.arange(len(th)) * dt
        ref = np.exp(-1j * w * t_arr)
        resp_phase = np.angle(np.sum(th * ref))
        drive_phase = np.angle(np.sum(dr * ref))
        lag = np.angle(np.exp(1j*(resp_phase - drive_phase)))
        lags.append(lag)
    lags = np.array(lags)
    return omegas, lags


def phaselag_readout(omegas, lags):
    lags = np.degrees(np.asarray(lags, dtype=float))
    omegas = np.asarray(omegas, dtype=float)
    good = np.isfinite(lags)
    if good.sum() < 3:
        return float("nan"), 0.0
    o, lg = omegas[good], lags[good]
    dlag = np.gradient(lg, o)
    w_steep = float(o[np.argmin(dlag)])
    swing = float(lg.max() - lg.min())
    return w_steep, swing


def estimate_effective_params_driven(model, states, actions, dt, device="cpu"):
    model.eval()
    n_traj, T, _ = states.shape
    preds = []
    with torch.no_grad():
        for t0 in range(T - 1):
            comp = model.encode_computational(states[:, t0, :].to(device))
            a = actions[:, t0].unsqueeze(-1).to(device)
            comp = model.step_computational(comp, a)
            preds.append(model.decode_computational(comp))
    pred = torch.stack(preds, dim=1)
    if not torch.isfinite(pred).all():
        return {"g_over_l": float("nan"), "b": float("nan"),
                "A": float("nan"), "r2": float("nan")}

    theta = torch.atan2(pred[:, :, 1], pred[:, :, 0])
    theta_u = np.unwrap(theta.cpu().numpy(), axis=1)
    thetadot = pred[:, :, 2].cpu().numpy()
    theta_ddot = np.gradient(thetadot, dt, axis=1, edge_order=2)

    sin_th   = np.sin(theta_u)
    drive_cos = states[:, :T-1, 3].cpu().numpy()

    X = np.stack([(-sin_th).reshape(-1),
                  thetadot.reshape(-1),
                  drive_cos.reshape(-1)], axis=1)
    y = theta_ddot.reshape(-1)
    m = np.isfinite(X).all(1) & np.isfinite(y)
    reg = LinearRegression().fit(X[m], y[m])
    c1, c2, c3 = reg.coef_
    return {"g_over_l": float(c1), "b": float(-c2),
            "A": float(c3), "r2": float(reg.score(X[m], y[m]))}
def _build_random(mf, cfg, device):
    
    model = make_model(cfg["model_name"],
                       state_dim=5 if len(STATE_CHANNELS[cfg['env']]) else 3, action_dim=1,
                       hidden_dim=HIDDEN_DIM.get(cfg["model_name"], 64),
                       latent_dim=cfg["latent"])
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--transient", type=int, default=1200)
    ap.add_argument("--n_omega", type=int, default= 75)
    ap.add_argument("--n_traj", type = int, default = 50)
    args = ap.parse_args()

    model_files = glob.glob(str(Path(args.models_dir) / "**" / "*.pt"), recursive=True)
    csv_handles = {}

    for mf in model_files:
        cfg = parse_model(Path(mf))

        trained  = load_model(mf, cfg, device=args.device)
        random_m = _build_random(mf, cfg, args.device)

        policy_tag = "sparse" if cfg["impulse"] else "noise"
        group = f"{ENV_NAMES[cfg['env']]}_{policy_tag}_{TAGS[cfg['model_name']]}"
        if group not in csv_handles:
            path = Path(f"{args.save_dir}/resonance_{group}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            hdr = not path.exists()
            fh = open(path, "a", newline="")
            w = csv.writer(fh)
            if hdr:
                w.writerow(["checkpoint", "model_type", "latent_dim", "k", "regime",
                            "eval_g", "eval_l", "true_omega0", "true_gl",
                            "meas_gl", "meas_b", "meas_A", "onestep_r2",
                            "w_steep", "swing", "amp_peak", "peakiness", "config_tag"])
            csv_handles[group] = (fh, w)
        fh, w = csv_handles[group]

        for model_type, model in [("trained", trained), ("random", random_m)]:
            
            for (g_eval, l_eval) in SWEEP_CONFIGS:
                true_omega0 = (g_eval / l_eval) ** 0.5
                params = {"gravity": g_eval, "length": l_eval,
              "mass1": 1.0, "dt": args.dt, "damping": 0.5,
              "drive_amp": 0.2, "max_action": 0}
                params["drive_omega"] = (g_eval / l_eval) ** 0.5
                states, actions = collect_for_config(
                cfg["env"], params, impulse=cfg["impulse"],
                seed=4200, n_traj=args.n_traj, steps=args.horizon, max_action=0)
                states  = states  if torch.is_tensor(states)  else torch.from_numpy(states)
                actions = actions if torch.is_tensor(actions) else torch.from_numpy(actions)
                states, actions = states.float(), actions.float()
                res = estimate_effective_params_driven(model, states, actions, args.dt, device=args.device)
                true_gl = g_eval / l_eval
                omegas, lags = estimate_resonance_phaselag(
                    model, g_eval, l_eval, args.dt,
                    horizon=args.horizon, transient=args.transient,
                    n_omega=args.n_omega, device=args.device)
                w_steep, swing = phaselag_readout(omegas, lags)

                amp_peak, peakiness = estimate_resonance(
                    model, g_eval, l_eval, args.dt,
                    horizon=args.horizon, transient=args.transient,
                    n_omega=args.n_omega, device=args.device)
                tag = "in_range" if (g_eval, l_eval) in DRIVEN_CONFIGS else "held_out"
                w.writerow([Path(mf).name, model_type, cfg["latent"], cfg["k"], cfg["regime"],
                            g_eval, l_eval, round(true_omega0, 6), round(true_gl, 6),
                            round(res["g_over_l"], 6) if np.isfinite(res["g_over_l"]) else "nan",
                            round(res["b"], 6)        if np.isfinite(res["b"])        else "nan",
                            round(res["A"], 6)        if np.isfinite(res["A"])        else "nan",
                            round(res["r2"], 6)       if np.isfinite(res["r2"])       else "nan",
                            round(w_steep, 6)         if np.isfinite(w_steep)         else "nan",
                            round(swing, 6),
                            round(amp_peak, 6)        if np.isfinite(amp_peak)        else "nan",
                            round(peakiness, 6), tag])
                fh.flush()
                print(f"[{Path(mf).name}] [{model_type:7s}] g={g_eval} l={l_eval} "
                      f"true_w0={true_omega0:.3f} w_steep={w_steep:.3f} swing={swing:.1f}° "
                      f"peakiness={peakiness:.2f}")
                print(f"[{Path(mf).name}] [{model_type:7s}] g={g_eval} l={l_eval} "
                f"true_gl={true_gl:.3f} meas_gl={res['g_over_l']:.3f} "
                f"meas_b={res['b']:.3f} meas_A={res['A']:.3f} r2={res['r2']:.3f}")
                

    for fh, _ in csv_handles.values():
        fh.close()
    print("Done.")