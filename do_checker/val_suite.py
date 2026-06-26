from checker import DoChecker
import numpy as np
import torch
from metrics import *

class ValidationSuite:

    def __init__(self, checker: DoChecker, lam_reg: float = 0, lr :float = 1e-3) -> None:
        
        self.checker = checker
        self.lam_reg = lam_reg
        self.lr = lr

    def optimisation_bound(self, states: torch.Tensor | np.ndarray, action: float| torch.Tensor, target: torch.Tensor | np.ndarray):
        
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(action, float):
            action = torch.Tensor([action])
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        action = action.reshape(-1, 1) 
        with torch.no_grad():
            z = self.checker.encode(states).detach()
        
        dz = torch.zeros_like(z, requires_grad= True)
        optimiser = torch.optim.Adam([dz], lr = self.lr)
        for _ in range(200):
            
            optimiser.zero_grad()
            pred = self.checker.decode(self.checker.step((z + dz), action))
            loss = ((target - pred)**2).sum() + self.lam_reg * (dz**2).sum()
            loss.backward()
            
            optimiser.step()
        with torch.no_grad():
            final_decode = self.checker.decode(self.checker.step((z + dz), action))
            final_err = ((target - final_decode)**2).sum().sqrt()
            return dz.detach(), final_err.detach(), final_decode.detach().numpy(), z.norm()

    def direction_transport(self, source_states: torch.Tensor, source_config: dict, target_config:dict, channel: int,
                probe_direction: np.ndarray, target_value, action: torch.Tensor, target_states: torch.Tensor | None = None):
        ground_truths, model_shifts = [], []
        for i, state in enumerate(source_states):
            target_state = None if target_states is None else target_states[i]
            delta_ground_truth, delta_model_shifts, on_manifold = self.checker.compare(state, source_config, target_config, probe_direction, target_value, action[i:i+1], 
                                                                                       target_state, )
            if on_manifold:
                if channel == 1:
                    ground_truth_theta = np.arctan2(delta_ground_truth[1], delta_ground_truth[0])
                    model_shift_theta = np.arctan2(delta_model_shifts[1], delta_model_shifts[0])
                    ground_truths.append(ground_truth_theta)
                    model_shifts.append(model_shift_theta)
                else:    
                    ground_truths.append(delta_ground_truth[channel])
                    model_shifts.append(delta_model_shifts[channel])
                
        gtruths_np = np.array(ground_truths, dtype = float)
        mshifts_np = np.array(model_shifts, dtype = float)
        n_used = len(gtruths_np)
        n_total = len(source_states)
        return transport_metrics(gtruths_np, mshifts_np, n_total)

    def report(self, source_states: torch.Tensor, source_config: dict, target_config: dict, channel: int,
               probe_direction: np.ndarray, action: torch.Tensor, calibration_pool: torch.Tensor,
               target_states: torch.Tensor | None = None, n_null: int = 30):

        oracle_targets = []
        for i, state in enumerate(source_states):
            target_state = None if target_states is None else target_states[i]
            _, gt_target = self.checker.validate(state, target_state, source_config, target_config, action[i])
            oracle_targets.append(gt_target)
        oracle_targets = torch.tensor(np.array(oracle_targets), dtype=torch.float32)

        dz, final_error, final_pred, latent = self.optimisation_bound(source_states, action, oracle_targets)
        cos_sim_dz = direction_alignment(dz, probe_direction)
        
        target_value = self.calibrate_target(calibration_pool, probe_direction)
        probe_result = self.direction_transport(source_states, source_config, target_config, channel,
                                                probe_direction, target_value, action, target_states)
        null_dicts = []
        for seed in range(n_null):
            rng = np.random.default_rng(seed)
            rand = rng.standard_normal(probe_direction.shape).astype(np.float32)
            rand = rand / np.linalg.norm(rand) * np.linalg.norm(probe_direction)
            rand_tv = self.calibrate_target(calibration_pool, rand)
            null_dicts.append(self.direction_transport(source_states, source_config, target_config, channel,
                                                       rand, rand_tv, action, target_states))

        null = null_summary(null_dicts)
        probe_frac = probe_result["fraction"]
        return {
            "ceiling_err": float(final_error),
            "probe_slope": probe_result["slope"],
            "probe_survival": probe_result["survival"],
            **null,
            "clears_null": bool(probe_result["slope"] > null["null_95"]),
            "dz_probe_cossim": cos_sim_dz,
            "probe_result": probe_result,
            "null_dicts": null_dicts,
        }


    def state_transfer(self, source_states, source_config, target_config, channel,
                   action, target_states=None, n_bins=4):
        if channel == 1 or channel == 0:
            raise NotImplementedError("Yet to be implemented for theta")
        oracle_targets, base_nexts = [], []
        for i, state in enumerate(source_states):
            target_state = None if target_states is None else target_states[i]
            src_next, gt_target = self.checker.validate(state, target_state, source_config, target_config, action[i])
            oracle_targets.append(gt_target)
            base_nexts.append(src_next)
        oracle_targets = torch.tensor(np.array(oracle_targets), dtype=torch.float32)
        base_nexts = np.array(base_nexts)

        true_effects = oracle_targets.numpy()[:, channel] - base_nexts[:, channel]
        dz, final_error, final_pred, latent = self.optimisation_bound(source_states, action, oracle_targets)
        encoded_states = self.checker.encode(source_states)
        dists, fracs, natives = [], [], []
        for i in range(len(source_states)):
            for j in range(len(source_states)):
                if abs(true_effects[j]) < 1e-6:
                    continue
                model_eff = self._apply_dz(source_states[j], dz[i], action[j: j+1], channel)
                frac = model_eff / true_effects[j]
                dist = float((encoded_states[i] - encoded_states[j]).norm())
                if i == j:
                    natives.append(frac)
                else:
                    dists.append(dist)
                    fracs.append(frac)
        dists = np.array(dists); fracs = np.array(fracs)
        edges = np.percentile(dists, np.linspace(0, 100, n_bins + 1))
        bins = []
        for k in range(n_bins):
            m = (dists >= edges[k]) & (dists <= edges[k+1])
            bins.append((float(edges[k]), float(edges[k+1]), float(fracs[m].mean())))

        return {
            "native_mean": float(np.mean(natives)),
            "bins": bins,
            "overall_transferred": float(fracs.mean()),
        }

    def _apply_dz(self, state, dz, action, channel):
        with torch.no_grad():
            z = self.checker.encode(state)
            base = self.checker.decode(self.checker.step(z, action))
            patched = self.checker.decode(self.checker.step(z + dz, action))
        return (patched - base)[channel].item()

    def calibrate_target(self, target_states, probe_directions):

        w_t = torch.from_numpy(probe_directions) if isinstance(probe_directions, np.ndarray) else probe_directions
        with torch.no_grad():
            z = self.checker.encode(target_states)
            return float((z @ w_t).mean())
        
