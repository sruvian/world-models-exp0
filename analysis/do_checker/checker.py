import torch
import numpy as np
from sim_envs import make_env

class DoChecker():

    def __init__(self, encode, step, decode, env, manifold_mult) -> None:
        
        self.env = env
        self.encode = encode
        self.step = step
        self.decode = decode
        self.manifold_mult = manifold_mult

    def validate(self, source_state: np.ndarray| torch.Tensor, target_state: torch.Tensor|None, source_config: dict, target_config:dict, action):
        seed = 0
        source_theta = np.arctan2(source_state[1], source_state[0])
        source_simulator = make_env(self.env, **source_config, seed = seed)
        source_simulator.reset()
        source_simulator.theta, source_simulator.theta_dot = source_theta, source_state[2]
        

        target_simulator = make_env(self.env, **target_config, seed = seed)
        target_simulator.reset()
        if target_state is None:
            target_simulator.theta, target_simulator.theta_dot = source_theta, source_state[2]
        else:
            target_theta = np.arctan2(target_state[1], target_state[0])
            target_simulator.theta, target_simulator.theta_dot = target_theta, target_state[2]
        source_next = source_simulator.step(action)
        target_next = target_simulator.step(action)

        return [source_next, target_next]

    def intervene(self, source_state, target_value, probe_direction, action, patch=True):
        with torch.no_grad():
            enc = self.encode(source_state)
            h, z = self._split(enc)
            if patch:
                w = torch.from_numpy(probe_direction)
                encoder_shift = ((target_value - (z @ w)) * w) / (w @ w)
            else:
                encoder_shift = torch.zeros_like(z)
            z_patched = z + encoder_shift
            on_manifold_dict = self._on_manifold(encoder_shift, z)
            transition = self.step(self._join(h, z_patched), action)
            dec = self.decode(transition)
        return {"decoder": dec, **on_manifold_dict}

    def compare(self, source_state: torch.Tensor, source_config: dict, target_config:dict,
                probe_direction: np.ndarray, target_value, action: torch.Tensor, target_state: torch.Tensor | None = None ):
        
        ground_truth_source, ground_truth_target = self.validate(source_state, target_state, source_config, target_config, action)
        state_2d = source_state.unsqueeze(0) if source_state.ndim == 1 else source_state
        patched_dict = self.intervene(state_2d, target_value, probe_direction, action)
        unpatched_dict = self.intervene(state_2d, target_value, probe_direction, action, False)

        if not patched_dict["on_manifold"]:
            print(f"Patch shift out of bounds! encoder_shift {patched_dict['shift_norm']:.2f} is {patched_dict['rel_shift']:.2f} times the latent!")
            delta_model = [np.nan]
        else: 
            delta_model = (patched_dict["decoder"] - unpatched_dict["decoder"]).squeeze(0).numpy()
        delta_ground_truth = ground_truth_target - ground_truth_source
        

        return delta_ground_truth, delta_model, patched_dict["on_manifold"]

    def _on_manifold(self, shift: torch.Tensor, z: torch.Tensor)->dict:   
        shift_norm = shift.norm().item()
        z_norm = z.norm().item()
        return {"on_manifold": shift_norm <= self.manifold_mult * z_norm,
                "shift_norm": shift_norm,
                "z_norm": z_norm,
                "rel_shift": shift_norm/ z_norm}

    def optimal_intervention(self, source_states, oracle_targets, action):
        dz_opts, dy_opts = [], []
        W = 1.0 / (torch.as_tensor(oracle_targets, dtype=torch.float32).std(dim=0) + 1e-6)

        for i in range(len(source_states)):
            enc = self.encode(source_states[i:i+1])
            h, z = self._split(enc)
            z = z.detach().requires_grad_(True)
            a = action[i:i+1]
            def f(zz):
                return self.decode(self.step(self._join(h, zz), a))
            y = f(z)
            J = torch.autograd.functional.jacobian(f, z).reshape(y.shape[-1], z.shape[-1])
            J_W = J *W[:, None]
            residual = torch.as_tensor(oracle_targets[i], dtype=torch.float32) - y.squeeze(0)
            r_W = residual*W
            # dz_opt = torch.linalg.pinv(J) @ residual
            try:
                dz_opt = torch.linalg.pinv(J_W) @ r_W
            except RuntimeError:
                dz_opt = torch.full_like(z.squeeze(0), float("nan"))
            dz_opts.append(dz_opt)
            dy_opts.append(J @ dz_opt)
        return torch.stack(dz_opts), torch.stack(dy_opts)
    
    def _split(self, enc):
        if isinstance(enc, tuple):
            return enc[0], enc[1]
        return None, enc

    def _join(self, h, z):
        return (h, z) if h is not None else z