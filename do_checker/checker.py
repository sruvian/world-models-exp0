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

        source_theta = np.arctan2(source_state[1], source_state[0])
        source_simulator = make_env(self.env, **source_config)
        source_simulator.reset()
        source_simulator.theta, source_simulator.theta_dot = source_theta, source_state[2]
        

        target_simulator = make_env(self.env, **target_config)
        target_simulator.reset()
        if target_state is None:
            target_simulator.theta, target_simulator.theta_dot = source_theta, source_state[2]
        else:
            target_theta = np.arctan2(target_state[1], target_state[0])
            target_simulator.theta, target_simulator.theta_dot = target_theta, target_state[2]
        source_next = source_simulator.step(action)
        target_next = target_simulator.step(action)

        return [source_next, target_next]

    def intervene(self, source_state: torch.Tensor, target_value, probe_direction: np.ndarray, action: torch.Tensor, patch: bool = True) -> dict:
        with torch.no_grad():
            enc = self.encode(source_state)
            if patch:
                probe_direction_tensor = torch.from_numpy(probe_direction)
                encoder_shift: torch.Tensor = ((target_value - (enc @ probe_direction_tensor)) * probe_direction_tensor) / (probe_direction_tensor @ probe_direction_tensor)  
            else: 
                encoder_shift = torch.Tensor([0.0])
            encoder_patched = enc + encoder_shift
            on_manifold_dict = self._on_manifold(encoder_shift, enc)
            transition = self.step(encoder_patched, action)
            dec = self.decode(transition)
        return {"decoder": dec, **on_manifold_dict}

    def compare(self, source_state: torch.Tensor, source_config: dict, target_config:dict,
                probe_direction: np.ndarray, target_value, action: torch.Tensor, target_state: torch.Tensor | None = None ):
        
        ground_truth_source, ground_truth_target = self.validate(source_state, target_state, source_config, target_config, action)

        patched_dict = self.intervene(source_state, target_value, probe_direction, action)
        unpatched_dict = self.intervene(source_state, target_value, probe_direction, action, False)

        if not patched_dict["on_manifold"]:
            print(f"Patch shift out of bounds! encoder_shift {patched_dict['shift_norm']:.2f} is {patched_dict['rel_shift']:.2f} times the latent!")
            delta_model = [np.nan]
        else: 
            model_patched = patched_dict["decoder"]
            model_unpatched = unpatched_dict["decoder"]
            delta_model = (model_patched - model_unpatched).numpy()

        delta_ground_truth = ground_truth_target - ground_truth_source
        

        return delta_ground_truth, delta_model, patched_dict["on_manifold"]

    def _on_manifold(self, shift: torch.Tensor, z: torch.Tensor)->dict:   
        shift_norm = shift.norm().item()
        z_norm = z.norm().item()
        return {"on_manifold": shift_norm <= self.manifold_mult * z_norm,
                "shift_norm": shift_norm,
                "z_norm": z_norm,
                "rel_shift": shift_norm/ z_norm}