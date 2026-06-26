import torch
import numpy as np
from sim_envs import make_env
from utils import parse_model



class DoChecker():

    def __init__(self, encode, step, decode, env) -> None:
        
        self.env = env
        self.encode = encode
        self.step = step
        self.decode = decode

    def validate(self, source_state: np.ndarray| torch.Tensor, source_config: dict, target_config:dict, action):

        source_theta = np.arctan2(source_state[1], source_state[0])
        source_simulator = make_env(self.env, **source_config)
        source_simulator.reset()
        source_simulator.theta, source_simulator.theta_dot = source_theta, source_state[2]
        

        target_simulator = make_env(self.env, **target_config)
        target_simulator.reset()
        target_simulator.theta, target_simulator.theta_dot = source_theta, source_state[2]

        source_next = source_simulator.step(action)
        target_next = target_simulator.step(action)

        return [source_next, target_next]

    def intervene(self, source_state, target_value, probe_direction, action, patch: bool = True):
        with torch.no_grad():
            enc = self.encode(source_state)
            if patch:
                probe_direction = torch.from_numpy(probe_direction)
                encoder_shift = ((target_value - (enc @ probe_direction)) * probe_direction) / (probe_direction @ probe_direction)
                
            else: 
                encoder_shift = 0
            encoder_patched = enc + encoder_shift
            transition = self.step(encoder_patched, action)
            dec = self.decode(transition)
        return dec

    def compare(self, source_state: torch.Tensor| np.ndarray, source_config: dict, target_config:dict, probe_direction, target_value, action):
        
        ground_truth_source, ground_truth_target = self.validate(source_state, source_config, target_config, action)

        model_patched = self.intervene(source_state, target_value, probe_direction, action)
        model_unpatched = self.intervene(source_state, target_value, probe_direction, action, False)

        delta_ground_truth = ground_truth_target - ground_truth_source
        delta_model = (model_patched - model_unpatched).numpy()

        return delta_ground_truth, delta_model
