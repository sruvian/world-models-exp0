import numpy as np

class Environment():

    def __init__(self, gravity: float, mass1: float, length: float, dt: float, max_action: float, damping: float, seed: int, mass2: int = 0) -> None:
       pass

    def step(self, action: float)-> np.ndarray:
        
        raise NotImplementedError("Implement step")

    def reset(self)-> np.ndarray:
        
        raise NotImplementedError("Implement reset")

    def get_state(self) -> np.ndarray:
        
        raise NotImplementedError("Implement Get state")
    
    def get_metadata(self) -> dict:
        
        raise NotImplementedError("Update metadata")