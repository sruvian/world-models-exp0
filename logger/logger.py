import numpy as np
import time

class Logger:

    def __init__(self, model_name: str, optimizer_name: str, 
                 loss_name: str, lr: float, 
                 batch_size: int, steps: int,
                 gravity: float, length: float,
                 latent_dim: int) -> None:
        
        self.config = {
            "model_name": model_name,
            "optimizer_name": optimizer_name,
            "loss_name": loss_name, "lr": lr, 
            "batch_size": batch_size, "steps": steps,
            "gravity": gravity, "length": length,
            "latent_dim": latent_dim
        }
        self.start_time = 0.0
        self.total_time = 0.0
        self.step_log = []
        self.train_loss_arr = []
        self.val_loss_arr = []
    
    def log(self, train_loss: float, val_loss: float, step: int) -> None :
        self.step_log.append(step)
        self.train_loss_arr.append(train_loss)
        self.val_loss_arr.append(val_loss)
    def start(self):
        self.start_time = time.time()
        
    def finish(self):
        self.total_time = time.time() - self.start_time

    def save(self, path):
        np.savez(path,
                 train_loss=np.array(self.train_loss_arr),
                 val_loss=np.array(self.val_loss_arr),
                 log_steps=np.array(self.step_log),
                 total_time=self.total_time,
                 **self.config)