from typing import Callable
from functools import partial
import numpy as np
import torch
from logger.logger import Logger
from models import WorldModel



def split_gen(states: np.ndarray | torch.Tensor, 
              actions: np.ndarray | torch.Tensor, 
              rollout:int = 1, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    N, T, state_dim = states.shape

    new_states, new_actions, new_nxt_states = [], [], []
    start_idx = np.random.randint(0, T - rollout, size = N)
    new_states = np.stack([states[i, start_idx[i]:start_idx[i]+rollout] for i in range(N)])
    new_actions = np.stack([actions[i, start_idx[i]:start_idx[i]+rollout] for i in range(N)])
    new_nxt_states = np.stack([states[i, start_idx[i]+1:start_idx[i]+rollout+1] for i in range(N)])

    new_states = np.array(new_states, dtype= np.float32)
    new_actions = np.array(new_actions, dtype= np.float32)
    new_nxt_states = np.array(new_nxt_states, dtype= np.float32)



    M = new_states.shape[0]
    perm = np.random.permutation(M)
    train_idx = int(0.9 * M)

    new_states = new_states[perm]
    new_actions = new_actions[perm]
    new_nxt_states = new_nxt_states[perm]

    train_s      = torch.from_numpy(new_states[:train_idx]).float().to(device)
    train_a      = torch.from_numpy(new_actions[:train_idx]).float().to(device)
    train_s_next = torch.from_numpy(new_nxt_states[:train_idx]).float().to(device)
    val_s        = torch.from_numpy(new_states[train_idx:]).float().to(device)
    val_a        = torch.from_numpy(new_actions[train_idx:]).float().to(device)
    val_s_next   = torch.from_numpy(new_nxt_states[train_idx:]).float().to(device)

    return train_s, train_s_next, train_a, val_s, val_s_next, val_a


def trainer(
        train_states: torch.Tensor,
        train_next_states: torch.Tensor,
        train_actions: torch.Tensor,
        val_states: torch.Tensor,
        val_next_states: torch.Tensor,
        val_actions: torch.Tensor,
        model: WorldModel,
        logger: Logger,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.Module,
        batch_size: int,
        steps: int,
        rollout_decay: str,
        gamma: float,
        log_interval: int
)-> torch.nn.Module:
    
    rollout_func = partial(lin_dec, gamma=gamma) if rollout_decay == "Linear" else partial(exp_dec, gamma=gamma)
    num_samples = train_states.shape[0]
    running_loss = 0
    
    for step in range(steps):
        
        idx = torch.randint(0, num_samples, (batch_size,), device = train_states.device)

        c_train_s = train_states[idx]
        c_train_n_s = train_next_states[idx]
        c_train_a = train_actions[idx]

        optimizer.zero_grad()

        K = c_train_s.shape[1]

        total_loss = rollout_loss(model, c_train_s, c_train_a, c_train_n_s, loss_func, rollout_func, device = train_states.device)

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

        if (step % log_interval == 0) or (step == steps-1):
            model.eval()
            with torch.no_grad():
                val_idx = torch.randint(0, val_states.shape[0], (batch_size,), device=val_states.device)
                val_loss = rollout_loss(model, val_states[val_idx], val_actions[val_idx], val_next_states[val_idx], loss_func, rollout_func, device = train_states.device)
                model.train()
                logger.log(running_loss / log_interval, val_loss.item())

                running_loss = 0            

    return model


def rollout_loss(model: WorldModel, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, loss_func: torch.nn.Module, rollout_func: Callable, device: torch.device):
    K = states.shape[1]
    z = model.encode(states[:, 0, :])
    total_loss = torch.zeros(1, device=device)
    for k in range(K):
        a_k = actions[:, k].unsqueeze(-1)
        z = model.step(z, a_k)
        s_hat = model.decode(z)
        weight = rollout_func(K, k)
        total_loss += weight * loss_func(s_hat, next_states[:, k, :])
    return total_loss


def lin_dec(K, k, gamma):
    return (K - k) / K

def exp_dec(K, k, gamma):
    return gamma ** k