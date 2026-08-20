from typing import Callable
from functools import partial
import numpy as np
import torch
from logger.logger import Logger
from models import WorldModel
import tqdm

from models.transfer import ProtocolAModel, ProtocolBModel
from models.wmodel import WorldModelDMD, WorldModelVAE, WorldModelGRU, WorldModelRSSM


def split_gen(states: np.ndarray | torch.Tensor,
              actions: np.ndarray | torch.Tensor,
              rollout: int = 1, device: str = "cpu",
              windows_per_traj: int = 1,
              split_seed: int = 42, val_horizon: int| None = None, transient:int|None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(split_seed)
    if isinstance(states, torch.Tensor):
        states = states.numpy()
    if isinstance(actions, torch.Tensor):
        actions = actions.numpy()
    if transient is not None:
        states = states[:,transient:, :]
        actions = actions[:,transient:]
    N, T, state_dim = states.shape
    traj_perm = rng.permutation(N)
    train_idx = int(0.9 * N)
    train_traj, val_traj = traj_perm[:train_idx], traj_perm[train_idx:]
    def make_windows(traj_idx, window, win_per_traj):
        all_states, all_actions, all_nxt = [], [], []
        for i in traj_idx:
            hi = T - window
            if hi <= 0:
                raise ValueError(
                    f"Trajectory too short for windowing: T={T} (after transient), "
                    f"window={window}. Reduce transient or rollout_steps, or increase episode_time.")
            start_idxs = rng.integers(0, hi, size=win_per_traj)
            for s in start_idxs:
                all_states.append(states[i, s:s+window])
                all_actions.append(actions[i, s:s+window])
                all_nxt.append(states[i, s+1:s+window+1])
        return (np.array(all_states, np.float32), np.array(all_actions, np.float32), np.array(all_nxt, np.float32))
    
    train_s, train_a, train_ns = make_windows(train_traj, rollout, windows_per_traj)
    p = rng.permutation(train_s.shape[0])
    tr_s, tr_a, tr_nx = train_s[p], train_a[p], train_ns[p]
    vlen = val_horizon if val_horizon is not None else rollout
    va_s, va_a, va_nx = make_windows(val_traj, vlen, windows_per_traj)

   
    to_t = lambda x: torch.from_numpy(x).float().to(device)
    return (to_t(tr_s), to_t(tr_nx), to_t(tr_a),
            to_t(va_s), to_t(va_nx), to_t(va_a))

def stratified_split_gen(states: np.ndarray | torch.Tensor, 
              actions: np.ndarray | torch.Tensor, config_sizes: list[int],
              rollout:int = 1, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    N, T, state_dim = states.shape
    new_states, new_actions, new_nxt_states = [], [], []
    
    
    boundaries = np.cumsum([0] + config_sizes)
    samples_per_config = N // len(config_sizes)
    
    for i in range(len(config_sizes)):
        start = boundaries[i]
        end = boundaries[i + 1]
        config_n = end - start
        
        traj_idx = np.random.randint(start, end, size=samples_per_config)
        time_idx = np.random.randint(0, T - rollout, size=samples_per_config)
        
        for j in range(samples_per_config):
            t = traj_idx[j]
            s = time_idx[j]
            new_states.append(states[t, s:s+rollout])
            new_actions.append(actions[t, s:s+rollout])
            new_nxt_states.append(states[t, s+1:s+rollout+1])
    
    new_states = np.array(new_states, dtype=np.float32)
    new_actions = np.array(new_actions, dtype=np.float32)
    new_nxt_states = np.array(new_nxt_states, dtype=np.float32)
    
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

    print(f"[STRATIFIED] {len(config_sizes)} configs, {samples_per_config} samples each, total={M}")
    
    return train_s, train_s_next, train_a, val_s, val_s_next, val_a


def trainer(
        train_states: torch.Tensor,
        train_next_states: torch.Tensor,
        train_actions: torch.Tensor,
        val_states: torch.Tensor,
        val_next_states: torch.Tensor,
        val_actions: torch.Tensor,
        model: WorldModel | ProtocolAModel| ProtocolBModel| WorldModelVAE | WorldModelGRU| WorldModelDMD| WorldModelRSSM,
        logger: Logger,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.Module,
        batch_size: int,
        steps: int,
        rollout_decay: str,
        gamma: float,
        log_interval: int,
        beta: float = 1.0,
        reg: bool = False,
        lam: float = 1.0,
)-> WorldModel| ProtocolAModel| ProtocolBModel| WorldModelVAE| WorldModelGRU| WorldModelDMD| WorldModelRSSM:
    
    rollout_func = partial(lin_dec, gamma=gamma) if rollout_decay == "Linear" else partial(exp_dec, gamma=gamma)
    num_samples = train_states.shape[0]
    running_loss = 0
    pbar = tqdm.tqdm(range(steps), desc="Training")
    for step in pbar:
        
        idx = torch.randint(0, num_samples, (batch_size,), device = train_states.device)

        c_train_s = train_states[idx]
        c_train_n_s = train_next_states[idx]
        c_train_a = train_actions[idx]

        optimizer.zero_grad()
        if isinstance(model, WorldModelVAE):
            total_loss = rollout_loss_vae(model, c_train_s, c_train_a, c_train_n_s, loss_func, rollout_func, device = train_states.device, beta = beta)
        elif isinstance(model, WorldModelDMD):
            total_loss = rollout_loss_dmd(model, c_train_s, c_train_a, c_train_n_s, loss_func, rollout_func, device = train_states.device, reg= reg, lam = lam)
        elif isinstance(model, WorldModelGRU):
            total_loss = rollout_loss_gru(model, c_train_s, c_train_a, c_train_n_s, loss_func, rollout_func, device = train_states.device)
        elif isinstance(model, WorldModelRSSM):
            total_loss = rollout_loss_rssm(model, c_train_s, c_train_a, c_train_n_s, loss_func, rollout_func, device = train_states.device, beta= beta)
        else:
            total_loss = rollout_loss(model, c_train_s, c_train_a, c_train_n_s, loss_func, rollout_func, device = train_states.device)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += total_loss.item()

        if (step % log_interval == 0) or (step == steps-1):
            model.eval()
            with torch.inference_mode():
                val_idx = torch.randint(0, val_states.shape[0], (batch_size,), device=val_states.device)
                if isinstance(model, WorldModelVAE):
                    val_loss = rollout_loss_vae_val(model, val_states[val_idx], val_actions[val_idx], val_next_states[val_idx], loss_func, rollout_func, device = train_states.device, beta = beta)
                elif isinstance(model, WorldModelDMD):
                    val_loss = rollout_loss_dmd(model, val_states[val_idx], val_actions[val_idx], val_next_states[val_idx], loss_func, rollout_func, device = train_states.device, reg = reg, lam=lam)
                elif isinstance(model, WorldModelGRU):
                    val_loss = rollout_loss_gru(model, val_states[val_idx], val_actions[val_idx], val_next_states[val_idx], loss_func, rollout_func, device = train_states.device)
                elif isinstance(model, WorldModelRSSM):
                    val_loss = rollout_loss_rssm_val(model, val_states[val_idx], val_actions[val_idx], val_next_states[val_idx], loss_func, rollout_func, device = train_states.device, beta = beta)
                else:
                    val_loss = rollout_loss(model, val_states[val_idx], val_actions[val_idx], val_next_states[val_idx], loss_func, rollout_func, device = train_states.device)
            model.train()
            logger.log(running_loss / log_interval, val_loss.item(), step)
            pbar.set_postfix(train_loss=running_loss/log_interval, val_loss = val_loss.item())
            running_loss = 0            

    return model


def rollout_loss(model: WorldModel | ProtocolAModel| ProtocolBModel, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                  loss_func: torch.nn.Module, rollout_func: Callable, device: torch.device):
    K = states.shape[1]
    z = model.encode(states[:, 0, :])
    total_loss = torch.zeros(1, device=device)
    w = torch.tensor([rollout_func(K,k) for k in range(K)], dtype =torch.float64, device=device)
    w = w / w.mean()
    for k in range(K):
        a_k = actions[:, k].unsqueeze(-1)
        z = model.step(z, a_k)
        s_hat = model.decode(z)
        total_loss += w[k] * loss_func(s_hat, next_states[:, k, :])
    return total_loss

def rollout_loss_dmd(model: WorldModelDMD, states: torch.Tensor, actions: torch.Tensor,
                     next_states: torch.Tensor, loss_func: torch.nn.Module,
                     rollout_func: Callable, device: torch.device,
                     lam: float = 1.0, reg: bool = False):
    K = states.shape[1]
    z = model.encode(states[:, 0, :])
    total_loss = torch.zeros(1, device=device)
    w = torch.tensor([rollout_func(K,k) for k in range(K)], dtype =torch.float64, device=device)
    w = w / w.mean()
    for k in range(K):
        a_k = actions[:, k].unsqueeze(-1)
        z = model.step(z, a_k)
        s_hat = model.decode(z)
        total_loss += w[k] * loss_func(s_hat, next_states[:, k, :])

    penalty = torch.zeros(1, device=device)
    if reg:
        sigma_max = torch.linalg.matrix_norm(model.transition.A.weight, ord=2)
        penalty = lam * torch.relu(sigma_max - 1.0) ** 2

    return total_loss + penalty

def rollout_loss_vae(model: WorldModelVAE, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, 
                     loss_func: torch.nn.Module, rollout_func, beta, device):
    K = states.shape[1]

    z, kl = model.encode_with_kl(states[:, 0, :])
    total_loss = torch.zeros(1, device=device)

    for k in range(K):
        a_k = actions[:, k].unsqueeze(-1)
        z = model.step(z, a_k)
        s_hat = model.decode(z)
        weight = rollout_func(K, k)
        total_loss += weight * loss_func(s_hat, next_states[:, k, :])
    total_loss += beta * kl
    return total_loss

def rollout_loss_vae_val(model, states, actions, next_states,
                          loss_func, rollout_func, beta, device):
    K = states.shape[1]
    mu, log_var = model.encoder(states[:, 0, :])
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=-1).mean()
    z = mu
    total_loss = torch.zeros(1, device=device)
    w = torch.tensor([rollout_func(K,k) for k in range(K)], dtype =torch.float64, device=device)
    w = w / w.mean()
    for k in range(K):
        a_k = actions[:, k].unsqueeze(-1)
        z = model.step(z, a_k)
        s_hat = model.decode(z)
        total_loss += w[k] * loss_func(s_hat, next_states[:, k, :])
    total_loss += beta * kl
    return total_loss

def rollout_loss_gru(model: WorldModelGRU, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                     loss_func: torch.nn.Module, rollout_func: Callable, device: torch.device):
    K = states.shape[1]
    h = model.encode(states[:, 0, :])
    total_loss = torch.zeros(1, device=device)
    for k in range(K):
        a_k = actions[:, k].unsqueeze(-1)
        h = model.step(h, a_k)
        z = model.probe_state(h)
        s_hat = model.decode(z)
        weight = rollout_func(K, k)
        total_loss += weight * loss_func(s_hat, next_states[:, k, :])
    return total_loss

def kl_divergence(mu_q, std_q, mu_p, std_p):
    var_q, var_p = std_q**2, std_p**2
    kl = (torch.log(std_p / std_q)
          + (var_q + (mu_q - mu_p)**2) / (2 * var_p)
          - 0.5)
    return kl  
def kl_balanced(mu_q, std_q, mu_p, std_p, alpha=0.8):
    kl_prior_moves = kl_divergence(mu_q.detach(), std_q.detach(), mu_p, std_p)
    kl_post_moves  = kl_divergence(mu_q, std_q, mu_p.detach(), std_p.detach())
    return alpha * kl_prior_moves + (1 - alpha) * kl_post_moves


def rollout_loss_rssm(model, states, actions, next_states, loss_func, rollout_func,
                      beta, device, step=0, warmup_steps=10000, free_nats=0.1):
    K = states.shape[1]
    B = states.shape[0]
    h, z = model.initial(B, device)
    total_recon = torch.zeros(1, device=device)
    total_kl    = torch.zeros(1, device=device)

    beta_k = beta * min(1.0, step / warmup_steps)
    w = torch.tensor([rollout_func(K,k) for k in range(K)], dtype =torch.float64, device=device)
    w = w / w.mean()
    for k in range(K):
        a_k = actions[:, k].unsqueeze(-1)
        obs_next = next_states[:, k, :]
        h = model.gru(torch.cat([z, a_k], dim=-1), h)
        z_prior, mu_p, std_p = model.prior(h)
        z_post,  mu_q, std_q = model.posterior(h, obs_next)
        s_hat = model.decode(h, z_post)
        total_recon += w[k] * loss_func(s_hat, obs_next)
        kl = kl_balanced(mu_q, std_q, mu_p, std_p)
        kl = torch.clamp(kl, min=free_nats / model.latent_dim).sum(dim=-1).mean()
        total_kl += w[k] * kl
        z = z_post
    return total_recon + beta_k * total_kl

def rollout_loss_rssm_val(model, states, actions, next_states,
                          loss_func, rollout_func, beta, device):

    K = states.shape[1]

    B = states.shape[0]
    h, z = model.initial(B, device)
    z, mu_q, _ = model.posterior(h, states[:, 0, :])
    z = mu_q
    w = torch.tensor([rollout_func(K,k) for k in range(K)], dtype =torch.float64, device=device)
    w = w / w.mean()
    total = torch.zeros(1, device=device)
    for k in range(K):
        a_k = actions[:, k].unsqueeze(-1)
        h = model.gru(torch.cat([z, a_k], dim=-1), h)
        z_prior, mu_p, std_p = model.prior(h)
        z = mu_p
        s_hat = model.decode(h, z)
        total += w[k] * loss_func(s_hat, next_states[:, k, :])
    return total

def lin_dec(K, k, gamma):
    return (K - k) / K

def exp_dec(K, k, gamma):
    return gamma ** k