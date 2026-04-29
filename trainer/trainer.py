import numpy as np
import torch
from logger.logger import Logger



def split_gen(states: np.ndarray | torch.Tensor, actions: np.ndarray | torch.Tensor, delta_states: bool = False, flatten: bool = True, rollout:int = 1, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    N = states.shape[0]
    perm = np.random.permutation(N)
    train_idx = int(0.9 * N)
    
    states = states[perm]
    actions = actions[perm]
    train_states = states[:train_idx]
    train_actions = actions[:train_idx]

    val_states = states[train_idx:]
    val_actions = actions[train_idx:]

    train_next_states = train_states[:, 1:, :]
    train_states = train_states[:, :-1, :]

    val_next_states = val_states[:, 1:, :]
    val_states = val_states[:, :-1, :]

    if delta_states:
        train_next_states = train_next_states - train_states
        val_next_states = val_next_states - val_states

    if flatten:
        train_states, train_next_states, train_actions = train_states.reshape(-1, train_states.shape[-1]), train_next_states.reshape(-1, train_next_states.shape[-1]), train_actions.reshape(-1, 1)
        val_states, val_next_states, val_actions = val_states.reshape(-1, val_states.shape[-1]), val_next_states.reshape(-1, val_next_states.shape[-1]), val_actions.reshape(-1, 1)

    

    train_states, train_next_states, train_actions = torch.from_numpy(train_states).float().to(device= device), torch.from_numpy(train_next_states).float().to(device= device), torch.from_numpy(train_actions).float().to(device= device)
    val_states, val_next_states, val_actions = torch.from_numpy(val_states).float().to(device= device), torch.from_numpy(val_next_states).float().to(device= device), torch.from_numpy(val_actions).float().to(device= device)
    

    return train_states, train_next_states, train_actions, val_states, val_next_states, val_actions



def trainer(
        train_states: torch.Tensor,
        train_next_states: torch.Tensor,
        train_actions: torch.Tensor,
        val_states: torch.Tensor,
        val_next_states: torch.Tensor,
        val_actions: torch.Tensor,
        model: torch.nn.Module,
        logger: Logger,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.Module,
        batch_size: int,
        steps: int,
        log_interval: int
)-> torch.nn.Module:
    
    num_samples = train_states.shape[0]
    running_loss = 0
    
    for step in range(steps):
        
        idx = torch.randint(0, num_samples, (batch_size,), device = train_states.device)

        c_train_s = train_states[idx]
        c_train_n_s = train_next_states[idx]
        c_train_a = train_actions[idx]

        model_features = torch.cat([c_train_s, c_train_a], dim =  1)

        optimizer.zero_grad()
        output = model(model_features)        
        loss = loss_func(output, c_train_n_s)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (step % log_interval == 0) or (step == steps-1):
            model.eval()
            with torch.no_grad():
                val_hat = model(torch.cat([val_states, val_actions], dim = 1))
                val_loss = loss_func(val_next_states, val_hat)
                model.train()
                logger.log(running_loss / log_interval, val_loss.item())

                running_loss = 0            

    return model


