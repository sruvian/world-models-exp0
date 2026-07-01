from typing import Any
import torch
import torch.nn as nn


class SimpleNN(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor :

        out = self.net(x)

        return out
    

class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_head(h), self.log_var_head(h)

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.forward(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var


class DMDTransition(nn.Module):

    def __init__(self, action_dim:int, latent_dim: int) -> None:
        super().__init__()

        self.A = nn.Linear(latent_dim, latent_dim, bias = False)
        self.control = nn.Linear(action_dim, latent_dim, bias = False)

    def forward(self, z, a):
        return self.A(z) + self.control(a)
