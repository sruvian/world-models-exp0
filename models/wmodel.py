import torch
import torch.nn as nn
from .simplenn import VAEEncoder

class WorldModel(nn.Module):
    def __init__(self, model, state_dim: int, action_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = model(state_dim, hidden_dim, latent_dim)
        self.transition = model(latent_dim + action_dim, hidden_dim, latent_dim)
        self.decoder = model(latent_dim, hidden_dim, state_dim)

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.transition(torch.cat([z, a], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(s)
        z_next = self.step(z, a)
        s_next = self.decode(z_next)
        return s_next, z
    
    def layer_spec(self) -> dict:
        return {
            "encoder_early":    self.encoder.net[1],
            "encoder_late":     self.encoder.net[3],
            "transition_early": self.transition.net[1],
            "transition_late":  self.transition.net[3],
            "decoder_early":    self.decoder.net[1],
            "decoder_late":     self.decoder.net[3],
        }
    def layer_timesteps(self) -> dict:
        return {
            "encoder_early": "current", "encoder_late": "current", "latent": "current",
            "transition_early": "next", "transition_late": "next",
            "decoder_early": "next", "decoder_late": "next",
        }

class WorldModelVAE(nn.Module):
    def __init__(self, model, state_dim: int, action_dim: int, 
                 hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = VAEEncoder(state_dim, hidden_dim, latent_dim)
        self.transition = model(latent_dim + action_dim, hidden_dim, latent_dim)
        self.decoder = model(latent_dim, hidden_dim, state_dim)

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(s)
        return mu

    def encode_with_kl(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        z, mu, log_var = self.encoder.sample(s)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=-1).mean()
        return z, kl

    def step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.transition(torch.cat([z, a], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, _, _ = self.encoder.sample(s)
        z_next = self.step(z, a)
        s_next = self.decode(z_next)
        return s_next, z
    
    def layer_spec(self) -> dict:
        return {
            "encoder_early":    self.encoder.net[1],
            "encoder_late":     self.encoder.net[3],
            "transition_early": self.transition.net[1],
            "transition_late":  self.transition.net[3],
            "decoder_early":    self.decoder.net[1],
            "decoder_late":     self.decoder.net[3],
        }
    
    def layer_timesteps(self) -> dict:
        return{
            "encoder_early": "current", "encoder_late": "current", "latent": "current",
            "transition_early": "next", "transition_late": "next",
            "decoder_early": "next", "decoder_late": "next",
        }