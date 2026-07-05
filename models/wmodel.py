from typing import Any

import torch
import torch.nn as nn
from .simplenn import VAEEncoder, DMDTransition

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
    def encode_computational(self, s):      return self.encode(s)
    def step_computational(self, z, a):     return self.step(z, a)
    def decode_computational(self, z):      return self.decode(z)
    def probe_representation(self, z):      return z

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

class WorldModelDMD(nn.Module):

    def __init__(self, model, state_dim: int, action_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = model(state_dim, hidden_dim, latent_dim)
        self.transition = DMDTransition(action_dim, latent_dim)
        self.decoder = model(latent_dim, hidden_dim, state_dim)

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.transition(z, a)

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
            "decoder_early":    self.decoder.net[1],
            "decoder_late":     self.decoder.net[3],
        }
    
    def layer_timesteps(self) -> dict:
        return{
            "encoder_early": "current", "encoder_late": "current", "latent": "current",
            "decoder_early": "next", "decoder_late": "next",
        }
    def encode_computational(self, s):      return self.encode(s)
    def step_computational(self, z, a):     return self.step(z, a)
    def decode_computational(self, z):      return self.decode(z)
    def probe_representation(self, z):      return z

class WorldModelGRU(nn.Module):

    def __init__(self, model, state_dim: int, action_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = model(state_dim, hidden_dim, hidden_dim)
        self.transition = nn.GRUCell(action_dim, hidden_dim)
        self.decoder = model(latent_dim, hidden_dim, state_dim)
        self.readout = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x):
        return self.encoder(x)
    
    def step(self, h, a):
        return self.transition(a, h)
    
    def probe_state(self, h):
        return self.readout(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, a):
        h = self.encode(x)
        h_next = self.step(h, a)
        z = self.probe_state(h_next)
        return self.decode(z), z
    
    def encode_computational(self, s):      return self.encode(s)
    def step_computational(self, h, a):     return self.step(h, a)
    def decode_computational(self, h):      return self.decode(self.probe_state(h))
    def probe_representation(self, h):      return self.probe_state(h)
    
class WorldModelRSSM(nn.Module):

    def __init__(self, model, state_dim: int, action_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior_net = model(hidden_dim, hidden_dim, 2 * latent_dim)
        self.obs_encoder = model(state_dim, hidden_dim, hidden_dim)
        self.posterior_net = model(hidden_dim + hidden_dim, hidden_dim, 2 * latent_dim)
        self.decoder = model(hidden_dim + latent_dim, hidden_dim, state_dim)

    def _split(self, params):
        mu, raw = params.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(raw) + 1e-4
        return mu, std

    def _sample(self, mu, std):
        return mu + std * torch.randn_like(std)
    
    def prior(self, h):
        mu, std = self._split(self.prior_net(h))
        z = self._sample(mu, std)
        return z, mu, std

    def posterior(self, h, obs):
        feat = self.obs_encoder(obs)
        mu, std = self._split(self.posterior_net(torch.cat([h, feat], dim=-1)))
        z = self._sample(mu, std)
        return z, mu, std
    
    def initial(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def encode(self, obs):
        h = torch.zeros(obs.shape[0], self.hidden_dim, device=obs.device)
        z, mu, std = self.posterior(h, obs)
        return h, mu

    def step(self, h, z, a):
        h = self.gru(torch.cat([z, a], dim=-1), h)
        z, mu, std = self.prior(h)
        return h, mu

    def probe_state(self, h, obs=None):
        if obs is not None:
            _, mu, _ = self.posterior(h, obs)
        else:
            _, mu, _ = self.prior(h)
        return mu

    def decode(self, h, z):
        return self.decoder(torch.cat([h, z], dim=-1))

    def encode_computational(self, s):      
        h, _ = self.encode(s);  return h
    def step_computational(self, h, a):     
        z = self.probe_state(h)
        h2 = self.gru(torch.cat([z, a], -1), h);  return h2
    def decode_computational(self, h):      
        return self.decode(h, self.probe_state(h))
    def probe_representation(self, h):      return self.probe_state(h)