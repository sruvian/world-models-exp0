from .simplenn import SimpleNN
import torch



class ProtocolBModel(torch.nn.Module):
    def __init__(self, angular_encoder: torch.nn.Module, cartpole_encoder: torch.nn.Module, 
                 latent_angular: int, latent_B: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.angular_encoder = angular_encoder 
        self.cartpole_encoder = cartpole_encoder
        total_latent = latent_angular + latent_B
        self.transition = SimpleNN(total_latent + action_dim, hidden_dim, total_latent)
        self.decoder = SimpleNN(total_latent, hidden_dim, 5) 

    def encode(self, state):
        angular_state = state[:, :3]
        trans_state = state[:, 3:]
        z_ang = self.angular_encoder(angular_state)
        z_trans = self.cartpole_encoder(trans_state)
        return torch.cat([z_ang, z_trans], dim=-1)

    def step(self, z, a):
        return self.transition(torch.cat([z, a], dim=-1))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, state, action):
        z = self.encode(state)
        z_next = self.step(z, action)
        return self.decode(z_next)
    

class ProtocolAModel(torch.nn.Module):
    def __init__(self, unified_encoder: torch.nn.Module,
                 latent_angular: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = unified_encoder 
        self.transition = SimpleNN(latent_angular + action_dim, hidden_dim, latent_angular)
        self.decoder = SimpleNN(latent_angular, hidden_dim, 5)

    def encode(self, state):
        return self.encoder(state)

    def step(self, z, a):
        return self.transition(torch.cat([z, a], dim=-1))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, state, action):
        z = self.encode(state)
        z_next = self.step(z, action)
        return self.decode(z_next)