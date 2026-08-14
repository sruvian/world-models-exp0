import torch
from models import WorldModel
from models.transfer import ProtocolAModel, ProtocolBModel
from models.wmodel import WorldModelDMD, WorldModelGRU, WorldModelVAE, WorldModelRSSM



class RolloutEngine:

    def __init__(self, model: WorldModel |ProtocolAModel| ProtocolBModel| WorldModelVAE| WorldModelDMD| WorldModelGRU| WorldModelRSSM,
                  loss: torch.nn.Module):

        self.model = model
        self.loss = loss


    def rollout(self, states: torch.Tensor, actions: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:

        
        if horizon < 1:
            raise ValueError("Horizon cannot be negative")
        
        total_loss = torch.zeros(1, device = states.device)
        self.model.eval()
        self.model.to(device = states.device)

        
        preds = []
        with torch.inference_mode():
            z = self.model.encode_computational(states[:, 0, :])
            for k in range(horizon):
                a_k = actions[:, k].unsqueeze(-1)
                z = self.model.step_computational(z, a_k)
                s_hat = self.model.decode_computational(z)
                preds.append(s_hat)
                total_loss += self.loss(s_hat, states[:, k + 1, :])
        preds = torch.stack(preds, dim=1)
        return preds, total_loss


    def get_latents(self, states: torch.Tensor, actions: torch.Tensor, horizon: int) -> torch.Tensor:
        with torch.inference_mode():
            z = self.model.encode(states[:, 0, :])
            latents = []
            for k in range(horizon):
                a_k = actions[:, k].unsqueeze(-1)
                z = self.model.step(z, a_k)
                latents.append(z)
        return torch.stack(latents, dim=1)
    def resonance_rollout(self, s0, drive_omega, dt, horizon, phase_dims=(3, 4)):
        self.model.eval()
        preds = []
        s = s0.clone()
        B = s.shape[0]
        with torch.inference_mode():
            for k in range(horizon):
                t = (k + 1) * dt
                comp = self.model.encode_computational(s)
                a0 = torch.zeros(B, 1, device=s.device)
                comp = self.model.step_computational(comp, a0)
                s_hat = self.model.decode_computational(comp)

                s_hat = s_hat.clone()
                ph = torch.as_tensor(drive_omega * t, device=s.device, dtype=s.dtype)
                s_hat[:, phase_dims[0]] = torch.cos(ph)
                s_hat[:, phase_dims[1]] = torch.sin(ph)

                preds.append(s_hat)
                s = s_hat
        return torch.stack(preds, dim=1)