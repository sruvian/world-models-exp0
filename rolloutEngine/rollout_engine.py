import torch
from models import WorldModel
from models.transfer import ProtocolAModel, ProtocolBModel



class RolloutEngine:

    def __init__(self, model: WorldModel |ProtocolAModel| ProtocolBModel, loss: torch.nn.Module):

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
            z = self.model.encode(states[:, 0, :])
            for k in range(horizon):
                a_k = actions[:, k].unsqueeze(-1)
                z = self.model.step(z, a_k)
                s_hat = self.model.decode(z)
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