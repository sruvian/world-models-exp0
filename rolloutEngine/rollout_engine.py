import torch
from models.simplenn import SimpleNN



class RolloutEngine:

    def __init__(self, model: torch.nn.Module, loss: torch.nn.Module):

        self.model = model
        self.loss = loss


    def rollout(self, states: torch.Tensor, actions: torch.Tensor, horizon: int|None = None) -> torch.Tensor:

        self.model.eval()
        horizons = []
        if horizon is None:
            horizon = actions.shape[1]

        with torch.no_grad():
            pred = states[:, 0, :]

            for i in range(horizon): 
                pred_action = torch.cat([pred, actions[:, i+1].unsqueeze(1)], dim =1 )
                pred = self.model(pred_action)
                horizons.append(pred)
        
        return torch.stack(horizons, dim = 1)
    
    def rollout_loss(self, true_states: torch.Tensor, rollout_states: torch.Tensor, horizon: int)-> torch.Tensor:

        return self.loss(true_states[:, 1: horizon+1, :], rollout_states)