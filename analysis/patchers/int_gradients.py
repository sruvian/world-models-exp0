import torch

def integrated_gradients(output_func, z: torch.Tensor, baseline: torch.Tensor, num_steps: int = 50):
    alphas = torch.linspace(0.5/num_steps, 1-0.5/num_steps, num_steps).view(-1, 1, 1)
    path = baseline.unsqueeze(0) + alphas * (z.unsqueeze(0) - baseline.unsqueeze(0))
    func_input = path.reshape(-1, z.shape[-1]).detach().requires_grad_(True)
    out = output_func(func_input)
    grads = torch.autograd.grad(out.sum(), func_input)[0]
    grads = grads.reshape(num_steps, z.shape[0], z.shape[-1]).mean(0)
    int_gradient = (z - baseline) * grads
    
    return int_gradient.detach()


if __name__ == "__main__":
    pass