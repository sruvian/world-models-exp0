import torch

def integrated_gradients(output_func, z: torch.Tensor, baseline: torch.Tensor, num_steps: int = 50):
    z, baseline = z.flatten(), baseline.flatten()
    alphas = torch.linspace(0.5/ num_steps, 1- (0.5/num_steps), steps = num_steps)
    alphas = alphas.unsqueeze(1)
    func_input = (alphas*z + ((1-alphas)*baseline)).detach().requires_grad_(True)
    out = output_func(func_input)
    grad_outputs = torch.ones_like(out)
    gradients = torch.autograd.grad(out, func_input, grad_outputs=grad_outputs)[0]
    mean_gradients = gradients.mean(dim=0)
    int_gradient = (z - baseline) * mean_gradients
    
    return int_gradient.detach()


if __name__ == "__main__":
    pass