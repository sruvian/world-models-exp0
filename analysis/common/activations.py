import torch
from models.wmodel import WorldModel, WorldModelVAE

def make_hook(acts, name):
    def fn(module, inp, out):
        acts[name] = out.detach()
    return fn

def collect_activations(model, states, actions):
    spec = model.layer_spec()
    acts = {}
    handles = [m.register_forward_hook(make_hook(acts, name)) for name, m in spec.items()]
    with torch.no_grad():
        c = model.encode_computational(states)
        t = model.step_computational(c, actions)
        _ = model.decode_computational(t)
    for handle in handles:
        handle.remove()
    acts["computational"] = c.detach()
    return acts, model.layer_timesteps()