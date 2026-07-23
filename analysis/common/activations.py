import torch
from models.wmodel import WorldModel, WorldModelVAE

def make_hook(acts, name):
    def fn(module, inp, out):
        acts[name] = out.detach()
    return fn

def collect_activations(model, states, actions):
    acts = {}
    spec = model.layer_spec() if hasattr(model, "layer_spec") else {}
    handles = [m.register_forward_hook(make_hook(acts, name)) for name, m in spec.items()]
    with torch.no_grad():
        c = model.encode_computational(states)
        if spec:
            t = model.step_computational(c, actions)
            _ = model.decode_computational(t)
    for handle in handles:
        handle.remove()
    if isinstance(c, tuple):
        acts["computational"] = c[1].detach()
        acts["computational_full"] = torch.cat(c, dim=-1).detach()
    else:
        acts["computational"] = c.detach()
    ts = model.layer_timesteps() if hasattr(model, "layer_timesteps") else {}
    ts.setdefault("computational", "current")
    return acts, ts