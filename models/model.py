from simplenn import SimpleNN
from wmodel import WorldModel

model_reg = {"SimpleNN" : SimpleNN}

def make_model(model_name, **kwargs):
    if model_name not in model_reg:
        raise ValueError("Model Unavailable")
    model = model_reg[model_name]

    world_model = WorldModel(model, **kwargs)

    return world_model